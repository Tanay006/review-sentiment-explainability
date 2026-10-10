"""Sentiment engine: NLP preprocessing, models, decision rule, explainability.

Heavy imports (torch / transformers) are lazy so the baseline works without them.
"""
from __future__ import annotations

import html as _html
import re
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

LABELS = ["Negative", "Neutral", "Positive"]  # column order everywhere: 0, 1, 2
THRESHOLD = 0.60          # P(pos) or P(neg) must exceed this, otherwise Neutral
CONTRAST_WEIGHT = 0.70    # weight of the main (post-contrast) clause in the blend
NEG_SCOPE = 4             # max content tokens a negator reaches

NEGATORS = {"not", "never", "no", "neither", "nor", "cannot"}
CONTRASTS = ["but", "however", "although", "though", "unfortunately", "except", "yet"]
_PUNCT = set(".,!?;:")
_STOP = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "for", "with", "is", "are",
    "was", "were", "be", "been", "am", "it", "this", "that", "i", "me", "my", "we", "you",
    "they", "he", "she", "its", "as", "by", "from", "so", "do", "did", "does", "have", "has",
    "had", "will", "would", "than", "then", "there", "these", "those", "just",
}  # negators and contrast markers are deliberately NOT in here


# ----------------------------------------------------------------------------- NLP
_CONTRACTION_RULES = [
    (re.compile(p, re.I), r)
    for p, r in [
        (r"\bcan't\b", "cannot"),
        (r"\bwon't\b", "will not"),
        (r"\bshan't\b", "shall not"),
        (r"\bain't\b", "is not"),
        (r"n't\b", " not"),
        (r"\b(it|that|there|he|she|what|here|who)'s\b", r"\1 is"),
        (r"'re\b", " are"),
        (r"'ve\b", " have"),
        (r"'ll\b", " will"),
        (r"'m\b", " am"),
        (r"'d\b", " would"),
    ]
]
_CONTRAST_RE = re.compile(r"\b(" + "|".join(CONTRASTS) + r")\b", re.I)


def expand_contractions(text: str) -> str:
    t = str(text).replace("\u2019", "'").replace("\u2018", "'").replace("`", "'")
    for pat, rep in _CONTRACTION_RULES:
        t = pat.sub(rep, t)
    return t


def split_clauses(text: str):
    """Return (lead, marker, main). main is the clause after the LAST contrast marker."""
    parts = _CONTRAST_RE.split(text)  # [lead, marker, seg, marker, seg, ...]
    if len(parts) < 3:
        return text, None, None
    main, marker, lead = parts[-1], parts[-2], "".join(parts[:-2])
    if len(re.findall(r"\w+", main)) < 2:
        return text, None, None
    return lead, marker.lower(), main


def _tag_negation(segment: str) -> list[str]:
    out, scope = [], 0
    for tok in re.findall(r"[a-z]+|[.,!?;:]", segment):
        if tok in _PUNCT:
            scope = 0
        elif tok in NEGATORS:
            out.append(tok)
            scope = NEG_SCOPE
        elif tok in CONTRASTS:
            out.append(tok)
            scope = 0
        elif tok in _STOP or len(tok) < 2:
            continue
        else:
            out.append(f"not_{tok}" if scope > 0 else tok)
            scope = max(0, scope - 1)
    return out


def preprocess_baseline(text: str) -> str:
    """Contractions -> negation scope tagging -> contrast-clause up-weighting."""
    t = re.sub(r"https?://\S+", " ", expand_contractions(str(text).lower()))
    lead, marker, main = split_clauses(t)
    if main is None:
        return " ".join(_tag_negation(t))
    toks = _tag_negation(lead) + [marker] + _tag_negation(main) * 2  # verdict clause counts double
    return " ".join(toks)


def rating_to_label(r: float) -> int:
    return 0 if r <= 2.5 else (1 if r < 3.5 else 2)


def decide(probs) -> np.ndarray:
    p = np.asarray(probs)
    out = np.full(len(p), "Neutral", dtype=object)
    out[p[:, 0] > THRESHOLD] = "Negative"
    out[p[:, 2] > THRESHOLD] = "Positive"
    return out


# --------------------------------------------------------------------------- models
class SentimentModel:
    chunk = 64

    def _predict_chunk(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, texts: Sequence[str], progress_cb: Optional[Callable[[float], None]] = None):
        texts = [str(t) for t in texts]
        n = len(texts)
        out = np.zeros((n, 3))
        for s in range(0, n, self.chunk):
            out[s:s + self.chunk] = self._predict_chunk(texts[s:s + self.chunk])
            if progress_cb:
                progress_cb(min(1.0, (s + self.chunk) / n))
        return out


class BaselineModel(SentimentModel):
    """Contrast-aware TF-IDF (1-2 grams) + regularised multinomial logistic regression."""

    chunk = 2000

    def __init__(self):
        self.pipe: Optional[Pipeline] = None
        self.metrics: dict = {}

    @staticmethod
    def _make(n: int) -> Pipeline:
        vec = TfidfVectorizer(
            preprocessor=preprocess_baseline, tokenizer=str.split, token_pattern=None,
            lowercase=False, ngram_range=(1, 2), min_df=2 if n >= 200 else 1,
            max_features=60000, sublinear_tf=True,
        )
        clf = LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced")
        return Pipeline([("tfidf", vec), ("clf", clf)])

    def fit(self, texts, y, evaluate: bool = True):
        texts, y = [str(t) for t in texts], np.asarray(y, dtype=int)
        counts = np.bincount(y, minlength=3)
        if (counts > 0).sum() < 2:
            raise ValueError("Training data needs at least two sentiment classes.")
        if evaluate and len(y) >= 50:
            strat = y if counts[counts > 0].min() >= 2 else None
            Xtr, Xte, ytr, yte = train_test_split(texts, y, test_size=0.2, random_state=42, stratify=strat)
            pred = self._make(len(Xtr)).fit(Xtr, ytr).predict(Xte)
            self.metrics = {
                "accuracy": float(accuracy_score(yte, pred)),
                "macro_f1": float(f1_score(yte, pred, average="macro")),
                "n_train": len(Xtr), "n_test": len(Xte),
            }
        self.pipe = self._make(len(texts)).fit(texts, y)
        return self

    def _predict_chunk(self, texts):
        p = self.pipe.predict_proba(texts)
        out = np.zeros((len(texts), 3))
        for j, c in enumerate(self.pipe.named_steps["clf"].classes_):
            out[:, int(c)] = p[:, j]
        return out


class TransformerModel(SentimentModel):
    """Pre-trained contextual model with contrast-clause weighting on top.

    Contractions are expanded, but negation tokens are NOT tagged: contextual models
    handle negation natively and tagged tokens would be out-of-vocabulary noise for them.
    """

    chunk = 32

    def __init__(self, name: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).eval()
        id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.col: dict[int, int] = {}
        for i, lab in id2label.items():
            if lab.startswith("neg"):
                self.col[i] = 0
            elif lab.startswith("neu"):
                self.col[i] = 1
            elif lab.startswith("pos"):
                self.col[i] = 2
        if len(self.col) != len(id2label):  # generic LABEL_n heads: assume neg[,neu],pos order
            self.col = {0: 0, 1: 2} if len(id2label) == 2 else {0: 0, 1: 1, 2: 2}

    def _raw(self, texts: list[str]) -> np.ndarray:
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with self._torch.no_grad():
            p = self._torch.softmax(self.model(**enc).logits, dim=-1).numpy()
        out = np.zeros((len(texts), 3))
        for i, c in self.col.items():
            out[:, c] = p[:, i]
        return out

    def _predict_chunk(self, texts):
        prep = [expand_contractions(t).strip() or "." for t in texts]
        full = self._raw(prep)
        splits = [split_clauses(t) for t in prep]
        idx = [i for i, s in enumerate(splits) if s[2] is not None]
        if idx:
            main = self._raw([splits[i][2].strip() for i in idx])
            full[idx] = (1 - CONTRAST_WEIGHT) * full[idx] + CONTRAST_WEIGHT * main
        return full


# ----------------------------------------------------------------------- explainers
def global_features(pipe: Pipeline, k: int = 15) -> dict[str, pd.DataFrame]:
    """Top-k terms per class from the TF-IDF + LR coefficients."""
    names = np.asarray(pipe.named_steps["tfidf"].get_feature_names_out())
    clf = pipe.named_steps["clf"]
    classes, coef = [int(c) for c in clf.classes_], clf.coef_
    if len(classes) == 2:
        coef = np.vstack([-coef[0], coef[0]])
    out = {lab: pd.DataFrame({"Term": [], "Weight": []}) for lab in LABELS}
    for row, c in zip(coef, classes):
        top = np.argsort(row)[::-1][:k]
        out[LABELS[c]] = pd.DataFrame({"Term": names[top], "Weight": np.round(row[top], 3)})
    return out


_LIME_SPLIT = r"[^\w'\u2019]+"  # keep "didn't" as ONE token for LIME


def explain_lime(predict_fn, text: str, num_samples: int = 500, num_features: int = 12, seed: int = 42):
    from lime.lime_text import LimeTextExplainer

    explainer = LimeTextExplainer(class_names=LABELS, random_state=seed, split_expression=_LIME_SPLIT)
    return explainer.explain_instance(
        text, predict_fn, num_features=num_features, num_samples=num_samples, labels=(0, 1, 2)
    )


def _weights(exp):
    return [dict(exp.as_list(label=i)) for i in range(3)]


def highlight_html(text: str, exp) -> str:
    """Green = pulls toward Positive, red = pulls toward Negative, gray = ~no pull."""
    wn, _, wp = _weights(exp)
    pol = {w: wp.get(w, 0.0) - wn.get(w, 0.0) for w in set(wn) | set(wp)}
    scale = max([abs(v) for v in pol.values()] or [1.0]) or 1.0
    parts = []
    for m in re.finditer(r"[\w'\u2019]+|[^\w'\u2019]+", text):
        tok = m.group(0)
        if tok not in pol:
            parts.append(_html.escape(tok))
            continue
        v = pol[tok]
        tip = f"Positive {wp.get(tok, 0):+.3f} | Negative {wn.get(tok, 0):+.3f}"
        if abs(v) < 0.01:
            bg = "rgba(128,128,128,0.30)"
        else:
            a = min(0.85, 0.2 + 0.65 * abs(v) / scale)
            bg = f"rgba(46,160,67,{a:.2f})" if v > 0 else f"rgba(218,54,51,{a:.2f})"
        parts.append(
            f'<span title="{tip}" style="background:{bg};padding:1px 4px;border-radius:4px">'
            f"{_html.escape(tok)}</span>"
        )
    return "".join(parts)


def lime_table(exp) -> pd.DataFrame:
    wn, wu, wp = _weights(exp)
    words = set(wn) | set(wu) | set(wp)
    if not words:
        return pd.DataFrame(columns=["Word", "Negative", "Neutral", "Positive"])
    df = pd.DataFrame(
        [{"Word": w, "Negative": wn.get(w, 0.0), "Neutral": wu.get(w, 0.0), "Positive": wp.get(w, 0.0)}
         for w in words]
    )
    order = (df["Positive"] - df["Negative"]).abs().sort_values(ascending=False).index
    return df.loc[order].round(4).reset_index(drop=True)
