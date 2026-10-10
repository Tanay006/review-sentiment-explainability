"""Review Sentiment & Explainability — Streamlit app."""
import io
import time

import numpy as np
import pandas as pd
import streamlit as st

import sentiment_engine as se

st.set_page_config(page_title="Review Sentiment", page_icon="💬", layout="wide")
st.markdown(
    """<style>
    .block-container{padding-top:2rem;max-width:1200px}
    .lime-box{padding:14px 16px;border:1px solid #2b303b;border-radius:10px;
              line-height:2.1;font-size:1.05rem;white-space:pre-wrap}
    </style>""",
    unsafe_allow_html=True,
)

MODEL_OPTIONS = {
    "RoBERTa 3-class (cardiffnlp)": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "DistilBERT 2-class (SST-2)": "distilbert-base-uncased-finetuned-sst-2-english",
    "Baseline: TF-IDF + Logistic Regression": "baseline",
}
EMOJI = {"Positive": "🟢", "Neutral": "⚪", "Negative": "🔴"}
TEXT_HINTS = ["review text", "review_text", "reviewtext", "review", "text", "comment", "comments",
              "content", "body", "feedback"]
RATING_HINTS = ["rating", "ratings", "stars", "star rating", "score", "overall"]


# ------------------------------------------------------------------------- helpers
@st.cache_data(show_spinner=False)
def sheet_names(name: str, data: bytes):
    if name.lower().endswith(".csv"):
        return []
    return pd.ExcelFile(io.BytesIO(data), engine="openpyxl").sheet_names


@st.cache_data(show_spinner="Reading file…")
def read_table(name: str, data: bytes, sheet=None) -> pd.DataFrame:
    if name.lower().endswith(".csv"):
        try:
            return pd.read_csv(io.BytesIO(data))
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(data), encoding="latin-1")
    return pd.read_excel(io.BytesIO(data), sheet_name=sheet or 0, engine="openpyxl")


def detect_text_col(df: pd.DataFrame) -> str:
    low = {str(c).lower().strip(): c for c in df.columns}
    for hint in TEXT_HINTS:
        if hint in low:
            return low[hint]
    obj = [c for c in df.columns if df[c].dtype == object]
    return max(obj, key=lambda c: df[c].astype(str).str.len().mean()) if obj else df.columns[0]


def detect_rating_col(df: pd.DataFrame):
    low = {str(c).lower().strip(): c for c in df.columns}
    for hint in RATING_HINTS:
        if hint in low:
            return low[hint]
    return None


def to_xlsx(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Results")
    return buf.getvalue()


@st.cache_resource(show_spinner="Loading model (first run downloads weights)…")
def load_transformer(name: str):
    return se.TransformerModel(name)


def send_row(text: str):
    st.session_state["inspect_text"] = text


@st.dialog("Top global features", width="large")
def show_global(model_key: str):
    if model_key == "baseline":
        base = st.session_state.get("baseline")
        if base is None:
            st.info("Train the baseline first: upload a file with a rating column and click Classify.")
            return
        pipe, note = base.pipe, "Learned logistic-regression coefficients (higher = stronger pull)."
    else:
        res = st.session_state.get("results")
        if not res:
            st.info("Classify a file first. Features are derived from your data.")
            return
        if st.session_state.get("surrogate_id") != res["id"]:
            with st.spinner("Fitting surrogate…"):
                labels = [se.LABELS.index(s) for s in res["df"]["Predicted Sentiment"]]
                sur = se.BaselineModel().fit(res["df"][res["text_col"]], labels, evaluate=False)
            st.session_state["surrogate"], st.session_state["surrogate_id"] = sur, res["id"]
        pipe = st.session_state["surrogate"].pipe
        note = ("A TF-IDF + LR surrogate fitted to the transformer's predictions on your uploaded data. "
                "`not_x` means x appeared inside a negation scope.")
    st.caption(note)
    feats = se.global_features(pipe, 15)
    for col, lab in zip(st.columns(3), ["Positive", "Neutral", "Negative"]):
        col.markdown(f"**{EMOJI[lab]} {lab}**")
        col.dataframe(feats[lab], hide_index=True, use_container_width=True)


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    model_label = st.selectbox("Model", list(MODEL_OPTIONS))
    model_key = MODEL_OPTIONS[model_label]
    lime_samples = st.slider("LIME samples", 200, 2000, 500, 100,
                             help="More samples = more stable explanations, but slower.")
    st.caption("Positive: P(pos) > 0.60 · Negative: P(neg) > 0.60 · otherwise Neutral")
    if st.button("Top global features", use_container_width=True):
        show_global(model_key)


def get_model():
    if model_key == "baseline":
        return st.session_state.get("baseline")
    return load_transformer(model_key)


# ---------------------------------------------------------------------------- upload
st.title("Review Sentiment & Explainability")
up = st.file_uploader("Upload reviews (.xlsx or .csv)", type=["xlsx", "csv"])

if up is not None:
    data = up.getvalue()
    sheets = sheet_names(up.name, data)
    sheet = st.selectbox("Sheet", sheets) if len(sheets) > 1 else None
    df = read_table(up.name, data, sheet)

    c1, c2 = st.columns(2)
    cols = list(df.columns)
    text_col = c1.selectbox("Text column", cols, index=cols.index(detect_text_col(df)))
    rating_col = None
    if model_key == "baseline":
        r_opts = ["(none — use already trained baseline)"] + cols
        auto = detect_rating_col(df)
        r_pick = c2.selectbox("Rating column (1–5) to train on", r_opts,
                              index=r_opts.index(auto) if auto in cols else 0)
        rating_col = None if r_pick == r_opts[0] else r_pick

    if st.button("Classify", type="primary"):
        keep = df[text_col].notna() & df[text_col].astype(str).str.strip().ne("")
        work = df[keep].reset_index(drop=True)
        dropped = int((~keep).sum())
        try:
            if model_key == "baseline" and rating_col:
                with st.spinner("Training baseline on your ratings…"):
                    r = pd.to_numeric(work[rating_col], errors="coerce")
                    ok = r.notna()
                    y = r[ok].map(se.rating_to_label)
                    st.session_state["baseline"] = se.BaselineModel().fit(work.loc[ok, text_col], y)
            model = get_model()
            if model is None:
                st.error("No trained baseline yet. Choose a rating column to train on.")
                st.stop()
            bar = st.progress(0.0, text="Classifying…")
            probs = model.predict_proba(
                work[text_col].tolist(),
                progress_cb=lambda f: bar.progress(f, text=f"Classifying… {int(f * 100)}%"),
            )
            bar.empty()
        except Exception as e:  # model download failure, bad labels, etc.
            st.error(f"Could not classify: {e}")
            st.stop()

        res_df = work.copy()
        res_df["Predicted Sentiment"] = se.decide(probs)
        res_df["P(Negative)"], res_df["P(Neutral)"], res_df["P(Positive)"] = probs.T.round(4)
        res_df["Confidence"] = probs.max(axis=1).round(4)
        st.session_state["results"] = {
            "df": res_df, "text_col": text_col, "model": model_label, "dropped": dropped,
            "xlsx": to_xlsx(res_df), "id": time.time(),
        }

# --------------------------------------------------------------------------- results
res = st.session_state.get("results")
if res:
    rdf, tcol = res["df"], res["text_col"]
    st.subheader("Results")
    if res["dropped"]:
        st.caption(f"{res['dropped']} empty rows skipped.")
    base = st.session_state.get("baseline")
    if res["model"].startswith("Baseline") and base is not None and base.metrics:
        m = base.metrics
        st.info(f"Baseline hold-out ({m['n_test']:,} rows): accuracy {m['accuracy']:.3f}, "
                f"macro-F1 {m['macro_f1']:.3f}. Predictions on the training file itself are optimistic.")

    counts = rdf["Predicted Sentiment"].value_counts()
    mc = st.columns(4)
    mc[0].metric("Reviews", f"{len(rdf):,}")
    for i, lab in enumerate(["Positive", "Neutral", "Negative"]):
        mc[i + 1].metric(f"{EMOJI[lab]} {lab}", f"{int(counts.get(lab, 0)):,}")

    f1, f2, f3, f4 = st.columns([2, 3, 2, 1])
    sel = f1.multiselect("Sentiment", se.LABELS, default=se.LABELS)
    q = f2.text_input("Search text")
    min_conf = f3.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)
    size = f4.selectbox("Rows", [10, 25, 50, 100], index=1)

    mask = rdf["Predicted Sentiment"].isin(sel) & (rdf["Confidence"] >= min_conf)
    if q:
        mask &= rdf[tcol].astype(str).str.contains(q, case=False, regex=False)
    fdf = rdf[mask]
    pages = max(1, -(-len(fdf) // size))
    page = int(st.number_input("Page", 1, pages, 1)) if pages > 1 else 1
    view = fdf.iloc[(page - 1) * size: page * size]

    show = pd.DataFrame({
        "Review Text": view[tcol].astype(str).values,
        "Sentiment": [f"{EMOJI[s]} {s}" for s in view["Predicted Sentiment"]],
        "Negative": view["P(Negative)"].values,
        "Neutral": view["P(Neutral)"].values,
        "Positive": view["P(Positive)"].values,
    })
    prob_cfg = {k: st.column_config.ProgressColumn(k, min_value=0.0, max_value=1.0, format="%.2f")
                for k in ["Negative", "Neutral", "Positive"]}
    st.caption(f"{len(fdf):,} of {len(rdf):,} rows · page {page}/{pages}")
    ev = st.dataframe(show, hide_index=True, use_container_width=True, on_select="rerun",
                      selection_mode="single-row", column_config={
                          "Review Text": st.column_config.TextColumn(width="large"), **prob_cfg})
    rows = ev.selection.rows
    b1, b2 = st.columns([1, 4])
    b1.download_button("Export to Excel", res["xlsx"], "sentiment_results.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    if rows:
        b2.button("Inspect selected row ↓", on_click=send_row, args=(str(view.iloc[rows[0]][tcol]),))
    else:
        b2.caption("Select a row to send it to the inspector below.")

# -------------------------------------------------------------------------- inspector
st.divider()
st.subheader("Review tester & LIME inspector")
text = st.text_area("Review text", key="inspect_text", height=110,
                    placeholder="Type or paste a review, e.g. “Lovely fabric but it didn't fit at all.”")
if st.button("Analyze", disabled=not text.strip()):
    model = get_model()
    if model is None:
        st.warning("Train the baseline first: upload a file with a rating column and click Classify.")
    else:
        try:
            probs = model.predict_proba([text])[0]
            with st.spinner("Running LIME…"):
                exp = se.explain_lime(lambda xs: model.predict_proba(list(xs)), text, lime_samples)
            st.session_state["analysis"] = {
                "probs": probs, "html": se.highlight_html(text, exp), "table": se.lime_table(exp),
            }
        except Exception as e:
            st.error(f"Analysis failed: {e}")

an = st.session_state.get("analysis")
if an:
    label = se.decide(an["probs"][None, :])[0]
    left, right = st.columns([1, 2])
    with left:
        st.metric("Prediction", f"{EMOJI[label]} {label}")
        st.bar_chart(pd.DataFrame({"Class": se.LABELS, "Probability": an["probs"]}),
                     x="Class", y="Probability", horizontal=True)
    with right:
        st.markdown(f'<div class="lime-box">{an["html"]}</div>', unsafe_allow_html=True)
        st.caption("🟩 pulls toward Positive · 🟥 pulls toward Negative · ⬜ negligible. "
                   "Hover a word for exact weights.")
    with st.expander("LIME word weights"):
        st.dataframe(an["table"], hide_index=True, use_container_width=True)
