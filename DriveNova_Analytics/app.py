from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
FEATURES_PATH = ROOT / "data" / "features.csv"
MODEL_PATH = ROOT / "models" / "random_forest.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
IMPORTANCE_PATH = ROOT / "reports" / "feature_importance.csv"
FEATURES = ["perclos", "ear_mean", "head_away_ratio", "hr_mean",
            "hr_std", "spo2_min", "alcohol_mean"]

st.set_page_config(page_title="DriveNova Analytics", page_icon="🚘", layout="wide")
st.title("🚘 DriveNova Analytics")
st.caption("Multi-sensor driver-risk analytics prototype | Educational use only")

if not FEATURES_PATH.exists():
    st.warning("No feature dataset found yet. Run `python generate_demo_data.py`, then `python run_pipeline.py`.")
    st.stop()

data = pd.read_csv(FEATURES_PATH)
if data.empty:
    st.error("features.csv is empty. Check your raw sessions.")
    st.stop()

if any(str(x).startswith("DEMO_") for x in data["session_id"].astype(str).unique()):
    st.info("Demo data detected: these results are synthetic and must not be presented as real validation.")

if not MODEL_PATH.exists():
    st.warning("Model not trained yet. Run `python train_model.py` from the project folder.")
    st.stop()

model = joblib.load(MODEL_PATH)
metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
data["predicted_probability"] = model.predict_proba(data[FEATURES])[:, list(model.classes_).index(1)]
data["predicted_label"] = (data["predicted_probability"] >= 0.5).astype(int)

# A simple transparent MQ-3 normalization baseline. It is a raw sensor index, not an alcohol concentration.
baseline = float(data["alcohol_mean"].median())
spread = max(float(data["alcohol_mean"].quantile(0.95) - baseline), 1.0)
data["alcohol_factor"] = ((data["alcohol_mean"] - baseline) / spread).clip(0, 1)
data["health_factor"] = ((data["hr_mean"] < 50) | (data["hr_mean"] > 120) | (data["spo2_min"] < 94)).astype(int)
data["risk_score"] = 100 * (
    0.40 * data["predicted_probability"] +
    0.25 * data["head_away_ratio"].clip(0, 1) +
    0.20 * data["alcohol_factor"] +
    0.15 * data["health_factor"]
)
data["risk_level"] = pd.cut(data["risk_score"], bins=[-0.01, 30, 60, 100.01],
                            labels=["Low", "Medium", "High"])

tab1, tab2, tab3 = st.tabs(["Overview", "Session view", "Model results"])
with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Sessions", int(data["session_id"].nunique()))
    c2.metric("Average risk score", f"{data['risk_score'].mean():.1f}/100")
    c3.metric("Windows predicted Drowsy", f"{100*data['predicted_label'].mean():.1f}%")
    st.subheader("Risk score by window")
    st.line_chart(data.sort_values("window_start" if "window_start" in data else "session_id")["risk_score"])
    st.subheader("Windows by risk level")
    st.bar_chart(data["risk_level"].value_counts().reindex(["Low", "Medium", "High"], fill_value=0))
    st.caption("Risk weights are project design choices, not clinically validated thresholds. Scores are not a substitute for driver monitoring or safe driving decisions.")

with tab2:
    sessions = sorted(data["session_id"].astype(str).unique())
    chosen = st.selectbox("Choose session", sessions)
    subset = data[data["session_id"].astype(str) == chosen].copy()
    xcol = "window_start" if "window_start" in subset.columns else None
    if xcol:
        subset[xcol] = pd.to_datetime(subset[xcol], errors="coerce")
        subset = subset.sort_values(xcol)
        st.line_chart(subset.set_index(xcol)[["risk_score", "perclos"]])
    else:
        st.line_chart(subset[["risk_score", "perclos"]])
    st.dataframe(subset[["session_id", "label", "perclos", "head_away_ratio",
                         "hr_mean", "spo2_min", "predicted_probability",
                         "risk_score", "risk_level"]], use_container_width=True)

with tab3:
    st.subheader("Held-out model evaluation")
    if metrics:
        st.metric("Grouped holdout accuracy", f"{metrics.get('accuracy', 0):.3f}")
        st.caption("The test split is grouped by session. Scores can still be unstable with few sessions.")
        cm = metrics.get("confusion_matrix")
        if cm:
            st.write("Confusion matrix (rows=true Alert, Drowsy; columns=predicted Alert, Drowsy)")
            st.dataframe(pd.DataFrame(cm, index=["True Alert", "True Drowsy"],
                                      columns=["Predicted Alert", "Predicted Drowsy"]))
        report = metrics.get("classification_report", {})
        if "Drowsy" in report:
            st.metric("Drowsy recall", f"{report['Drowsy'].get('recall', 0):.3f}")
    else:
        st.info("Metrics not found. Run `python train_model.py`.")
    if IMPORTANCE_PATH.exists():
        importance = pd.read_csv(IMPORTANCE_PATH, index_col=0)
        st.subheader("Feature importance")
        st.bar_chart(importance)
    st.caption("A small desk-recorded dataset cannot establish real-world driving safety. Include limitations in the report.")
