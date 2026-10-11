from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "features.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FEATURES = ["perclos", "ear_mean", "head_away_ratio", "hr_mean",
            "hr_std", "spo2_min", "alcohol_mean"]

def train():
    if not DATA.exists():
        raise FileNotFoundError("Run `python run_pipeline.py` first.")
    data = pd.read_csv(DATA)
    if data["label"].nunique() < 2:
        raise ValueError("Training needs both labels (0=Alert and 1=Drowsy). Collect both classes.")
    if data["session_id"].nunique() < 4:
        raise ValueError("Need at least 4 distinct sessions for a minimally useful grouped holdout. More sessions are strongly recommended.")
    X, y, groups = data[FEATURES], data["label"].astype(int), data["session_id"]
    splitter = GroupShuffleSplit(n_splits=20, test_size=0.25, random_state=42)
    train_idx, test_idx = next(
        (a, b) for a, b in splitter.split(X, y, groups)
        if y.iloc[a].nunique() == 2 and y.iloc[b].nunique() == 2
    )
    model = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                   random_state=42, min_samples_leaf=2)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = model.predict(X.iloc[test_idx])
    metrics = {
        "accuracy": float(accuracy_score(y.iloc[test_idx], pred)),
        "classification_report": classification_report(
            y.iloc[test_idx], pred, labels=[0, 1],
            target_names=["Alert", "Drowsy"], output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y.iloc[test_idx], pred, labels=[0, 1]).tolist(),
        "train_sessions": sorted(groups.iloc[train_idx].unique().tolist()),
        "test_sessions": sorted(groups.iloc[test_idx].unique().tolist()),
        "feature_names": FEATURES,
        "note": "Interpret cautiously, especially for small or synthetic datasets."
    }
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "random_forest.joblib")
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False).to_csv(
        REPORT_DIR / "feature_importance.csv", header=["importance"])
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print("Confusion matrix (rows=true [Alert,Drowsy], columns=predicted [Alert,Drowsy]):")
    print(metrics["confusion_matrix"])
    print("Drowsy recall:", round(metrics["classification_report"]["Drowsy"]["recall"], 3))
    print(f"Saved model to {MODEL_DIR / 'random_forest.joblib'}")
    print(f"Saved metrics to {REPORT_DIR / 'metrics.json'}")
    return model, metrics

if __name__ == "__main__":
    train()
