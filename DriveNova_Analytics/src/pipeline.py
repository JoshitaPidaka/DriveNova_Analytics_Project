from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
CLEAN_PATH = ROOT / "data" / "clean_data.csv"
FEATURES_PATH = ROOT / "data" / "features.csv"

REQUIRED = ["session_id", "timestamp", "ear", "head_yaw",
            "heart_rate", "spo2", "alcohol_raw", "label"]

def load_raw_files():
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No raw CSV files found in {RAW_DIR}. Add sessions or run generate_demo_data.py.")
    frames = []
    for file in files:
        frame = pd.read_csv(file)
        missing = set(REQUIRED) - set(frame.columns)
        if missing:
            raise ValueError(f"{file.name} is missing columns: {', '.join(sorted(missing))}")
        frames.append(frame[REQUIRED])
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in REQUIRED[2:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["session_id", "timestamp", "ear", "head_yaw", "label"])
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])].copy()
    return df

def clean_data(df):
    df = df.sort_values(["session_id", "timestamp"]).copy()
    df.loc[~df["heart_rate"].between(40, 180), "heart_rate"] = np.nan
    df.loc[df["spo2"] < 85, "spo2"] = np.nan
    for _, idx in df.groupby("session_id").groups.items():
        df.loc[idx, ["heart_rate", "spo2"]] = (
            df.loc[idx].sort_values("timestamp")[["heart_rate", "spo2"]]
            .interpolate(limit_direction="both").values
        )
    df["eyes_closed"] = (df["ear"] < 0.20).astype(int)
    df["looking_away"] = (df["head_yaw"].abs() > 25).astype(int)
    return df

def make_features(df, window="30s"):
    rows = []
    for sid, group in df.groupby("session_id"):
        group = group.sort_values("timestamp").set_index("timestamp")
        for start, w in group.resample(window):
            if len(w) < 20:
                continue
            labels = w["label"].dropna()
            if labels.empty:
                continue
            rows.append({
                "session_id": sid,
                "window_start": start,
                "perclos": float(w["eyes_closed"].mean() * 100),
                "ear_mean": float(w["ear"].mean()),
                "head_away_ratio": float(w["looking_away"].mean()),
                "hr_mean": float(w["heart_rate"].mean()),
                "hr_std": float(w["heart_rate"].std(ddof=0)),
                "spo2_min": float(w["spo2"].min()),
                "alcohol_mean": float(w["alcohol_raw"].mean()),
                "label": int(labels.mode().iloc[0]),
            })
    features = pd.DataFrame(rows)
    if features.empty:
        raise ValueError("No 30-second windows with at least 20 readings. Check timestamps and raw data.")
    return features.dropna().reset_index(drop=True)

def run_pipeline():
    raw = load_raw_files()
    cleaned = clean_data(raw)
    features = make_features(cleaned)
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(CLEAN_PATH, index=False)
    features.to_csv(FEATURES_PATH, index=False)
    return cleaned, features
