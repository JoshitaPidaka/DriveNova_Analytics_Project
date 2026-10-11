from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "raw"
RNG = np.random.default_rng(24)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Synthetic data is for testing the code path only, not scientific evaluation.
    for s in range(1, 9):
        label = 0 if s <= 4 else 1
        n = 300
        ts = pd.date_range("2026-10-01 18:00:00", periods=n, freq="s") + pd.Timedelta(minutes=s*7)
        drowsy = label == 1
        ear = np.clip(RNG.normal(0.28 if not drowsy else 0.21, 0.035, n), 0.08, 0.42)
        if drowsy:
            blink_idx = RNG.choice(n, size=35, replace=False)
            ear[blink_idx] = RNG.uniform(0.08, 0.17, len(blink_idx))
        yaw = RNG.normal(0, 10 if not drowsy else 18, n)
        hr = RNG.normal(76 if not drowsy else 70, 5, n)
        spo2 = RNG.normal(98, 0.8, n)
        alc = RNG.normal(300, 8, n)
        df = pd.DataFrame({
            "session_id": f"DEMO_S{s:02d}", "timestamp": ts.astype(str),
            "ear": ear.round(4), "head_yaw": yaw.round(2),
            "heart_rate": hr.round().astype(int), "spo2": spo2.round().astype(int),
            "alcohol_raw": alc.round().astype(int), "label": label
        })
        df.to_csv(OUT / f"DEMO_S{s:02d}.csv", index=False)
    (ROOT / "data" / "DEMO_DATA_NOTICE.txt").write_text(
        "All DEMO_S*.csv files are synthetic. Do not report their model metrics as real-world validation. Replace with real, consented prototype data."
    )
    print("Created 8 synthetic demo sessions in data/raw. Replace these with real sensor recordings before final evaluation.")

if __name__ == "__main__":
    main()
