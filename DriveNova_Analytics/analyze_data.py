from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "features.csv"
OUT = ROOT / "reports"

def main():
    if not DATA.exists():
        raise FileNotFoundError("Run `python run_pipeline.py` first.")
    d = pd.read_csv(DATA)
    OUT.mkdir(exist_ok=True)
    labels = d["label"].map({0: "Alert", 1: "Drowsy"})
    plots = [
        ("perclos", "PERCLOS by label", "reports/perclos_by_label.png", "box"),
        ("hr_mean", "Mean heart rate by label", "reports/heart_rate_by_label.png", "box"),
    ]
    for col, title, filename, kind in plots:
        plt.figure(figsize=(7, 4))
        sns.boxplot(x=labels, y=d[col])
        plt.title(title); plt.tight_layout(); plt.savefig(ROOT / filename, dpi=150); plt.close()
    plt.figure(figsize=(7, 4))
    d.assign(label_name=labels).groupby("label_name")["head_away_ratio"].mean().reindex(["Alert", "Drowsy"]).plot(kind="bar")
    plt.ylabel("Mean head-away ratio"); plt.title("Average head-away ratio by label")
    plt.tight_layout(); plt.savefig(OUT / "head_away_ratio.png", dpi=150); plt.close()
    plt.figure(figsize=(8, 6))
    sns.heatmap(d.select_dtypes("number").drop(columns=["label"], errors="ignore").corr(), annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Feature correlation heatmap"); plt.tight_layout()
    plt.savefig(OUT / "correlation_heatmap.png", dpi=150); plt.close()
    # If raw files contain a sanitiser test, chart its MQ-3 readings. This is only a raw sensor demonstration.
    raw_files = list((ROOT / "data" / "raw").glob("*.csv"))
    raw = pd.concat([pd.read_csv(f) for f in raw_files], ignore_index=True) if raw_files else pd.DataFrame()
    if not raw.empty and "alcohol_raw" in raw:
        plt.figure(figsize=(8, 4))
        plt.plot(range(len(raw)), raw["alcohol_raw"])
        plt.xlabel("Sample index"); plt.ylabel("MQ-3 raw reading")
        plt.title("MQ-3 raw readings (mark sanitiser-test session separately)")
        plt.tight_layout(); plt.savefig(OUT / "mq3_readings.png", dpi=150); plt.close()
    print(f"Charts saved to {OUT}")

if __name__ == "__main__":
    main()
