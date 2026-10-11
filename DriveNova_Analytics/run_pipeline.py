from src.pipeline import run_pipeline

if __name__ == "__main__":
    cleaned, features = run_pipeline()
    print(f"Cleaned rows: {len(cleaned):,}")
    print(f"Feature windows: {len(features):,}")
    print("Saved data/clean_data.csv and data/features.csv")
