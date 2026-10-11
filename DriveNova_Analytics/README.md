# DriveNova Analytics
### Predictive Driver Risk Analytics from Multi-Sensor Data

This starter project follows the supplied **DriveNova Analytics: Project Build Guide**. It converts sensor readings into 30-second windows, explores patterns, trains a Random Forest classifier, computes a configurable risk score, and presents results in Streamlit.

## Important limitations
- The included demo dataset is **synthetic** and exists only to test the workflow/UI. It is not evidence of real-world model performance.
- Replace demo data with consented measurements from your prototype before reporting results.
- This is an educational prototype, **not a certified driver-safety system**. Never test while driving.
- MQ-3 values are raw sensor readings, not legal blood-alcohol measurements. Never use real alcohol on a person; follow the guide's safe sensor test approach.
- Heart rate and SpO₂ are noisy consumer-sensor measurements and are not medical assessments.

## Quick start (Windows)
1. Install Python 3.11.
2. Open this folder in VS Code.
3. In the terminal:
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   python generate_demo_data.py
   python run_pipeline.py
   streamlit run app.py
   ```
4. Open the local URL shown by Streamlit.

## Workflow
1. **Collect:** one row per second in `data/raw/<session_id>.csv`.
2. **Clean + features:** `python run_pipeline.py` writes `data/clean_data.csv` and `data/features.csv`.
3. **Explore:** run `python analyze_data.py` to write charts to `reports/`.
4. **Train:** run `python train_model.py`; metrics and model are saved to `reports/` and `models/`.
5. **Dashboard:** `streamlit run app.py`.

## Raw CSV schema
Required columns:
`session_id,timestamp,ear,head_yaw,heart_rate,spo2,alcohol_raw,label`

- `ear`: eye aspect ratio from the face/eye landmark pipeline
- `head_yaw`: head rotation in degrees
- `heart_rate`, `spo2`: MAX30102-derived readings
- `alcohol_raw`: MQ-3 raw ADC/sensor reading
- `label`: 0 = Alert, 1 = Drowsy, set for the whole session

Use anonymous session IDs, get volunteer consent, and store numeric readings only. The guide proposes 15 sessions of five minutes each across two volunteers. Record only at a desk, never during real driving.

## Hardware integration
`src/serial_logger.py` expects Arduino lines such as `HR:74,SPO2:98,ALC:310`. Update `SERIAL_PORT` and replace the `ear, head_yaw` placeholder with your existing DriveNova eye/head outputs. Start with one test session and verify the CSV before collecting a full dataset.

## Model evaluation
The split is grouped by `session_id`, not random rows, to reduce leakage from similar windows in the same session. With only a small number of sessions, scores will be unstable. Report accuracy, precision, recall, confusion matrix, and especially drowsy-class recall. Do not claim general driving safety from desk-based data.

## Risk score
`100 * (0.40*D + 0.25*H + 0.20*A + 0.15*V)`, where:
- `D`: model probability of Drowsy
- `H`: head-away ratio, clipped to [0, 1]
- `A`: normalized increase in MQ-3 reading over a baseline, clipped to [0, 1]
- `V`: 1 if heart rate is outside 50–120 or SpO₂ below 94, otherwise 0

Levels: 0–30 Low, 31–60 Medium, 61–100 High. These weights and thresholds are design choices for this project, not clinically or legally validated thresholds.

## Project structure
- `app.py` – Streamlit dashboard
- `run_pipeline.py` – cleaning and 30-second feature extraction
- `train_model.py` – session-grouped Random Forest training/evaluation
- `analyze_data.py` – five charts from the guide
- `generate_demo_data.py` – clearly labelled synthetic test data
- `src/serial_logger.py` – serial logger template
