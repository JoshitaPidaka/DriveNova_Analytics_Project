"""Serial logging template for DriveNova.
Arduino output per line: HR:74,SPO2:98,ALC:310
IMPORTANT: replace the eye/head placeholder with values from your existing vision pipeline.
Run only at a desk, never while driving. Obtain consent and use anonymous IDs.
"""
import csv
import time
from datetime import datetime
from pathlib import Path
import serial

SERIAL_PORT = "COM3"  # Change in Device Manager
BAUD_RATE = 9600
SESSION = "S01"
LABEL = 0  # 0=Alert, 1=Drowsy. Set before each session.
DURATION_SECONDS = 300
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "raw" / f"{SESSION}.csv"

def read_arduino(ser):
    line = ser.readline().decode(errors="ignore").strip()
    try:
        values = dict(part.split(":") for part in line.split(","))
        return int(values["HR"]), int(values["SPO2"]), int(values["ALC"])
    except (ValueError, KeyError):
        return None, None, None

def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1) as ser, OUTPUT.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["session_id", "timestamp", "ear", "head_yaw",
                         "heart_rate", "spo2", "alcohol_raw", "label"])
        start = time.time()
        while time.time() - start < DURATION_SECONDS:
            hr, spo2, alc = read_arduino(ser)
            if hr is None:
                continue
            # TODO: replace with live values from DriveNova's MediaPipe pipeline.
            ear, head_yaw = 0.27, 0.0
            writer.writerow([SESSION, datetime.now().isoformat(), ear, head_yaw, hr, spo2, alc, LABEL])
            f.flush()
            print(f"{SESSION}: HR={hr}, SpO2={spo2}, MQ3={alc}")
    print(f"Saved {OUTPUT}")

if __name__ == "__main__":
    main()
