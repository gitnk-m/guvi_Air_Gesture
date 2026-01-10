import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# =========================
# CONFIG
# =========================
DATASET_DIR = "Dataset"   # root folder: 0/,1/,2/...
TIMESTEPS = 150
SENSOR_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

# =========================
# RESAMPLING FUNCTION
# =========================
def resample_csv(df, timesteps):
    """
    Resample sensor data to fixed timesteps using interpolation
    """
    old_len = len(df)
    old_x = np.linspace(0, 1, old_len)
    new_x = np.linspace(0, 1, timesteps)

    resampled = []
    for col in SENSOR_COLS:
        f = interp1d(old_x, df[col].values, kind="linear")
        resampled.append(f(new_x))

    return np.stack(resampled, axis=1)  # (150, 6)

# =========================
# MAIN PIPELINE
# =========================
X, y = [], []

for digit in sorted(os.listdir(DATASET_DIR)):
    digit_path = os.path.join(DATASET_DIR, digit)
    if not os.path.isdir(digit_path):
        continue

    for file in os.listdir(digit_path):
        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(digit_path, file)

        try:
            df = pd.read_csv(file_path)

            # column safety check
            if not all(col in df.columns for col in SENSOR_COLS):
                continue

            df = df[SENSOR_COLS]
            resampled = resample_csv(df, TIMESTEPS)

            X.append(resampled)
            y.append(int(digit))

        except Exception as e:
            print(f"Skipped {file_path}: {e}")

# =========================
# SAVE OUTPUTS
# =========================
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

np.save("X.npy", X)
np.save("y.npy", y)

print("Preprocessing complete")
print("X shape:", X.shape)
print("y shape:", y.shape)
