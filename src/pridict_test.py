import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from scipy.interpolate import interp1d

model = load_model("./model/air_gesture_cnn.h5")
mean = np.load("./processed/norm_mean.npy")
std  = np.load("./processed/norm_std.npy")

SENSOR_COLS = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
TIMESTEPS = 150

def resample_gesture(df, timesteps=150):
    old_len = len(df)
    old_x = np.linspace(0, 1, old_len)
    new_x = np.linspace(0, 1, timesteps)

    data = []
    for col in SENSOR_COLS:
        f = interp1d(old_x, df[col].values, kind="linear")
        data.append(f(new_x))

    return np.stack(data, axis=1)  # (150, 6)

def normalize(X):
    return (X - mean) / std

def predict_digit(raw_df):
    """
    raw_df: DataFrame with acc & gyro columns for ONE gesture
    """
    df = raw_df[SENSOR_COLS]

    X = resample_gesture(df)
    X = normalize(X)
    X = np.expand_dims(X, axis=0)  # (1, 150, 6)

    probs = model.predict(X)
    digit = np.argmax(probs)

    return digit, probs

raw_df = pd.read_csv(r"D:\Learn\Guvi\DS\project\Air Gesture\Dataset_Predict\0\Nirmal\Nirmal_20260110_112456.csv")  # simulated real-time capture
digit, confidence = predict_digit(raw_df)

print("Predicted Digit:", digit)