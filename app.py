import requests
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from scipy.interpolate import interp1d

# ---------------- CONFIG ----------------
MODEL_PATH = "./model/air_gesture_cnn.h5"
mean = np.load("./processed/norm_mean.npy")
std  = np.load("./processed/norm_std.npy")
TIME_STEPS = 150
VALID_DIGITS = list(range(10))
CONFIDENCE_THRESHOLD = 0.6

# url = "192.168.0.109:8080"
# PHY_PHOX_URL = (
#     f"http://{api_url}/get?gyrX=full&gyrY=full&gyrZ=full&accX=full&accY=full&accZ=full"
# )
SAMPLING_RATE = 100    
Z_DOWN_THRESHOLD = -7.0
Z_UP_THRESHOLD = 7.0
WINDOW_SIZE = 5            # consecutive samples


# ==============================
# ExTRACT SERIES
# ==============================
def extract_series(buffer):
    if isinstance(buffer, list) and len(buffer) == 1:
        return buffer[0]
    return buffer


# ==============================
# FETCH DATA
# ==============================
def fetch_phyphox_data(api_url):
    PHY_PHOX_URL = (
            f"http://{api_url}/get?gyrX=full&gyrY=full&gyrZ=full&accX=full&accY=full&accZ=full"
        )
    response = requests.get(PHY_PHOX_URL, timeout=10)
    response.raise_for_status()
    return response.json()["buffer"]


# ==============================
# PARSE DATA
# ==============================
def parse_sensor_data(raw):
    # print(raw["buffer"])
     # Find common length
    acc_x = extract_series(raw["accX"]["buffer"])
    acc_y = extract_series(raw["accY"]["buffer"])
    acc_z = extract_series(raw["accZ"]["buffer"])

    gyro_x = extract_series(raw["gyrX"]["buffer"])
    gyro_y = extract_series(raw["gyrY"]["buffer"])
    gyro_z = extract_series(raw["gyrZ"]["buffer"])
    min_len = min(
        len(acc_x), len(acc_y), len(acc_z),
        len(gyro_x), len(gyro_y), len(gyro_z)
    )
    df = pd.DataFrame({
        "acc_x": acc_x[:min_len],
        "acc_y": acc_y[:min_len],
        "acc_z": acc_z[:min_len],
        "gyro_x": gyro_x[:min_len],
        "gyro_y": gyro_y[:min_len],
        "gyro_z": gyro_z[:min_len],
    })
    return df



# ==============================
# ADD TIME COLUMN
# ==============================
def add_time(df):
    df["time"] = np.arange(len(df)) / SAMPLING_RATE
    return df


# ==============================
# TRIM USING FLIP
# ==============================
def trim_gesture(df):
    down = df["acc_z"] < Z_DOWN_THRESHOLD
    up = df["acc_z"] > Z_UP_THRESHOLD

    down_confirm = down.rolling(WINDOW_SIZE).sum() >= WINDOW_SIZE
    up_confirm = up.rolling(WINDOW_SIZE).sum() >= WINDOW_SIZE

    if not down_confirm.any():
        raise ValueError("❌ Flip DOWN not detected")

    start_idx = down_confirm.idxmax()

    up_after = up_confirm.loc[start_idx + 1:]
    if not up_after.any():
        raise ValueError("❌ Flip UP not detected")

    end_idx = up_after.idxmax()

    return df.loc[start_idx:end_idx].reset_index(drop=True)


def collect_gesture_dataframe():
    """
    Collects sensor data from phyphox and returns a DataFrame
    containing only the gesture segment trimmed using flip detection.
    """
    raw_data = fetch_phyphox_data()
    df = parse_sensor_data(raw_data)
    df = add_time(df)
    gesture_df = trim_gesture(df)
    return gesture_df


# ----------------------------------------

st.set_page_config(page_title="Air Gesture Prediction", layout="centered")
st.title("✋ Air Gesture Prediction")
api_url = st.text_input("Enter Phypox IP")
# ---------- LOAD MODEL ----------
@st.cache_resource
def load_gesture_model():
    return load_model(MODEL_PATH)

model = load_gesture_model()

# ============================================================
# 🔴 INSERT HERE — YOU WILL ADD YOUR OWN LOGIC BELOW 🔴
# ============================================================

def get_truncated_gesture_dataframe():
    """
    YOU MUST IMPLEMENT THIS FUNCTION

    Expected return:
        pd.DataFrame containing ONLY the gesture segment

    Required columns:
        acc_x, acc_y, acc_z,
        gyro_x, gyro_y, gyro_z

    Optional:
        time
    """

    raw_data = fetch_phyphox_data(api_url)
    df = parse_sensor_data(raw_data)
    df = add_time(df)
    gesture_df = trim_gesture(df)
    return gesture_df

    # ------------------------------
    # Example placeholder (REMOVE)
    # ------------------------------
    # return None

# ============================================================
# 🔴 INSERT HERE — YOU WILL ADD YOUR OWN LOGIC ABOVE 🔴
# ============================================================

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



# ---------- PREPROCESS ----------
def preprocess(df):
    
    df = df[SENSOR_COLS]

    X = resample_gesture(df)
    X = normalize(X)
    X = np.expand_dims(X, axis=0)  # (1, 150, 6)
    
    return X

# ---------- UI ----------
st.subheader("Gesture Prediction")

if st.button("🔮 Predict Gesture"):
    try:
        with st.spinner("Waiting for gesture data..."):
            gesture_df = get_truncated_gesture_dataframe()

        # ---------- VALIDATION ----------
        if gesture_df is None or not isinstance(gesture_df, pd.DataFrame):
            st.error("Draw a Valid Gesture")
            st.stop()

        if len(gesture_df) < 20:
            st.error("Draw a Valid Gesture")
            st.stop()

        st.success("Gesture data received")

        # ---------- PREPROCESS ----------
        processed = preprocess(gesture_df)

        # ---------- MODEL ----------
        prediction = model.predict(processed)
        predicted_digit = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        # ---------- OUTPUT ----------
        if predicted_digit not in VALID_DIGITS or confidence < CONFIDENCE_THRESHOLD:
            st.error("❌ Draw a Valid Gesture")
        else:
            st.success(f"✅ Predicted Digit: {predicted_digit}")
            # st.metric("Confidence", f"{confidence * 100:.2f}%")

    except Exception as e:
        st.error("Prediction failed")
        st.exception(e)
