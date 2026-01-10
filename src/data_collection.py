import requests
import pandas as pd
import numpy as np
from datetime import datetime
import os

# ==============================
# CONFIGURATION
# ==============================
# url = input("Enter phyphox URL : ")
url = "192.168.0.109:8080"
PHY_PHOX_URL = (
    f"http://{url}/get?gyrX=full&gyrY=full&gyrZ=full&accX=full&accY=full&accZ=full"
)

dir_path = "Dataset_Predict/" 
# OUTPUT_CSV = "gesture_dataset_2.csv"

SAMPLING_RATE = 100        # Hz (same as phyphox)
# DIGIT_LABEL = 5            # <-- change per recording

Z_DOWN_THRESHOLD = -7.0
Z_UP_THRESHOLD = 7.0
WINDOW_SIZE = 5            # consecutive samples


# ==============================
# ExTRACT SERIES
# ==============================
def extract_series(buffer):
    """
    phyphox format:
    buffer = [[v1, v2, v3, ...]]
    We want:
    [v1, v2, v3, ...]
    """
    if isinstance(buffer, list) and len(buffer) == 1:
        return buffer[0]
    return buffer


# ==============================
# FETCH DATA
# ==============================
def fetch_phyphox_data():
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


# ==============================
# SAVE CSV
# ==============================
# import os

# def save_csv(df, user_name, digit_label):

#     # folder where this digit's files go
#     output_dir = os.path.join(dir_path, str(digit_label))

#     # make sure the directory exists
#     os.makedirs(output_dir, exist_ok=True)

#     # full csv path
#     csv_path = os.path.join(output_dir, f"{user_name}.csv")

#     # write header only if file doesn't exist
#     write_header = not os.path.exists(csv_path)

#     # append data
#     df.to_csv(csv_path, mode="a", index=False, header=write_header)
 
def save_csv(df, user_name, digit_label):

    # username folder
    user_dir = os.path.join(dir_path, user_name)

    # digit folder inside username
    digit_dir = os.path.join(user_dir, str(digit_label))

    # ensure directories exist
    os.makedirs(digit_dir, exist_ok=True)

    # unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{digit_label}_{timestamp}.csv"
    csv_path = os.path.join(digit_dir, filename)

    # save one gesture per file
    df.to_csv(csv_path, index=False)

    print(f"✅ Saved: {csv_path}")



def normalize(df):
    sensor_cols = [
        "acc_x", "acc_y", "acc_z",
        "gyro_x", "gyro_y", "gyro_z"
    ]
    df[sensor_cols] = (
        df[sensor_cols] - df[sensor_cols].mean()
    ) / df[sensor_cols].std()
    return df


# ==============================
# MAIN
# ==============================
def main():
    # data Details
    # user_name = input("Enter user name: ")
    # digit_label = input("Enter digit label (0-9): ")

    # user_name = input("Enter user digit count: ")

    print("📡 Fetching phyphox data...")
    raw = fetch_phyphox_data()

    print("📊 Parsing sensor buffers...")
    df = parse_sensor_data(raw)

    print("⏱️ Creating timestamps...")
    df = add_time(df)

    print("✂️ Trimming gesture...")
    gesture_df = trim_gesture(df)

    # print("📏 Normalizing sensor values...")
    # gesture_df = normalize(gesture_df)
    
    print("💾 Writing to CSV...")
    # save_csv(gesture_df, user_name, digit_label)

    # save_csv(gesture_df, "9", "Moni")
    # save_csv(gesture_df, "0", "Kumar")
    save_csv(gesture_df, "0", "Nirmal")
    # save_csv(gesture_df, "9", "Gowtham")

    print("✅ Done!")
    # print(gesture_df.describe())
    print("Saved rows:", len(gesture_df))


if __name__ == "__main__":
    main()
