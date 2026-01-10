import numpy as np
from sklearn.model_selection import train_test_split

X = np.load("./processed/X.npy")   # (N, 150, 6)
y = np.load("./processed/y.npy")

X_train, X_test, y_train, Y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# compute per-channel mean & std over ALL timesteps
mean = X_train.mean(axis=(0, 1))   # shape (6,)
std  = X_train.std(axis=(0, 1))    # shape (6,)

# numerical safety
std[std == 0] = 1e-6
def normalize(X, mean, std):
    return (X - mean) / std

X_train = normalize(X_train, mean, std)
X_test   = normalize(X_test, mean, std)

np.save("./processed/X_train.npy", X_train)
np.save("./processed/X_test.npy", X_test)
np.save("./processed/y_train.npy", y_train)
np.save("./processed/Y_test.npy", Y_test)

np.save("./processed/norm_mean.npy", mean)
np.save("./processed/norm_std.npy", std)
