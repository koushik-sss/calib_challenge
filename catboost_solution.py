import os
import cv2
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

# Load labeled data
labeled_dir = 'labeled'
unlabeled_dir = 'unlabeled'

def load_data(directory):
    X = []
    y_pitch = []
    y_yaw = []
    for i in range(5):
        video_file = f"{directory}/{i}.hevc"
        label_file = f"{directory}/{i}.txt"
        video = cv2.VideoCapture(video_file)
        labels = np.loadtxt(label_file)

        features = []
        prev_frame = None
        frame_count = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_frame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                features.append(np.mean(flow, axis=(0, 1)))
            prev_frame = frame
            frame_count += 1

        min_length = min(len(features), len(labels))
        X.append(features[:min_length])
        y_pitch.append(labels[:min_length, 0])
        y_yaw.append(labels[:min_length, 1])

    return np.concatenate(X, axis=0), np.concatenate(y_pitch), np.concatenate(y_yaw)

X, y_pitch, y_yaw = load_data(labeled_dir)

# Remove rows with NaN in labels
valid_indices_pitch = ~np.isnan(y_pitch)
valid_indices_yaw = ~np.isnan(y_yaw)
X_pitch = X[valid_indices_pitch]
y_pitch = y_pitch[valid_indices_pitch]
X_yaw = X[valid_indices_yaw]
y_yaw = y_yaw[valid_indices_yaw]

# Training for pitch
X_train_pitch, X_test_pitch, y_train_pitch, y_test_pitch = train_test_split(X_pitch, y_pitch, test_size=0.2, random_state=42)
model_pitch = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.1, loss_function='RMSE', verbose=True)
model_pitch.fit(X_train_pitch, y_train_pitch)
pitch_score = model_pitch.score(X_test_pitch, y_test_pitch)
print(f"Pitch R^2 score: {pitch_score:.4f}")

# Training for yaw
X_train_yaw, X_test_yaw, y_train_yaw, y_test_yaw = train_test_split(X_yaw, y_yaw, test_size=0.2, random_state=42)
model_yaw = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.1, loss_function='RMSE', verbose=True)
model_yaw.fit(X_train_yaw, y_train_yaw)
yaw_score = model_yaw.score(X_test_yaw, y_test_yaw)
print(f"Yaw R^2 score: {yaw_score:.4f}")
