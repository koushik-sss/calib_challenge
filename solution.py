import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load labeled data
labeled_dir = 'labeled'
unlabeled_dir = 'unlabeled'

def load_data(directory):
    X = []
    y = []
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
        y.append(labels[:min_length])

    return np.concatenate(X, axis=0), np.concatenate(y, axis=0)

X, y = load_data(labeled_dir)

# Remove rows with NaN in labels
valid_indices = ~np.isnan(y).any(axis=1)
X = X[valid_indices]
y = y[valid_indices]

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model on validation set
val_score = model.score(X_val, y_val)
print(f"Validation R^2 score: {val_score:.4f}")

# Evaluate model on test set
test_score = model.score(X_test, y_test)
print(f"Test R^2 score: {test_score:.4f}")

# Generate labels for unlabeled data
unlabeled_videos = []
for i in range(5, 10):
    video_file = f"{unlabeled_dir}/{i}.hevc"
    video = cv2.VideoCapture(video_file)
    unlabeled_videos.append(video)

for i, video in enumerate(unlabeled_videos, start=5):
    features = []
    prev_frame = None
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_frame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            features.append(np.mean(flow, axis=(0, 1)))
        prev_frame = frame

    features = np.array(features)
    labels = model.predict(features)
    labels_file_path = f"{unlabeled_dir}/{i}.txt"
    np.savetxt(labels_file_path, labels, fmt='%.6f')

print("Labels generated successfully.")
