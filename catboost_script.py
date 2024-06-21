import os
import cv2
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

def extract_features(prev_frame, frame, prev_flow):
    flow = cv2.calcOpticalFlowFarneback(prev_frame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)

    # Basic statistics
    mean_flow = np.mean(flow_mag)
    std_flow = np.std(flow_mag)
    max_flow = np.max(flow_mag)
    min_flow = np.min(flow_mag)

    # Regional analysis
    h, w = flow_mag.shape
    quadrants = [
        flow_mag[:h//2, :w//2], flow_mag[h//2:, :w//2],
        flow_mag[:h//2, w//2:], flow_mag[h//2:, w//2:]
    ]
    regional_means = [np.mean(quadrant) for quadrant in quadrants]
    regional_stds = [np.std(quadrant) for quadrant in quadrants]

    # Temporal changes in flow (frame-to-frame differences)
    if prev_flow is not None:
        flow_diff = np.abs(flow - prev_flow)
        change_mean = np.mean(flow_diff)
        change_std = np.std(flow_diff)
    else:
        change_mean = 0
        change_std = 0

    # Ensure flow is returned to maintain its state across frames
    return [mean_flow, std_flow, max_flow, min_flow, change_mean, change_std] + regional_means + regional_stds, flow

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
        prev_flow = None
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame is not None:
                frame_features, prev_flow = extract_features(prev_frame, frame, prev_flow)
                features.append(frame_features)
            prev_frame = frame

        min_length = min(len(features), len(labels))
        X.append(features[:min_length])
        y_pitch.append(labels[:min_length, 0])
        y_yaw.append(labels[:min_length, 1])

    return np.concatenate(X, axis=0), np.concatenate(y_pitch), np.concatenate(y_yaw)

def save_predictions_and_ground_truth(predictions, ground_truth, filename):
    combined = np.column_stack((predictions, ground_truth))
    np.savetxt(filename, combined, fmt='%.6f', delimiter=',', header='Prediction,Ground Truth', comments='')

def process_video(video_file, model_pitch, model_yaw):
    video = cv2.VideoCapture(video_file)
    features = []
    prev_frame = None
    prev_flow = None
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            frame_features, prev_flow = extract_features(prev_frame, frame, prev_flow)
            features.append(frame_features)
        prev_frame = frame

    features = np.array(features)
    predictions_pitch = model_pitch.predict(features)
    predictions_yaw = model_yaw.predict(features)

    predictions = np.column_stack((predictions_pitch, predictions_yaw))
    return predictions

# Load labeled data
labeled_dir = 'labeled'
X, y_pitch, y_yaw = load_data(labeled_dir)

# Remove rows with NaN in labels
valid_indices_pitch = ~np.isnan(y_pitch)
valid_indices_yaw = ~np.isnan(y_yaw)
X_pitch = X[valid_indices_pitch]
y_pitch = y_pitch[valid_indices_pitch]
X_yaw = X[valid_indices_yaw]
y_yaw = y_yaw[valid_indices_yaw]

# Split data for pitch model
X_train_pitch, X_temp_pitch, y_train_pitch, y_temp_pitch = train_test_split(X_pitch, y_pitch, test_size=0.4, random_state=42)
X_val_pitch, X_test_pitch, y_val_pitch, y_test_pitch = train_test_split(X_temp_pitch, y_temp_pitch, test_size=0.5, random_state=42)

# Train pitch model
model_pitch = CatBoostRegressor(iterations=700, depth=9, learning_rate=0.1, loss_function='RMSE', verbose=True)
model_pitch.fit(X_train_pitch, y_train_pitch)

# Evaluate pitch model
pitch_train_score = model_pitch.score(X_train_pitch, y_train_pitch)
pitch_val_score = model_pitch.score(X_val_pitch, y_val_pitch)
pitch_test_score = model_pitch.score(X_test_pitch, y_test_pitch)
print(f"Pitch Train R^2 score: {pitch_train_score:.4f}")
print(f"Pitch Validation R^2 score: {pitch_val_score:.4f}")
print(f"Pitch Test R^2 score: {pitch_test_score:.4f}")

# Generate and save predictions for pitch
predictions_pitch = model_pitch.predict(X_test_pitch)
save_predictions_and_ground_truth(predictions_pitch, y_test_pitch, 'pitch_predictions_and_truth.txt')

# Split data for yaw model
X_train_yaw, X_temp_yaw, y_train_yaw, y_temp_yaw = train_test_split(X_yaw, y_yaw, test_size=0.4, random_state=42)
X_val_yaw, X_test_yaw, y_val_yaw, y_test_yaw = train_test_split(X_temp_yaw, y_temp_yaw, test_size=0.5, random_state=42)

# Train yaw model
model_yaw = CatBoostRegressor(iterations=700, depth=9, learning_rate=0.1, loss_function='RMSE', verbose=True)
model_yaw.fit(X_train_yaw, y_train_yaw)

# Evaluate yaw model
yaw_train_score = model_yaw.score(X_train_yaw, y_train_yaw)
yaw_val_score = model_yaw.score(X_val_yaw, y_val_yaw)
yaw_test_score = model_yaw.score(X_test_yaw, y_test_yaw)
print(f"Yaw Train R^2 score: {yaw_train_score:.4f}")
print(f"Yaw Validation R^2 score: {yaw_val_score:.4f}")
print(f"Yaw Test R^2 score: {yaw_test_score:.4f}")

# Generate and save predictions for yaw
predictions_yaw = model_yaw.predict(X_test_yaw)
save_predictions_and_ground_truth(predictions_yaw, y_test_yaw, 'yaw_predictions_and_truth.txt')

# Process unlabeled data
unlabeled_dir = 'unlabeled'
for i in range(5, 10):
    video_file = f"{unlabeled_dir}/{i}.hevc"
    predictions = process_video(video_file, model_pitch, model_yaw)
    np.savetxt(f"{unlabeled_dir}/{i}.txt", predictions, fmt='%.6f')

print("Predictions for unlabeled data saved successfully.")
