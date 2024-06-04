import cv2
import csv
import math
import numpy as np
from tqdm import tqdm
import os


def read_csv(filename):
    data = []
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data


def get_keypoints(row, prefix):
    keypoints = []
    for i in range(16):
        x = float(row.get(f'{prefix}_x{i}', '0') or '0')
        y = float(row.get(f'{prefix}_y{i}', '0') or '0')
        conf = float(row.get(f'{prefix}_conf{i}', '0') or '0')
        keypoints.append((x, y, conf))
    return keypoints


def cosine_distance(pose1, pose2):
    pose1 = np.array(pose1).flatten()
    pose2 = np.array(pose2).flatten()
    cos_sim = np.dot(pose1, pose2) / (np.linalg.norm(pose1) * np.linalg.norm(pose2))
    return 1 - cos_sim  # Cosine similarity to distance


def weight_distance(pose1, pose2, conf1):
    sum1 = 1 / np.sum(conf1)
    sum2 = 0

    for i in range(len(pose1)):
        conf_ind = i // 2  # Integer division to find the confidence index
        sum2 += conf1[conf_ind] * abs(pose1[i] - pose2[i])

    weighted_dist = sum1 * sum2
    return weighted_dist


def calculate_similarity(row1, row2):
    face1 = get_keypoints(row1, 'face')
    body1 = get_keypoints(row1, 'body')
    face2 = get_keypoints(row2, 'face')
    body2 = get_keypoints(row2, 'body')

    face1_flat = [coord for kp in face1[:5] for coord in kp[:2]]
    face2_flat = [coord for kp in face2[:5] for coord in kp[:2]]
    face_conf = [kp[2] for kp in face1[:5]]

    body1_flat = [coord for kp in body1[:7] for coord in kp[:2]]
    body2_flat = [coord for kp in body2[:7] for coord in kp[:2]]
    body_conf = [kp[2] for kp in body1[:7]]

    face_similarity = cosine_distance(face1_flat, face2_flat)
    body_similarity = weight_distance(body1_flat, body2_flat, body_conf)

    return (face_similarity + body_similarity) / 2  # Average of both similarities


def compare_csvs(file1, file2):
    data1 = read_csv(file1)
    data2 = read_csv(file2)

    similar_frames = []

    total_comparisons = len(data1) * len(data2)
    progress_bar = tqdm(total=total_comparisons, desc='Comparing frames')

    for i, row1 in enumerate(data1):
        for j, row2 in enumerate(data2):
            similarity = calculate_similarity(row1, row2)
            if similarity < threshold:  # Define a similarity threshold
                similar_frames.append((int(row1['frame_number']), int(row2['frame_number']), similarity))
            progress_bar.update(1)

    progress_bar.close()
    return similar_frames


def extract_frame(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()
    cap.release()
    return frame if success else None


def visualize_and_save_comparisons(video1, video2, similar_frames, output_dir='output'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for frame1_num, frame2_num, similarity in similar_frames:
        frame1 = extract_frame(video1, frame1_num)
        frame2 = extract_frame(video2, frame2_num)

        if frame1 is not None and frame2 is not None:
            combined_frame = cv2.hconcat([frame1, frame2])
            filename = os.path.join(output_dir, f'similar_frames_{frame1_num}_{frame2_num}.png')
            cv2.imwrite(filename, combined_frame)
        else:
            print(f"Error extracting frames: {frame1_num} or {frame2_num}")


# Define the similarity threshold
threshold = 0.5  # Adjust this value based on your requirements

# Compare the two CSV files
file1 = 'test1.csv'
file2 = 'test2.csv'
similar_frames = compare_csvs(file1, file2)

# Define the video paths
video1 = "pose_sync_ive_baddie_1.mp4"
video2 = "pose_sync_ive_baddie_2.mp4"

# Visualize and save the comparisons
visualize_and_save_comparisons(video1, video2, similar_frames)
