import csv
import numpy as np
from tqdm import tqdm
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean, cosine
import cv2
import pandas as pd
import os


def read_csv(filename):
    data = pd.read_csv(filename)
    return data


def get_keypoints(row, start_index, count):
    keypoints = []
    for i in range(count):
        x_index = start_index + i * 3 + 1
        y_index = start_index + i * 3 + 2
        if x_index < len(row) and y_index < len(row):
            x = row.iloc[x_index]
            y = row.iloc[y_index]
            # Replace NaN or infinite values with 0
            if np.isnan(x) or np.isinf(x):
                x = 0.0
            if np.isnan(y) or np.isinf(y):
                y = 0.0
            keypoints.append((x, y))
        else:
            keypoints.append((0.0, 0.0))
    return keypoints


def normalize_keypoints(keypoints, width, height):
    return [(x / width, y / height) for x, y in keypoints]


def calculate_dtw_distance(keypoints1, keypoints2):
    distance, _ = fastdtw(keypoints1, keypoints2, dist=euclidean)
    return distance


def calculate_cosine_similarity(keypoints1, keypoints2):
    flat1 = np.array(keypoints1).flatten()
    flat2 = np.array(keypoints2).flatten()
    return cosine(flat1, flat2)


def calculate_combined_similarity(row1, row2, width, height):
    face1 = get_keypoints(row1, 0, 5)
    body1 = get_keypoints(row1, 15, 7)
    leg1 = get_keypoints(row1, 36, 4)

    face2 = get_keypoints(row2, 0, 5)
    body2 = get_keypoints(row2, 15, 7)
    leg2 = get_keypoints(row2, 36, 4)

    face1 = normalize_keypoints(face1, width, height)
    body1 = normalize_keypoints(body1, width, height)
    leg1 = normalize_keypoints(leg1, width, height)

    face2 = normalize_keypoints(face2, width, height)
    body2 = normalize_keypoints(body2, width, height)
    leg2 = normalize_keypoints(leg2, width, height)

    face_similarity = calculate_dtw_distance(face1, face2) + calculate_cosine_similarity(face1, face2)
    body_similarity = calculate_dtw_distance(body1, body2) + calculate_cosine_similarity(body1, body2)
    leg_similarity = calculate_dtw_distance(leg1, leg2) + calculate_cosine_similarity(leg1, leg2)

    return (face_similarity + body_similarity + leg_similarity) / 3


def compare_csvs(file1, file2, threshold=10, width=1920, height=1080):
    data1 = read_csv(file1)
    data2 = read_csv(file2)

    similar_frames = []

    total_comparisons = len(data1) * len(data2)
    progress_bar = tqdm(total=total_comparisons, desc='Comparing frames')

    for i, row1 in data1.iterrows():
        for j, row2 in data2.iterrows():
            frame1_num = int(row1['frame_number'])
            frame2_num = int(row2['frame_number'])
            frame_difference = abs(frame1_num - frame2_num)

            if frame_difference > 1 or frame1_num <= frame2_num:
                continue

            similarity = calculate_combined_similarity(row1, row2, width, height)
            if similarity < threshold:
                similar_frames.append((frame1_num, frame2_num, similarity))
            progress_bar.update(1)

    progress_bar.close()
    similar_frames.sort()
    return similar_frames


def extract_frame(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()
    cap.release()
    return frame if success else None


def create_transition_video(video1_path, video2_path, similar_frames, output_path):
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap1.get(cv2.CAP_PROP_FPS)
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    use_video1 = True
    last_transition_frame = -3  # 초기값 설정, 첫 프레임이 0이기 때문에 -3으로 설정하여 첫 전환이 가능하도록 함

    print(f"Total similar frames: {len(similar_frames)}")

    similar_frame_dict = {frame1_num: frame2_num for frame1_num, frame2_num, _ in similar_frames}
    print("similar_frame_dict:", similar_frame_dict)

    while True:
        if use_video1:
            ret, frame = cap1.read()
        else:
            ret, frame = cap2.read()

        if not ret:
            break

        if frame_idx in similar_frame_dict:
            next_frame_idx = similar_frame_dict[frame_idx]
            if frame_idx - last_transition_frame >= 5:  # 3프레임 연속 전환 방지
                if use_video1:
                    cap2.set(cv2.CAP_PROP_POS_FRAMES, next_frame_idx)
                    ret, frame = cap2.read()
                else:
                    cap1.set(cv2.CAP_PROP_POS_FRAMES, next_frame_idx)
                    ret, frame = cap1.read()
                use_video1 = not use_video1
                last_transition_frame = frame_idx
                print(f"Transitioning at frame {frame_idx} to frame {next_frame_idx} in the other video")
                frame_idx = next_frame_idx

        out.write(frame)
        frame_idx += 1

    cap1.release()
    cap2.release()
    out.release()

# Define the similarity threshold
threshold = 7  # Adjust this value based on your requirements

# Compare the two CSV files
file1 = 'test1.csv'
file2 = 'test2.csv'
similar_frames = compare_csvs(file1, file2, threshold)

# Define the video paths
video1 = "pose_sync_ive_baddie_1.mp4"
video2 = "pose_sync_ive_baddie_2.mp4"

# Output video path
output_video_path = "output/transition_video.mp4"

# Create and save the transition video
create_transition_video(video1, video2, similar_frames, output_video_path)
