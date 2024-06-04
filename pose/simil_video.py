import cv2
import csv
import math
from tqdm import tqdm

def read_csv(filename):
    data = []
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data

def calculate_similarity(row1, row2):
    def get_keypoints(row, prefix):
        keypoints = []
        for i in range(16):
            x = row.get(f'{prefix}_x{i}', '0') or '0'
            y = row.get(f'{prefix}_y{i}', '0') or '0'
            conf = row.get(f'{prefix}_conf{i}', '0') or '0'
            keypoints.append((float(x), float(y), float(conf)))
        return keypoints

    def calculate_distance(kp1, kp2):
        if kp1[2] == 0 or kp2[2] == 0:  # Skip keypoints with zero confidence
            return 0
        return math.sqrt((kp1[0] - kp2[0]) ** 2 + (kp1[1] - kp2[1]) ** 2)

    # Get keypoints for face and body
    face1 = get_keypoints(row1, 'face')
    body1 = get_keypoints(row1, 'body')
    face2 = get_keypoints(row2, 'face')
    body2 = get_keypoints(row2, 'body')

    # Compare keypoints (excluding leg keypoints)
    distance = 0
    count = 0
    for kp1, kp2 in zip(face1[:5] + body1[:7], face2[:5] + body2[:7]):
        dist = calculate_distance(kp1, kp2)
        if dist != 0:
            distance += dist
            count += 1

    # Return average distance as similarity measure
    return distance / count if count > 0 else float('inf')

def compare_csvs(file1, file2, threshold):
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

def visualize_comparisons(video1, video2, similar_frames):
    for frame1_num, frame2_num, similarity in similar_frames:
        frame1 = extract_frame(video1, frame1_num)
        frame2 = extract_frame(video2, frame2_num)

        if frame1 is not None and frame2 is not None:
            combined_frame = cv2.hconcat([frame1, frame2])
            cv2.imshow(f'Similar frames (Similarity: {similarity:.2f})', combined_frame)
            cv2.waitKey(0)  # Press any key to move to the next comparison
        else:
            print(f"Error extracting frames: {frame1_num} or {frame2_num}")

    cv2.destroyAllWindows()

def switch_videos_and_save(video1_path, video2_path, switch_frame1, switch_frame2, output_path):
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    frame_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap1.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_number = 0
    while cap1.isOpened() and frame_number < switch_frame1:
        success, frame = cap1.read()
        if not success:
            break
        out.write(frame)
        frame_number += 1

    frame_number = 0
    cap2.set(cv2.CAP_PROP_POS_FRAMES, switch_frame2)
    while cap2.isOpened():
        success, frame = cap2.read()
        if not success:
            break
        out.write(frame)
        frame_number += 1

    cap1.release()
    cap2.release()
    out.release()

def find_switch_point(similar_frames, min_frames=5):
    if not similar_frames:
        return None, None
    for i in range(len(similar_frames) - min_frames + 1):
        if all(similar_frames[j][0] == similar_frames[i][0] + j for j in range(min_frames)):
            return similar_frames[i][0], similar_frames[i][1]
    return None, None

# Define the similarity threshold
threshold = 30  # Adjust this value based on your requirements

# Compare the two CSV files
file1 = 'test1.csv'
file2 = 'test2.csv'
similar_frames = compare_csvs(file1, file2, threshold)

# Define the video paths
video1 = "pose_sync_ive_baddie_1.mp4"
video2 = "pose_sync_ive_baddie_2.mp4"

# Visualize the comparisons
visualize_comparisons(video1, video2, similar_frames)

# Find the switch point
switch_frame1, switch_frame2 = find_switch_point(similar_frames, min_frames=5)
if switch_frame1 is not None and switch_frame2 is not None:
    print(f'Switching at frame {switch_frame1} in video1 and frame {switch_frame2} in video2')
    output_path = 'output_video.mp4'
    switch_videos_and_save(video1, video2, switch_frame1, switch_frame2, output_path)
else:
    print('No suitable switch point found.')
