import numpy as np
from scipy.spatial.distance import euclidean, cosine
from fastdtw import fastdtw
import pandas as pd
from tqdm import tqdm

def calculate_similarities(csv_files, width, height, threshold, position_threshold, size_threshold, avg_similarity_threshold):
    def get_keypoints(row, start_index, count):
        keypoints = []
        for i in range(count):
            x_index = start_index + i * 3 + 1
            y_index = start_index + i * 3 + 2
            if x_index < len(row) and y_index < len(row):
                x = row[x_index]
                y = row[y_index]
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
        uu = np.dot(flat1, flat1)
        vv = np.dot(flat2, flat2)
        if uu == 0 or vv == 0:
            return 1.0
        return cosine(flat1, flat2)

    def calculate_centroid(keypoints):
        x_coords = [x for x, y in keypoints]
        y_coords = [y for x, y in keypoints]
        centroid = (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))
        return centroid

    def calculate_size(keypoints):
        x_coords = [x for x, y in keypoints]
        y_coords = [y for x, y in keypoints]
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        return width, height

    def filter_by_position_and_size(keypoints1, keypoints2, width, height, position_threshold=0.1, size_threshold=0.1):
        centroid1 = calculate_centroid(keypoints1)
        centroid2 = calculate_centroid(keypoints2)

        size1 = calculate_size(keypoints1)
        size2 = calculate_size(keypoints2)

        # Normalize by video dimensions
        norm_centroid1 = (centroid1[0] / width, centroid1[1] / height)
        norm_centroid2 = (centroid2[0] / width, centroid2[1] / height)

        norm_size1 = (size1[0] / width, size1[1] / height)
        norm_size2 = (size2[0] / width, size2[1] / height)

        position_diff = euclidean(norm_centroid1, norm_centroid2)
        size_diff = euclidean(norm_size1, norm_size2)

        is_similar = position_diff < position_threshold and size_diff < size_threshold
        return is_similar, position_diff, size_diff

    def calculate_combined_similarity_with_filter(row1, row2, width, height, position_threshold, size_threshold):
        face1 = get_keypoints(row1, 0, 5)
        face2 = get_keypoints(row2, 0, 5)
        body1 = get_keypoints(row1, 15, 7)
        body2 = get_keypoints(row2, 15, 7)
        leg1 = get_keypoints(row1, 36, 4)
        leg2 = get_keypoints(row2, 36, 4)

        keypoints1 = face1 + body1 + leg1
        keypoints2 = face2 + body2 + leg2

        is_similar, position_diff, size_diff = filter_by_position_and_size(keypoints1, keypoints2, width, height,
                                                                           position_threshold, size_threshold)
        if not is_similar:
            return float('inf'), position_diff, size_diff  # Return a high distance if not similar in position/size

        # Calculate pose similarity if position and size are similar
        similarity = calculate_combined_similarity(row1, row2, width, height)
        return similarity, position_diff, size_diff

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

    def get_similar_frames_dict(results):
        frame_similarities = {}
        for frame_num in results:
            for result in results[frame_num]:
                file_pair = result['similar_files']
                for file in file_pair:
                    if file not in frame_similarities:
                        frame_similarities[file] = []
                    frame_similarities[file].append((frame_num, file_pair))
        return frame_similarities

    def find_max_transformation_order(frame_similarities, frame_count):
        # 현재 프레임과 파일을 추적
        current_frame = 0
        current_file = None
        transformation_order = []

        # 가능한 파일 목록
        files = list(frame_similarities.keys())

        while current_frame < frame_count:
            max_similar_frames = []
            max_file = None

            # 현재 프레임에서 변환 가능한 모든 파일 중 가장 많은 변환을 제공하는 파일을 선택
            for file in files:
                similar_frames = [frame for frame, pair in frame_similarities[file] if frame >= current_frame and (file != current_file)]
                if len(similar_frames) > len(max_similar_frames):
                    max_similar_frames = similar_frames
                    max_file = file

            if not max_similar_frames:
                break  # 변환 가능한 파일이 없으면 종료

            transformation_order.append((current_frame, max_file))
            current_file = max_file
            current_frame = max_similar_frames[0] + 5  # 최소 5 프레임 후에 다시 변환 가능

        return transformation_order

    # 각 CSV 파일 데이터를 읽어들임
    data_list = [pd.read_csv(file) for file in csv_files]

    # 유사한 프레임을 저장할 딕셔너리
    similar_frames = {}

    # 모든 가능한 프레임 번호 가져오기
    all_frame_numbers = set()
    for data in data_list:
        all_frame_numbers.update(data.iloc[:, 0].unique())

    # 총 작업량 계산
    total_comparisons = len(all_frame_numbers) * len(data_list) * (len(data_list) - 1) // 2

    # 진행 상황 표시
    progress = tqdm(total=total_comparisons, desc="Comparing frames")

    # 유사 프레임 계산
    for frame_num in all_frame_numbers:
        for i, data1 in enumerate(data_list):
            frame1_rows = data1[data1.iloc[:, 0] == frame_num].to_numpy()
            for j, data2 in enumerate(data_list):
                if i >= j:
                    continue

                frame2_rows = data2[data2.iloc[:, 0] == frame_num].to_numpy()

                for row1 in frame1_rows:
                    for row2 in frame2_rows:
                        similarity, position_diff, size_diff = calculate_combined_similarity_with_filter(row1, row2, width, height, position_threshold, size_threshold)

                        if similarity < threshold:
                            key = (frame_num, csv_files[i], csv_files[j])
                            if key not in similar_frames:
                                similar_frames[key] = []
                            similar_frames[key].append((similarity, position_diff, size_diff))

                    progress.update(1)

    progress.close()

    # 결과를 변수로 저장
    results = {}

    for (frame_num, csv_file1, csv_file2), values in similar_frames.items():
        avg_similarity = sum(val[0] for val in values) / len(values)
        avg_position_diff = sum(val[1] for val in values) / len(values)
        avg_size_diff = sum(val[2] for val in values) / len(values)
        if avg_similarity < avg_similarity_threshold:
            continue

        if frame_num not in results:
            results[frame_num] = []

        results[frame_num].append({
            "frame_num": frame_num,
            "similar_files": (csv_file1, csv_file2),
            "avg_similarity": avg_similarity,
            "avg_position_diff": avg_position_diff,
            "avg_size_diff": avg_size_diff,
            "similar_person_count": len(values)
        })

    # 유사한 프레임을 저장할 딕셔너리
    frame_similarities = get_similar_frames_dict(results)

    # 전체 프레임 수 계산
    frame_count = max(all_frame_numbers) + 1

    # 최대 변환 순서를 찾기 위한 함수 호출
    max_transformation_order = find_max_transformation_order(frame_similarities, frame_count)

    return results, max_transformation_order
