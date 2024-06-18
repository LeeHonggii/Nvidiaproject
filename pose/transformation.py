def find_max_transformation_order(frame_similarities, frame_count, random_point):
    max_transitions = [0] * frame_count
    previous_frame = [-1] * frame_count
    previous_file = [None] * frame_count
    current_file = [None] * frame_count

    # 초기 파일 설정
    for file, similarities in frame_similarities.items():
        for frame, (next_frame, (file1, file2)) in enumerate(similarities):
            if frame < frame_count:
                max_transitions[frame] = 1
                previous_frame[frame] = -1
                previous_file[frame] = None
                current_file[frame] = file1

    # 각 프레임에 대해 반복
    for frame in range(frame_count):
        if current_file[frame] is None:
            continue

        for next_frame, (file1, file2) in frame_similarities.get(current_file[frame], []):
            if next_frame >= frame + random_point:
                # 다음 프레임으로의 전환이 중복되지 않도록 확인
                if next_frame < frame_count and max_transitions[next_frame] < max_transitions[frame] + 1 and current_file[frame] != file2:
                    max_transitions[next_frame] = max_transitions[frame] + 1
                    previous_frame[next_frame] = frame
                    previous_file[next_frame] = current_file[frame]
                    current_file[next_frame] = file2

    # 최대 전환 횟수를 가진 마지막 프레임을 찾음
    last_frame = max(range(frame_count), key=lambda x: max_transitions[x])
    transformation_order = []
    current_frame = last_frame

    # 최적 경로를 역추적
    while current_frame != -1:
        if previous_file[current_frame] is not None and current_file[current_frame] is not None:
            transformation_order.append(
                (current_frame, previous_file[current_frame], current_file[current_frame]))
        current_frame = previous_frame[current_frame]

    # 경로를 역순으로 정렬
    transformation_order.reverse()
    return transformation_order

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
