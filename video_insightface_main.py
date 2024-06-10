import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
import time


def process_video(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"No file found at {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Failed to open video file.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    is_4k = width >= 1080 and height >= 1080
    display_scale_factor = 0.5 if is_4k else 1

    app = FaceAnalysis(
        allowed_modules=["detection", "landmark_2d_106"],
        providers=["CUDAExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    window_name = "Video"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if display_scale_factor != 1:
        cv2.resizeWindow(
            window_name,
            int(width * display_scale_factor),
            int(height * display_scale_factor),
        )

    frame_count = 0
    face_positions = []
    face_recognitions = []
    previous_embeddings = []
    eye_endpoint = []

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 5 == 0:
            faces = app.get(frame)
            current_embeddings = []
            current_frame_positions = []
            for face in faces:
                bbox = face["bbox"].astype(int)
                x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                current_frame_positions.append(
                    (x, y, w, h)
                )  # Append this face's bounding box as a tuple to the current frame list.

                # Print bounding box and any other face details
                print(f"Frame {frame_count}: Face detected with bbox={bbox}")

                if "landmark_2d_106" in face:
                    lmk = face["landmark_2d_106"]
                    lmk = np.round(lmk).astype(np.int64)
                    for point in lmk:
                        cv2.circle(
                            frame, tuple(point), 3, (200, 160, 75), 1, cv2.LINE_AA
                        )
                    # Print landmarks
                    # print(f"Landmarks for the face: {lmk}")
                    eye_endpoint.append((lmk[35], lmk[93]))

                if "normed_embedding" in face:
                    embedding = face["normed_embedding"]
                    current_embeddings.append(embedding)
                    face_recognitions.append(embedding)
                    # Print embedding vector
                    print(f"Embedding vector for detected face: {embedding}")

                    # Compare with previous embeddings
                    if previous_embeddings:
                        similarity_scores = np.dot(
                            embedding, np.array(previous_embeddings).T
                        )
                        max_similarity = np.max(similarity_scores)
                        if (
                            max_similarity > 0.6
                        ):  # Assuming 0.6 as the threshold for same person
                            print(
                                f"Frame {frame_count}: Detected same person with similarity {max_similarity:.2f}."
                            )

            face_positions.append(current_frame_positions)
            previous_embeddings = current_embeddings  # Update the previous embeddings

            if display_scale_factor != 1:
                display_frame = cv2.resize(
                    frame,
                    (
                        int(width * display_scale_factor),
                        int(height * display_scale_factor),
                    ),
                )
                cv2.imshow(window_name, display_frame)
            else:
                cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds")

    return face_positions, eye_endpoint, face_recognitions


def intersection_over_union(x1, y1, w1, h1, x2, y2, w2, h2):

    x1_max = x1 + w1
    y1_max = y1 + h1
    x2_max = x2 + w2
    y2_max = y2 + h2

    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)

    inter_area = inter_w * inter_h
    rect1_area = w1 * h1
    rect2_area = w2 * h2
    union_area = rect1_area + rect2_area - inter_area

    iou = inter_area / union_area

    return iou


def calculate_cosine_similarity(v1, v2):
    # 코사인 유사도 계산
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cosine_similarity = dot_product / (norm_v1 * norm_v2)
    return cosine_similarity


def find_matching_faces(
    face_positions1, eye_endpoints1, face_positions2, eye_endpoints2
):
    matched_faces = []
    min_length = min(len(face_positions1), len(face_positions2))
    for frame_index in range(min_length):
        frame_faces1 = face_positions1[frame_index]
        frame_faces2 = face_positions2[frame_index]
        min_faces_length = min(len(frame_faces1), len(frame_faces2))
        for i in range(min_faces_length):
            x1, y1, w1, h1 = frame_faces1[i]
            x2, y2, w2, h2 = frame_faces2[i]

            iou_score = intersection_over_union(x1, y1, w1, h1, x2, y2, w2, h2)

            if iou_score > 0.7:  # IOU threshold
                vector_1 = np.array(eye_endpoints1[frame_index][i][1]) - np.array(
                    eye_endpoints1[frame_index][i][0]
                )
                vector_2 = np.array(eye_endpoints2[frame_index][i][1]) - np.array(
                    eye_endpoints2[frame_index][i][0]
                )
                cosine_sim = calculate_cosine_similarity(vector_1, vector_2)

                if cosine_sim > 0.7:  # Cosine similarity threshold
                    matched_faces.append((frame_index, i, i))
                    print(
                        f"Matched Face at Frame {frame_index}: Index {i} with IOU {iou_score:.2f} and Cosine {cosine_sim:.2f}"
                    )

    return matched_faces


def recognition():
    pass


if __name__ == "__main__":
    video_path1 = "./video/pose_sync_ive_baddie_1.mp4"
    video_path2 = "./video/pose_sync_ive_baddie_2.mp4"
    face_positions1, eye_endpoint1, face_recognitions1 = process_video(video_path1)
    face_positions2, eye_endpoint2, face_recognitions2 = process_video(video_path2)

    matched_faces = find_matching_faces(
        face_positions1, eye_endpoint1, face_positions2, eye_endpoint2
    )
    print(matched_faces)
    # print("Face positions:", face_positions)
    # print("Face recognitions:", face_recognitions)
