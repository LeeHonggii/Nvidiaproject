import cv2
import numpy as np
import os
import insightface
from insightface.app import FaceAnalysis
import time

def process_video(video_path, output_path):
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

    app = FaceAnalysis(allowed_modules=["detection", "landmark_2d_106"], providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    window_name = "Video"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if display_scale_factor != 1:
        cv2.resizeWindow(window_name, int(width * display_scale_factor), int(height * display_scale_factor))

    frame_count = 0
    face_positions = []
    face_recognitions = []
    previous_embeddings = []

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 5 == 0:
            faces = app.get(frame)
            current_embeddings = []

            for face in faces:
                bbox = face["bbox"].astype(int)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                face_positions.append((bbox[0], bbox[1], bbox[2], bbox[3]))

                # Print bounding box and any other face details
                print(f"Frame {frame_count}: Face detected with bbox={bbox}")

                if 'landmark_2d_106' in face:
                    lmk = face["landmark_2d_106"]
                    lmk = np.round(lmk).astype(np.int64)
                    for point in lmk:
                        cv2.circle(frame, tuple(point), 3, (200, 160, 75), 1, cv2.LINE_AA)
                    # Print landmarks
                    print(f"Landmarks for the face: {lmk}")

                if 'normed_embedding' in face:
                    embedding = face['normed_embedding']
                    current_embeddings.append(embedding)
                    face_recognitions.append(embedding)
                    # Print embedding vector
                    print(f"Embedding vector for detected face: {embedding}")

                    # Compare with previous embeddings
                    if previous_embeddings:
                        similarity_scores = np.dot(embedding, np.array(previous_embeddings).T)
                        max_similarity = np.max(similarity_scores)
                        if max_similarity > 0.6:  # Assuming 0.6 as the threshold for same person
                            print(f"Frame {frame_count}: Detected same person with similarity {max_similarity:.2f}.")

            previous_embeddings = current_embeddings  # Update the previous embeddings

            out.write(frame)
            if display_scale_factor != 1:
                display_frame = cv2.resize(frame, (int(width * display_scale_factor), int(height * display_scale_factor)))
                cv2.imshow(window_name, display_frame)
            else:
                cv2.imshow(window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    total_time = time.time() - start_time
    print(f"Video processing complete. Output saved to {output_path}")
    print(f"Total processing time: {total_time:.2f} seconds")

    return face_positions, face_recognitions

if __name__ == "__main__":
    video_path = "./video/ive_baddie_1.mp4"
    output_path = "./video/output_ive_baddie_1.mp4"
    face_positions, face_recognitions = process_video(video_path, output_path)
    print("Video processing complete. Output saved to", output_path)
    print("Face positions:", face_positions)
    print("Face recognitions:", face_recognitions)
