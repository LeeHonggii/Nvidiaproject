# import cv2
# import numpy as np
# import os
# import insightface
# from insightface.app import FaceAnalysis
# import time

# # Initialize FaceAnalysis with specified modules and provider
# app = FaceAnalysis(
#     allowed_modules=["detection", "landmark_2d_106"],
#     providers=["CUDAExecutionProvider"],
# )
# app.prepare(ctx_id=0, det_size=(640, 640))

# # Load an image into 'img' and ensure 'frame' is a copy of 'img' for drawing
# img = cv2.imread("data/aespa3.jpg")  # Ensure to provide the correct path
# frame = img.copy()

# # Initialize list to store face positions
# face_positions = []

# # Detect faces in the image
# faces = app.get(img)
# for face in faces:
#     bbox = face["bbox"].astype(int)
#     cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
#     face_positions.append((bbox[0], bbox[1], bbox[2], bbox[3]))

#     if "landmark_2d_106" in face:
#         lmk = face["landmark_2d_106"]
#         lmk = np.round(lmk).astype(np.int64)
#         for point in lmk:
#             cv2.circle(frame, tuple(point), 3, (200, 160, 75), 1, cv2.LINE_AA)
#         print(f"Landmarks for the face: {lmk}")

#         # Transform points if necessary, for example:
#         # transformed_points = some_transformation_function(lmk)
#         # then convert them into tuples
#         transformed_points = lmk  # Placeholder for actual transformation logic
#         transformed_points_tuples = [tuple(point) for point in transformed_points]
#         print(f"Transformed points: {transformed_points_tuples}")


# # Show the resulting frame cv2.imshow("Frame with Detections", frame)
# cv2.waitKey(0)  # Wait for a key press to close
# cv2.destroyAllWindows()
