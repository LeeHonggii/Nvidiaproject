# from deepface import DeepFace
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2

# backends = [
#   'opencv',
#   'ssd',
#   'dlib',
#   'mtcnn',
#   'fastmtcnn',
#   'retinaface',
#   'mediapipe',
#   'yolov8',
#   'yunet',
#   'centerface',
# ]
# models = [
#   "VGG-Face",
#   "Facenet",
#   "Facenet512",
#   "OpenFace",
#   "DeepFace",
#   "DeepID",
#   "ArcFace",
#   "Dlib",
#   "SFace",
#   "GhostFaceNet",
# ]
# metrics = ["cosine", "euclidean", "euclidean_l2"]

# # Extract faces from each image
# faces_img1 = DeepFace.extract_faces(
#     img_path="data/aespa3.jpg", detector_backend="mtcnn"
# )
# faces_img2 = DeepFace.extract_faces(
#     img_path="data/aespa4.jpg", detector_backend="mtcnn"
# )

# # Function to safely attempt to display an image
# def safe_display(face_dict, index, image_label):
#     if "face" in face_dict and isinstance(face_dict["face"], np.ndarray):
#         print(
#             f"Displaying {image_label} Face {index}: shape {face_dict['face'].shape}, dtype {face_dict['face'].dtype}"
#         )
#         plt.imshow(face_dict["face"])
#     else:
#         print(
#             f"Error with {image_label} Face {index}: Expected numpy array in 'face' key, got {type(face_dict.get('face', None))}"
#         )
#         raise TypeError("Extracted face is not in the correct format for display.")

# # Display extracted faces (Optional step to manually select which faces to compare)
# print("Faces from Image 1:")
# for i, face_dict in enumerate(faces_img1):
#     plt.subplot(1, len(faces_img1), i + 1)
#     safe_display(face_dict, i + 1, "Img1")  # Using the safe display function
#     plt.title(f"Img1 Face {i+1}")
#     plt.axis("off")

# plt.show()

# print("Faces from Image 2:")
# for i, face_dict in enumerate(faces_img2):
#     plt.subplot(1, len(faces_img2), i + 1)
#     safe_display(face_dict, i + 1, "Img2")  # Using the safe display function
#     plt.title(f"Img2 Face {i+1}")
#     plt.axis("off")

# plt.show()

# def convert_to_rgb(image):
#     if image.ndim == 2:  # If the image is grayscale
#         return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
#     elif image.shape[2] == 4:  # If the image has an alpha channel
#         return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
#     return image  # If the image is already RGB

# def compare_faces(faces1, faces2):
#     results = []
#     for i, face_dict1 in enumerate(faces1):
#         face1 = face_dict1.get("face")
#         if face1 is not None:
#             face1 = convert_to_rgb(face1)
#         for j, face_dict2 in enumerate(faces2):
#             face2 = face_dict2.get("face")
#             if face2 is not None:
#                 face2 = convert_to_rgb(face2)
#             if isinstance(face1, np.ndarray) and isinstance(face2, np.ndarray):
#                 # Perform verification directly on the extracted face images
#                 result = DeepFace.verify(face1, face2, model_name=models[2], distance_metric=metrics[1], enforce_detection=False)
#                 results.append((i, j, result['verified'], result['distance'], face1, face2))
#     return results

# match_results = compare_faces(faces_img1, faces_img2)

# for res in match_results:
#     print(f"Face {res[0]+1} from Image 1 and Face {res[1]+1} from Image 2: {'Match' if res[2] else 'No Match'}, Distance: {res[3]:.2f}")
#     if res[2]:  # If it is a match, display the matching faces
#         plt.figure()
#         plt.subplot(1, 2, 1)
#         plt.imshow(res[4])
#         plt.title(f"Image 1 - Face {res[0]+1}")
#         plt.axis("off")

#         plt.subplot(1, 2, 2)
#         plt.imshow(res[5])
#         plt.title(f"Image 2 - Face {res[1]+1}")
#         plt.axis("off")

#         plt.show()
