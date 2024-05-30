import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt

scaler = 0.3

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture("video/Adore_U.mp4")

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))

    faces = detector(img)
    if len(faces) == 0:
        continue

    face = faces[0]
    dlib_shape = predictor(img, face)
    shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

    top_left = np.min(shape_2d, axis=0)
    bottom_right = np.max(shape_2d, axis=0)
    center_x, center_y = np.mean(shape_2d, axis=0).astype(int)

    img = cv2.rectangle(
        img,
        (face.left(), face.top()),
        (face.right(), face.bottom()),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for s in shape_2d:
        cv2.circle(img, tuple(s), 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(img, tuple(top_left), 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.circle(img, tuple(bottom_right), 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.circle(img, (center_x, center_y), 1, (0, 0, 255), 2, cv2.LINE_AA)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Detected Faces")
    plt.show()
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
