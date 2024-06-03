import face_recognition
import os
import cv2

known_faces_dir = "known_faces"
# unknown_faces_dir = "unknown_faces"
tolerance = 0.5
frame_thickness = 3
font_thickness = 2
model = "cnn"

video = cv2.VideoCapture(0) # could put in a filename

print("loading known faces")

known_faces = []
known_names = []


for name in os.listdir(known_faces_dir):
    for filename in os.listdir(f'{known_faces_dir}/{name}'):
        image = face_recognition.load_image_file(f'{known_faces_dir}/{name}/{filename}')
        encoding = face_recognition.face_encodings(image)[0]

        known_faces.append(encoding)
        known_names.append(name)

print("processing unknown faces")
while True:
    # print(filename)

    # image = face_recognition.load_image_file(f"{unknown_faces_dir}/{filename}")
    ret, image = video.read()

    locations = face_recognition.face_locations(image, model=model)
    encodings = face_recognition.face_encodings(image, locations)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, tolerance)
        match = None

        if True in results:
            match = known_names[results.index(True)]
            print(f"Match found: {match}")

            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            color = [0, 255, 0]
            cv2.rectangle(image, top_left, bottom_right, color, frame_thickness)
            
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2]+22)
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, ))

    cv2.imshow(filename, image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    # cv2.waitKey(10000)
    # cv2.deestroyWindow(filename)