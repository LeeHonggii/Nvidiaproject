import face_recognition

picture_of_me = face_recognition.load_image_file("o_family.jpg")
my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]

# my_face_encoding now contains a universal 'encoding' of my facial features that can be compared to any other picture of a face!

unknown_picture = face_recognition.load_image_file("unknown.jpg")
unknown_face_encoding = face_recognition.face_encodings(unknown_picture)[0]

# Now we can see the two face encodings are of the same person with `compare_faces`!

results = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)

if results[0] == True:
    print("It's a picture of me!")
else:
    print("It's not a picture of me!")


# import face_recognition

# # Load the jpg files into numpy arrays
# biden_image = face_recognition.load_image_file("frineds.jpg")
# # obama_image = face_recognition.load_image_file("obama.jpg")
# # unknown_image = face_recognition.load_image_file("obama2.jpg")

# # Get the face encodings for each face in each image file
# # Since there could be more than one face in each image, it returns a list of encodings.
# # But since I know each image only has one face, I only care about the first encoding in each image, so I grab index 0.
# try:
#     biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
#     # obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
#     # unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
# except IndexError:
#     print(
#         "I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting..."
#     )
#     quit()

# known_faces = [biden_face_encoding]

# # results is an array of True/False telling if the unknown face matched anyone in the known_faces array
# results = face_recognition.compare_faces(known_faces)

# print("Is the unknown face a picture of Biden? {}".format(results[0]))
# # print("Is the unknown face a picture of Obama? {}".format(results[1]))
# print(
#     "Is the unknown face a new person that we've never seen before? {}".format(
#         not True in results
#     )
# )
