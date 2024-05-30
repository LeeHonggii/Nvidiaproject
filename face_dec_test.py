# pip3 install face_recognition


# import face_recognition
# from PIL import Image

# image = face_recognition.load_image_file("friends.jpg")
# face_locations = face_recognition.face_locations(image)

# print("I found {} face(s) in this photograph.".format(len(face_locations)))

# for face_location in face_locations:

#     # Print the location of each face in this image
#     top, right, bottom, left = face_location
#     print(
#         "A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(
#             top, left, bottom, right
#         )
#     )
#     # You can access the actual face itself like this:
#     face_image = image[top:bottom, left:right]
#     pil_image = Image.fromarray(face_image)
#     pil_image.show()


import face_recognition
from PIL import Image, ImageDraw

# Load the image file
image = face_recognition.load_image_file("friends.jpg")

# Find all face locations in the image
face_locations = face_recognition.face_locations(image)

# Convert the numpy array image into a PIL Image
pil_image = Image.fromarray(image)

# Create a PIL drawing object to be able to draw on the image
draw = ImageDraw.Draw(pil_image)

# Loop through each face found in the image
for face_location in face_locations:
    top, right, bottom, left = face_location
    # Draw a box around the face
    draw.rectangle([left, top, right, bottom], outline="red", width=4)

# Display the image on which we've drawn boxes
pil_image.show()
