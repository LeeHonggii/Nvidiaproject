# install cmake
# pip3 install insightface

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os

# Check if the image file exists
img_path = "C:/Users/hancomtst/Desktop/Nvidiaproject/data/youjin1.png"
if not os.path.exists(img_path):
    raise FileNotFoundError(f"No file found at {img_path}")

# Load the image
img = cv2.imread(img_path)
if img is None:
    raise Exception(
        "Failed to load image. Check if the path is correct and the file is a valid image."
    )

# Initialize the FaceAnalysis app
app = FaceAnalysis(allowed_modules=["detection"])
app.prepare(ctx_id=-1, det_size=(640, 640))

# Perform face detection
faces = app.get(img)

# Draw bounding boxes on the original image
for face in faces:
    bbox = face["bbox"].astype(int)
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

# Save the result
output_path = "./youjin1.jpg"
cv2.imwrite(output_path, img)

print(f"Output saved to {output_path}")
