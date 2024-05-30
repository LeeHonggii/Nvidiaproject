import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

# Initialize FaceAnalysis for detection
app = FaceAnalysis(providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))

# Load recognition model
model_path = "buffalo_l\w600k_r50.onnx"  # Make sure this path is correct
recognizer = insightface.model_zoo.get_model(model_path)
recognizer.prepare(ctx_id=-1)  # Using CPU

# Load image
img_path = "friends.jpg"
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"No file found at {img_path}")

# Detect faces
faces = app.get(img)

# Assuming a method to obtain embeddings exists and is named appropriately
for face in faces:
    aligned_face = face.normed_image  # This is the aligned and cropped face image
    try:
        embedding = recognizer.infer(aligned_face)  # Hypothetical method
    except AttributeError:
        print(
            "Method not found. Check the correct method for embedding in the documentation."
        )

    # Draw bounding box
    bbox = face.bbox.astype(int)
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

# Save and display results
output_path = "./friends.jpg"
cv2.imwrite(output_path, img)
print(f"Output saved to {output_path}")
