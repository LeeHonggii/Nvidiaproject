import torch
from torchvision import models
import torchvision.transforms as T
import numpy as np
import cv2
import plotly.express as px
import plotly.io as pio

# Set up plotly rendering
pio.renderers.default = 'png'
pio.templates.default = 'plotly_dark'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Torch using {DEVICE}')

# Load the Keypoint R-CNN model
model = models.detection.keypointrcnn_resnet50_fpn(
    weights=models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
).eval().to(DEVICE)

# Define keypoints
keypoints = [
    'nose', 'left_eye', 'right_eye',
    'left_ear', 'right_ear', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip',
    'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle'
]

# Transform function
transforms = T.Compose([
    T.ToTensor()
])


# Define the limbs based on keypoints
def get_limbs_from_keypoints(keypoints):
    limbs = [
        [keypoints.index("right_eye"), keypoints.index("nose")],
        [keypoints.index("right_eye"), keypoints.index("right_ear")],
        [keypoints.index("left_eye"), keypoints.index("nose")],
        [keypoints.index("left_eye"), keypoints.index("left_ear")],
        [keypoints.index("right_shoulder"), keypoints.index("right_elbow")],
        [keypoints.index("right_elbow"), keypoints.index("right_wrist")],
        [keypoints.index("left_shoulder"), keypoints.index("left_elbow")],
        [keypoints.index("left_elbow"), keypoints.index("left_wrist")],
        [keypoints.index("right_hip"), keypoints.index("right_knee")],
        [keypoints.index("right_knee"), keypoints.index("right_ankle")],
        [keypoints.index("left_hip"), keypoints.index("left_knee")],
        [keypoints.index("left_knee"), keypoints.index("left_ankle")],
        [keypoints.index("right_shoulder"), keypoints.index("left_shoulder")],
        [keypoints.index("right_hip"), keypoints.index("left_hip")],
        [keypoints.index("right_shoulder"), keypoints.index("right_hip")],
        [keypoints.index("left_shoulder"), keypoints.index("left_hip")],
    ]
    return limbs


# Function to draw the skeleton for each person detected
def draw_skeleton_per_person(
        img,
        all_keypoints,
        all_scores,
        confs,
        limbs,
        conf_threshold=0.9
):
    img_copy = img.copy()
    for person_id in range(len(all_keypoints)):
        if confs[person_id] > conf_threshold:
            keypoints = all_keypoints[person_id, ...]
            scores = all_scores[person_id, ...]
            for joint in limbs:
                pt1, pt2 = [tuple(map(int, keypoints[joint_idx, :2].cpu())) for joint_idx in joint]
                cv2.line(img_copy, pt1, pt2, (0, 255, 255), 3)
    return img_copy


# Load video
video_path = 'pose_sync_ive_baddie_1.mp4'
cap = cv2.VideoCapture(video_path)

# Define output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR to RGB
    input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transformed_input = transforms(input_image)[None, ...].to(DEVICE)

    # Get model predictions
    with torch.no_grad():
        output = model(transformed_input)

    # Draw skeleton on the frame
    output_image = draw_skeleton_per_person(
        img=input_image,
        all_keypoints=output[0]['keypoints'],
        all_scores=output[0]['keypoints_scores'],
        confs=output[0]['scores'],
        limbs=get_limbs_from_keypoints(keypoints)
    )

    # Convert RGB back to BGR
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    # Write the frame to the output video
    out.write(output_image)

    # Display the frame (optional)
    cv2.imshow('Pose Detection', output_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
