import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import time

# Load the model
model = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite0/detection/1")

labels_mapping = {1:'person',2:'bicycle',3:'car',4:'motorcycle',5:'airplane',6:'bus',7:'train',8:'truck',9:'boat',10:'traffic light',
                    11:'fire hydrant',12:'street sign',13:'stop sign',14:'parking meter',15:'bench',16:'bird',17:'cat',18:'dog',19:'horse',20:'sheep',
                    21:'cow',22:'elephant',23:'bear',24:'zebra',25:'giraffe',26:'hat',27:'backpack',28:'umbrella',29:'shoe',30:'eye glasses',
                    31:'handbag',32:'tie',33:'suitcase',34:'frisbee',35:'skis',36:'snowboard',37:'sports ball',38:'kite',39:'baseball bat',40:'baseball glove',
                    41:'skateboard',42:'surfboard',43:'tennis racket',44:'bottle',45:'plate',46:'wine glass',47:'cup',48:'fork',49:'knife',50:'spoon',
                    51:'bowl',52:'banana',53:'apple',54:'sandwich',55:'orange',56:'broccoli',57:'carrot',58:'hot dog',59:'pizza',60:'donut',
                    61:'cake',62:'chair',63:'couch',64:'potted plant',65:'bed',66:'mirror',67:'dining table',68:'window',69:'desk',70:'toilet',
                    71:'door',72:'tv',73:'laptop',74:'mouse',75:'remote',76:'keyboard',77:'cell phone',78:'microwave',79:'oven',80:'toaster',
                    81:'sink',82:'refrigerator',83:'blender',84:'book',85:'clock',86:'vase',87:'scissors',88:'teddy bear',89:'hair drier',90:'toothbrush',
                    91:'hair brush'}

def detect_objects(model, img_array, threshold, max_objects=100, print_time=True):
    img_copy = img_array.copy()
    green = (0, 255, 0)
    red = (0, 0, 255)
    tensor_img = tf.convert_to_tensor(img_array, dtype=tf.uint8)[tf.newaxis, ...]
    start = time.time()
    boxes, scores, classes, num_detections = model(tensor_img)
    boxes = boxes.numpy()
    scores = scores.numpy()
    classes = classes.numpy()
    num_detections = num_detections.numpy()

    for i in range(int(num_detections[0])):
        if scores[0, i] < threshold:
            break
        box = boxes[0, i]
        left, top, right, bottom = box[1], box[0], box[3], box[2]
        class_id = classes[0, i]
        caption = "{}: {:.4f}".format(labels_mapping[class_id], scores[0, i])
        cv2.rectangle(img_copy, (int(left), int(top)), (int(right), int(bottom)), color=green, thickness=2)
        cv2.putText(img_copy, caption, (int(left), int(top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, red, 1)

    if print_time:
        print('Detection time:', round(time.time() - start, 2), "seconds")

    return img_copy

def process_video(video_path, output_path, threshold=0.3, max_objects=100):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = detect_objects(model, rgb_frame, threshold, max_objects, print_time=False)
        bgr_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)

        cv2.imshow('Frame', bgr_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Usage
process_video('test.mp4', 'output_video3.mp4', threshold=0.3)
