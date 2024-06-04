#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 14:41:57 2019

@author: AIRocker
"""
import cv2
from model import bodypose_model, PoseEstimationWithMobileNet
from Demo_picture import Net_Prediction
from utils.util import *
import time
import torch
import argparse


def get_limb_coordinates(candidate, subset):
    limb_coordinates = []

    for person in subset:
        person_coords = []
        for i, index in enumerate(person[:-2]):
            if index != -1:
                x, y = candidate[int(index)][:2]
                person_coords.append((i, x, y))
            else:
                person_coords.append((i, None, None))
        limb_coordinates.append(person_coords)

    return limb_coordinates

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Open Pose Demo')
    parser.add_argument("-backbone", help='CMU, Mobilenet', default='CMU', type=str)
    parser.add_argument("-video", help='video path', default='images/ive_baddie_6.mp4', type=str)
    parser.add_argument("-scale", help='image to scale',default=1, type=float)
    parser.add_argument("-show", nargs='+', help="types to show: -1 shows the skeletons, or idx for specific part", default = (-1, 2), type=int)
    parser.add_argument("-thre", help="threshold for heatmap part",default=0.2, type=str)

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.backbone == 'CMU':
        model = bodypose_model().to(device)
        model.load_state_dict(torch.load('weights/body_pose_model.pth', map_location=lambda storage, loc: storage), strict=False)
    elif args.backbone == 'Mobilenet':
        model = PoseEstimationWithMobileNet().to(device)
        model.load_state_dict(torch.load('weights/MobileNet_bodypose_model', map_location=lambda storage, loc: storage), strict=False)

    model.eval()
    print('openpose {} model is successfully loaded...'.format(args.backbone))
    print(device)


    total_coordinates = []
    cap = cv2.VideoCapture(args.video)

    _, frame = cap.read()
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 15.0, (int(frame.shape[1]), int(frame.shape[0])), isColor=True)

    frame_count = 0
    while True:

        isSuccess, frame = cap.read()
        if isSuccess:
            if frame_count % 5 == 0:
                try:
                    start_time = time.time()
                    imageToTest = cv2.resize(frame, (0, 0), fx=args.scale, fy=args.scale, interpolation=cv2.INTER_CUBIC)
                    heatmap, paf = Net_Prediction(model, imageToTest, device, backbone = args.backbone)
                    all_peaks = peaks(heatmap, args.thre)

                    if args.show[0] == -1:
                        frame_limb_coordinates = []
                        connection_all, special_k = connection(all_peaks, paf, imageToTest)
                        candidate, subset = merge(all_peaks, connection_all, special_k)
                        canvas = draw_bodypose(frame, candidate, subset, args.scale)


                        limb_coordinates = get_limb_coordinates(candidate, subset)
                        frame_limb_coordinates.append(frame_count)
                        frame_limb_coordinates.append(limb_coordinates)
                        print(frame_limb_coordinates)
                        total_coordinates.append(frame_limb_coordinates)
                        frame_count2 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        # print("for frame : " + str(frame_count) + ',' + str(frame_count2) + "   timestamp is: ", str(cap.get(cv2.CAP_PROP_POS_MSEC)))

                    else:
                        canvas = draw_part(frame, all_peaks, args.show, args.scale)

                    # FPS = 1.0 / (time.time() - start_time)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(canvas, 'FPS: {:.1f}'.format(fps), (10, 20), font, 0.5, (0,255,255), 1, cv2.LINE_AA)

                    out.write(canvas)


                except:
                    print('detect error')

        frame_count += 1

        cv2.imshow('video', canvas)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break



    cap.release()
    out.release()
    cv2.destroyAllWindows()
    # print(total_coordinates)


