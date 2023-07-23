import csv

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
model_path = './hand_landmarker.task'
import os
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = HandLandmarkerOptions(
    num_hands=1,
    base_options=BaseOptions(model_asset_path='./hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE)
landmarker=HandLandmarker.create_from_options(options)



csv_file_path = 'traindata.csv'
img_dir="./train"
labels=['paper', 'rock', 'scissors']
sub_file=""
data_list = []
for idx in labels:
    print(idx)
    dir=os.path.join(img_dir,idx)
    imgs=os.listdir(dir)
    for img in imgs:
        mp_image = mp.Image.create_from_file(os.path.join(dir,img))
        hand_landmarker_result = landmarker.detect(mp_image)
        num=len(hand_landmarker_result.handedness)
        if num > 0:
            for hand_world_landmarks in hand_landmarker_result.hand_world_landmarks:
                point_list = []
                for point in hand_world_landmarks:
                    x = point.x
                    y = point.y
                    z=point.z
                    point_list.append(x)
                    point_list.append(y)
                    point_list.append(z)
                point_list.insert(0,labels.index(idx))
                data_list.append(point_list)

with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['class', 'x1', 'y1', 'z1','x2', 'y2', 'z2', 'x3', 'y3', 'z3', 'x4', 'y4', 'z4', 'x5', 'y5', 'z5', 'x6', 'y6', 'z6', 'x7', 'y7', 'z7', 'x8', 'y8', 'z8', 'x9', 'y9', 'z9', 'x10', 'y10', 'z10', 'x11', 'y11', 'z11', 'x12', 'y12', 'z12', 'x13', 'y13', 'z13', 'x14', 'y14', 'z14', 'x15', 'y15', 'z15', 'x16', 'y16', 'z16', 'x17', 'y17', 'z17', 'x18', 'y18', 'z18', 'x19', 'y19', 'z19', 'x20', 'y20', 'z20', 'x21', 'y21', 'z21'])
    for data_row in data_list:
        csv_writer.writerow(data_row)