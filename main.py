import cv2
import mediapipe as mp
import numpy as np
import torch
from ANN import ANN
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
colors = [
    (0, 128, 128),
    (255, 165, 0),
    (0, 191, 255),
    (139, 0, 139),
    (46, 139, 87),
    (0, 206, 209),
    (255, 0, 255),
    (255, 140, 0),
    (128, 0, 128),
    (128, 0, 0),
    (32, 178, 170),
    (255, 99, 71),
    (186, 85, 211),
    (0, 128, 0),
    (128, 0, 128),
    (255, 20, 147),
    (255, 215, 0),
    (0, 0, 128),
    (0, 250, 154),
    (128, 0, 128),
    (60, 179, 113)
]

cap=cv2.VideoCapture(0)
rock=cv2.imread("r.png")
paper=cv2.imread('p.png')
scissors=cv2.imread('s.png')
none=cv2.imread('none.png')
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
WINDOWS_WIDTH=cap.get(3)
WINDOWS_HEIGHT=cap.get(4)
frame=None
res=None
flag=0
model=ANN()
model.load_state_dict(torch.load("model.pth"))
model.eval()
labels=['paper', 'rock', 'scissors']
def Draw_Line(p1,p2,color=(0,0,255)):
    global frame
    cv2.line(frame,(p1[0],p1[1]),(p2[0],p2[1]),color=color,thickness=2)
def Draw_RangeLine(points,color=(0,0,255)):
    for idx in range(len(points)-1):
        Draw_Line(points[idx],points[idx+1],color=color)
def Draw_Handmarker(result:mp.tasks.vision.HandLandmarkerResult):
    global frame
    num=len(result.handedness)

    if num>0:
        for hand_landmarks in result.hand_landmarks:
            point_list = []
            for point,color in zip(hand_landmarks,colors):
                x=int(WINDOWS_WIDTH * point.x)
                y=int(WINDOWS_HEIGHT * point.y)
                z=point.z
                point_list.append([x,y,z])
                # cv2.circle(frame,(int(WINDOWS_WIDTH*x),int(WINDOWS_HEIGHT*y)),radius=5,color=color,thickness=-1)

            Draw_RangeLine(point_list[0:5])
            Draw_RangeLine(point_list[5:9])
            Draw_RangeLine(point_list[9:13])
            Draw_RangeLine(point_list[13:17])
            Draw_RangeLine(point_list[17:21])
            Draw_Line(point_list[0],point_list[5])
            Draw_Line(point_list[5],point_list[9])
            Draw_Line(point_list[9],point_list[13])
            Draw_Line(point_list[13],point_list[17])
            Draw_Line(point_list[0],point_list[17])
            for point,color in zip(point_list,colors):
                cv2.circle(frame, (point[0], point[1]), radius=3, color=color, thickness=-1)


def print_result(result:mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global frame
    global res
    frame=cv2.cvtColor(output_image.numpy_view(),cv2.COLOR_RGB2BGR)
    res=(result,timestamp_ms)


def Inference(result:mp.tasks.vision.HandLandmarkerResult,model):
    num = len(result.handedness)
    if num > 0:
        for hand_world_landmarks in result.hand_world_landmarks:
            point_list = []
            for point in hand_world_landmarks:
                x = point.x
                y = point.y
                z = point.z
                point_list.append(x)
                point_list.append(y)
                point_list.append(z)
            pred = model(torch.tensor(point_list))
            predicted=labels[pred.argmax(0)]
            return predicted

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    num_hands=1,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

landmarker=HandLandmarker.create_from_options(options)
while True:
    ret,frame=cap.read()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    result=landmarker.detect_async(mp_image,int(cap.get(0)))
    if res!=None:
        inf=Inference(res[0], model)
        cv2.putText(frame,"Result:"+str(inf),(0,25),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=1,color=(0,0,255))
        if inf==None:
            cv2.imshow('resist',none)
        elif inf=='paper':
            cv2.imshow('resist',scissors)
        elif inf=='rock':
            cv2.imshow('resist', paper)
        else:
            cv2.imshow('resist',rock)
        Draw_Handmarker(res[0])
    cv2.imshow('frame',frame)
    if(cv2.waitKey(1)&0xff)==27:
        break
cap.release()
cv2.destroyAllWindows()