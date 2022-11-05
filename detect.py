
from numpy.linalg import norm
from re import X
import threading
import cv2
import os
from facenet_pytorch import InceptionResnetV1
# import tensorflow as tf
import time 
import torch

import cv2
import numpy as np
import cv2

from align_faces import warp_and_crop_face, get_reference_facial_points
from mtcnn.detector import MtcnnDetector

detector = MtcnnDetector()

def mask_detect(image):
    img = image.copy()
    (h,w) = image.shape[:2]
    boxes, facial5points = detector.detect_faces(image)

    if(len(boxes)!=0):
        for box in boxes:
            (startX,startY,endX,endY)=box[:4].astype('int')
            #ensure the bounding boxes fall within the dimensions of the frame
            (startX,startY)=(max(0,startX),max(0,startY))
            (endX,endY)=(min(w-1,endX), min(h-1,endY))

            #extract the face ROI, convert it from BGR to RGB channel, resize it to 224,224 and preprocess it
            face=img[startY:endY, startX:endX]

            color = (255,0,0)
            cv2.rectangle(img,(startX,startY),(endX,endY),color,2)
    return img


def PlayCamera(id):    
    video_capture = cv2.VideoCapture(id)
    while True:
        x = time.time()
        ret, frame = video_capture.read()
        # img = frame[0:128,0:128]
        # print(model.predict(np.array([img])))
        # img = mask_detect(frame)
        img = mask_detect(frame)
        print(time.time() - x)
        cv2.imshow('{}'.format(id), img)        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
def run():
    cameraIDs = [0]
    threads = []
    for id in cameraIDs:
        threads += [threading.Thread(target=PlayCamera, args=(id,))]
    for t in threads:    
        t.start()
    for t in threads: 
        t.join()
if __name__ == '__main__':
    run()
