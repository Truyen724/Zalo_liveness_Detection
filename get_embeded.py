
from numpy.linalg import norm
from re import X
import threading
import cv2
import os
from sklearn.metrics.pairwise import cosine_similarity
# import tensorflow as tf
import time 
import pickle

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


len_of_lst = 160
def get_sim_matrix(lst_img):
    lst_reshape = lst_img.reshape(192,x)
    list_sim = cosine_similarity(lst_reshape, lst_reshape)
    return list_sim
def PlayCamera(id):    
    num_of_face = 0
    video_capture = cv2.VideoCapture(id)
    face = None
    list_face = np.zeros(shape = (len_of_lst, 256, 256,3), dtype=np.uint8)
    while True:
        ret, frame = video_capture.read()
        img, face = mask_detect(frame)
        list_face[num_of_face] = face
        num_of_face +=1      
        # cv2.imshow("img",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            video_capture.release()
            cv2.destroyAllWindows()
            break
            
    # video_capture.release()
    cv2.destroyAllWindows()
    return list_face
name_dataset = "../train/videos/"
lst_data = os.listdir("../train/videos/")
def get_embeded(lst_data, name_out = "train" ):
    lst_out = []
    for name in lst_data:
        scr = name_dataset + name

        x = PlayCamera(scr)
        lib = {
            "embed":x,
            "name":name
        }
        out = (name)
        lst_out.append()
    pass
if __name__ == '__main__':
    PlayCamera(0)
