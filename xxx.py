import cv2
id = "../train/videos/7.mp4"
video_capture = cv2.VideoCapture(id)
width  = int(video_capture.get(3))
height = int(video_capture.get(4))
if height > width:
    scale = height / width
else:
    scale = width / height
while height>1080 and width>1080: 
    height = int(height/scale)
    width = int(width/scale)
while True:
    ret, frame = video_capture.read()
    frame = cv2.resize(frame,(width,height))
    cv2.imshow("img",frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        video_capture.release()
        cv2.destroyAllWindows()
        break