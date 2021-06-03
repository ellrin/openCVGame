import cv2
import mediapipe as mp
from mediapipe.python import solutions
import numpy as np
import time
from PIL import ImageGrab

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()


pTime = 0
while True:
    img = ImageGrab.grab()
    img = np.array(img)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    
    if results.pose_landmarks:
        mpDraw.draw_landmarks(imgRGB, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(imgRGB, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 
                3, (255,255,0), 3)
    
    cv2.imshow("Screen", imgRGB)
    cv2.waitKey(1)