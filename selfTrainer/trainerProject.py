import cv2
import numpy as np
import mediapipe as mp
import time
import poseModule as pm

cap = cv2.VideoCapture(0)
detector = pm.poseDetector()
count = 0
fromDown = 1
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('test1.mp4', fourcc, 20.0, (1280, 720))


while True:
    success, img = cap.read()
    img = cv2.resize(img, (1280,720))

    img = detector.findPose(img, draw=False)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)
    if len(lmList) != 0:
        
        # left arm
        angle = detector.findAngle(img, 11, 13, 15)
        # right arm
        # angle = detector.findAngle(img, 12, 14, 16)

        percentage = 100-np.interp(angle, (1,179), (0,100))
        bar = np.interp(percentage, (15,75), (650, 100))
        barPercentage = np.interp(percentage, (15,75), (0, 100))
        # print(percentage)

    if fromDown == 1:
        if percentage >= 75:
            count += 0.5
            fromDown = 0

        cv2.putText(img, 'UP', (70, 150), 
            cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 5)

        cv2.rectangle(img, (1100, int(bar)), (1175, 650), (255,255,0), cv2.FILLED)

    if fromDown == 0:
        if percentage <= 15:
            count += 0.5
            fromDown = 1

        cv2.putText(img, 'DOWN', (70, 150), 
            cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 5)

        cv2.rectangle(img, (1100, int(bar)), (1175, 650), (255,255,0), cv2.FILLED)

    # draw bar
    cv2.rectangle(img, (1100, 100), (1175, 650), (0,0,0), 3)
    cv2.putText(img, f'{int(barPercentage)}%', (1100,75), 
                cv2.FONT_HERSHEY_PLAIN, 3, (0,0,0), 3)
    
    
    cv2.putText(img, str(int(count)), (70, 100), 
                cv2.FONT_HERSHEY_PLAIN, 5, (255,255,255), 7)

    # out.write(img)

    cv2.imshow("Image", img)
    cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

# out.release()