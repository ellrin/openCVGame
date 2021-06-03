import cv2
import numpy as np
import time
import faceModule as fm
from PIL import ImageGrab


pTime = 0
detector = fm.FaceDetector()
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img, bboxs = detector.findFaces(img)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, 
                (255,255,255), 3 )
    cv2.imshow("Image", img)
    cv2.waitKey(1)