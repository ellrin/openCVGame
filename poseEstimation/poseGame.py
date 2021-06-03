import cv2
import numpy as np
import time
import poseModule as pm
from PIL import ImageGrab


pTime = 0
detector = pm.poseDetector()

while True:
    img_init = ImageGrab.grab()
    img_np = np.array(img_init)
    img = detector.findPose(img_np)

    lmList = detector.findPosition(img)
    print(lmList)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,0), 3)
    
    cv2.imshow("Screen", img)
    cv2.waitKey(1)