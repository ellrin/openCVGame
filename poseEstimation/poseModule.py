import cv2
import mediapipe as mp
from mediapipe.python import solutions
import numpy as np
import time
from PIL import ImageGrab


class poseDetector():
    def __init__(self, mode=False, complexity=0, smooth=True,
                 detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smooth, 
                                     self.detectionCon, self.trackCon)


    def findPose(self, img, draw=True):
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(imgRGB, self.results.pose_landmarks, 
                                           self.mpPose.POSE_CONNECTIONS)
        
        return imgRGB


    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0,0,0), cv2.FILLED)

        return lmList




def main():
    pTime = 0
    detector = poseDetector()

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


if __name__=="__main__":
    main()