import cv2
import mediapipe as mp
from mediapipe.python import solutions
import numpy as np
import time
import math
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
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0,0,0), cv2.FILLED)

        return self.lmList


    def findAngle(self, img, p1, p2, p3, draw=True):
        # get the landmarks coordinates
        id1, x1, y1 = self.lmList[p1]
        id2, x2, y2 = self.lmList[p2]
        id3, x3, y3 = self.lmList[p3]

        # calculate the angle
        def getSideLength(coordinate1, coordinate2):
            lenx = abs(coordinate1[0] - coordinate2[0])
            leny = abs(coordinate1[1] - coordinate2[1])
            return (lenx**2+leny**2)**0.5
        
        triLen1 = getSideLength((x2,y2), (x3, y3))
        triLen2 = getSideLength((x1,y1), (x3, y3))
        triLen3 = getSideLength((x1,y1), (x2, y2))

        cosAng = (triLen1**2+triLen3**2-triLen2**2)/(2*triLen1*triLen3)
        angle = math.degrees(math.acos(cosAng))
        # print(angle)

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255,255,255), 5)
            cv2.line(img, (x2, y2), (x3, y3), (255,255,255), 5)
            cv2.circle(img, (x1, y1), 15, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 20, (0,0,255), 2)
            cv2.circle(img, (x1, y1), 25, (0,0,255), 1)
            cv2.circle(img, (x2, y2), 15, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 20, (0,0,255), 2)
            cv2.circle(img, (x2, y2), 25, (0,0,255), 1)
            cv2.circle(img, (x3, y3), 15, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 20, (0,0,255), 2)
            cv2.circle(img, (x3, y3), 25, (0,0,255), 1)
            cv2.putText(img, 'the angle is : '+str(int(angle)), (x2-100, y2+100), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)

        return angle




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