import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxNum=2, detectCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxNum = maxNum
        self.detectCon = detectCon
        self.trackCon  = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxNum, 
                                        self.detectCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, 
                                               self.mpHands.HAND_CONNECTIONS)

        return img


    def findPosition(self, img, handNo=0, draw=True, drawNum=12):

        lmList = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:

            try:
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])

                    if draw:
                        if id == drawNum:
                            cv2.circle(img, (cx, cy), 10, (0,0,0), cv2.FILLED)


            except:
                pass

        return lmList




def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    cTime = 0
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, 
                    (255,255,0), 3 )

        cv2.imshow("ImageTest", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()