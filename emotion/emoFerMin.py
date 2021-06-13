import cv2
import numpy as np
import time
from PIL import ImageGrab
from fer import FER

pTime = 0
detector = FER()

def fancy_draw(img, bbox, l=20, t=3, rt=1):
    x, y, w, h = bbox
    x1, y1 = x+w, y+h
    cv2.rectangle(img, bbox, (255,255,0), rt)
    cv2.line(img, (x, y), (x+l, y), (255,255,0), t)
    cv2.line(img, (x, y), (x, y+l), (255,255,0), t)

    cv2.line(img, (x1, y), (x1-l, y), (255,255,0), t)
    cv2.line(img, (x1, y), (x1, y+l), (255,255,0), t)

    cv2.line(img, (x, y1), (x+l, y1), (255,255,0), t)
    cv2.line(img, (x, y1), (x, y1-l), (255,255,0), t)

    cv2.line(img, (x1, y1), (x1-l, y1), (255,255,0), t)
    cv2.line(img, (x1, y1), (x1, y1-l), (255,255,0), t)

    return img


def drawBar(img, emoList):

    def getBar(emotion):
        emotionNum = emoList[0]['emotions'][emotion]
        bar = int(np.interp(emotionNum, (0,1), (150, 700)))
        return bar, int(emotionNum*100)
    
    angryBar, angryEmoNum = getBar('angry')
    disgustBar, disgustEmoNum = getBar('disgust')
    fearBar, fearEmoNum = getBar('fear')
    happyBar, happyEmoNum = getBar('happy')
    sadBar, sadEmoNum = getBar('sad')
    surpriseBar, surpriseEmoNum = getBar('surprise')
    neutralBar, neutralEmoNum = getBar('neutral')

    for i in range(7):
        cv2.rectangle(img, (70, 650-30*i), (700, 630-30*i), (255,255,255), cv2.FILLED)

    cv2.rectangle(img, (150, 650), (angryBar, 630), (255,100,0), cv2.FILLED)
    cv2.rectangle(img, (150, 620), (disgustBar, 600), (255,120,40), cv2.FILLED)
    cv2.rectangle(img, (150, 590), (fearBar, 570), (255,140,80), cv2.FILLED)
    cv2.rectangle(img, (150, 560), (happyBar, 540), (255,160,120), cv2.FILLED)
    cv2.rectangle(img, (150, 530), (sadBar, 510), (255,180,160), cv2.FILLED)
    cv2.rectangle(img, (150, 500), (surpriseBar, 480), (255,200,200), cv2.FILLED)
    cv2.rectangle(img, (150, 470), (neutralBar, 450), (255,220,240), cv2.FILLED)

    cv2.putText(img, f'angry    :{angryEmoNum}%', (70, 640+2), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2)
    cv2.putText(img, f'disgust  :{disgustEmoNum}%', (70, 610+2), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2)
    cv2.putText(img, f'fear     :{fearEmoNum}%', (70, 580+2), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2)
    cv2.putText(img, f'happy    :{happyEmoNum}%', (70, 550+2), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2)
    cv2.putText(img, f'sad      :{sadEmoNum}%', (70, 520+2), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2)
    cv2.putText(img, f'surprise :{surpriseEmoNum}%', (70, 490+2), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2)
    cv2.putText(img, f'neutral  :{neutralEmoNum}%', (70, 460+2), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2)

    return img

while True:
    img_init = ImageGrab.grab()
    img_np = np.array(img_init)
    
    img = cv2.resize(img_np, (1280,720))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    emolist = detector.detect_emotions(img)
    if emolist:
        
        print(emolist[0])
        bbox = emolist[0]['box']
        
        img = fancy_draw(img, bbox)
        img = drawBar(img, emolist)
    

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,0), 3)
    
    cv2.imshow("Screen", img)
    cv2.waitKey(1)


