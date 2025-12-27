import cv2
import mediapipe as mp
import time

class handDetector(): 
    
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            model_complexity=1,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

        # ✅ Custom drawing styles (MORE VISIBLE)
        self.landmark_style = self.mpDraw.DrawingSpec(
            color=(0, 255, 0), thickness=2, circle_radius=6
        )
        self.connection_style = self.mpDraw.DrawingSpec(
            color=(255, 0, 0), thickness=3
        )

        
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img,
                        handLms,
                        self.mpHands.HAND_CONNECTIONS,
                        self.landmark_style,
                        self.connection_style
                    )
        return img

    
    def findPosition(self, img, handNo=0, draw=True):
        lmList = []

        if self.results.multi_hand_landmarks and handNo < len(self.results.multi_hand_landmarks):
            myHand = self.results.multi_hand_landmarks[handNo]

            h, w, _ = img.shape
            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

                if draw:
                    # ✅ Bigger & brighter landmark points
                    cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)
                    cv2.putText(
                        img, str(id), (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1
                    )

        return lmList
