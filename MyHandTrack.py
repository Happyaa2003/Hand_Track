import cv2
import mediapipe as mp
import time
import handTRackModule as htm

pTime = 0
cap = cv2.VideoCapture(0)
detector = htm.handDetector()

while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        # ✅ SAFE ACCESS
        if len(lmList) > 4:
            print(lmList[4])  # Thumb tip landmark

        cTime = time.time()
        fps = 1 / (cTime - pTime) if pTime != 0 else 0
        pTime = cTime

        cv2.putText(
            img, str(int(fps)), (10, 70),
            cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2
        )
        
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

cap.release()
cv2.destroyAllWindows()