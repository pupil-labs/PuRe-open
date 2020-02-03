from pure_detector import PuReDetector

import cv2

cap = cv2.VideoCapture("../eye0.mp4")

success, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

d = PuReDetector()

result, debug = d.detect_debug(gray)

cv2.imshow("img", frame)
cv2.imshow("debug", debug)
cv2.waitKey(-1)

