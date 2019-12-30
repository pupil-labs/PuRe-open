from pathlib import Path

import cv2
import numpy as np

data_path = Path("../eye_segmentation_500K")

def video_iterator():
    for part in (1, 2):

        raw_video = data_path / f"p{part}_image.mp4"
        pupil_video = data_path / f"p{part}_pupil.mp4"

        raw_cap = cv2.VideoCapture(str(raw_video))
        pupil_cap = cv2.VideoCapture(str(pupil_video))

        n = 0
        while True:
            raw_success, raw_frame = raw_cap.read()
            pupil_success, pupil_frame = pupil_cap.read()
            
            if raw_success != pupil_success:
                raise RuntimeError("Error matching raw and pupil frames!")
            
            if not raw_success:
                print(f"part {part} end")
                break

            pupil_frame = cv2.cvtColor(pupil_frame, cv2.COLOR_BGR2GRAY)
            _, pupil_frame = cv2.threshold(pupil_frame, 127, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(pupil_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            if not contours:
                continue

            contour = max(contours, key=len)
            center, axes, angle = cv2.fitEllipse(contour)
        
            yield part, n, raw_frame, center, axes, angle
            
            n += 1