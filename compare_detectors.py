from pupil_detectors import Detector2D
from pure_detector import PuReDetector

import LPW

import cv2
import pandas as pd

import time


pupil_d = Detector2D()
pure_d = PuReDetector()

data = []

for subject, video_id, n, target, frame in LPW.video_iterator():
    print(subject, video_id, n)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    t1 = time.perf_counter()
    result_2d = pupil_d.detect(gray)
    t2 = time.perf_counter()
    result_pure = pure_d.detect(gray)
    t3 = time.perf_counter()


    data.append({
        "subject": subject,
        "video": video_id,
        "frame": n,
        "target_x": target[0],
        "target_y": target[1],
        "2d.confidence": result_2d["confidence"],
        "2d.angle": result_2d["ellipse"]["angle"],
        "2d.first_ax": result_2d["ellipse"]["axes"][0],
        "2d.second_ax": result_2d["ellipse"]["axes"][1],
        "2d.center_x": result_2d["ellipse"]["center"][0],
        "2d.center_y": result_2d["ellipse"]["center"][1],
        "2d.time": t2 - t1,
        "pure.confidence": result_pure["confidence"],
        "pure.angle": result_pure["angle"],
        "pure.first_ax": result_pure["first_ax"],
        "pure.second_ax": result_pure["second_ax"],
        "pure.center_x": result_pure["center_x"],
        "pure.center_y": result_pure["center_y"],
        "pure.time": t3 - t2,
    })

df = pd.DataFrame(data)

df.to_pickle("data.pkl")

