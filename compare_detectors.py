from pupil_detectors import Detector2D
from pure_detector import PuReDetector

import LPW

import cv2
import pandas as pd
import numpy as np
import time


pupil_d = Detector2D()
pure_d = PuReDetector()

data = []

for subject, video_id, n, target, frame in LPW.video_iterator():
    if n % 100 == 0:
        print(subject, video_id, n)

    frame = cv2.resize(frame, (320, 240))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # t1 = time.perf_counter()
    # result_2d = pupil_d.detect(gray)
    t2 = time.perf_counter()
    result_pure = pure_d.detect(gray)
    t3 = time.perf_counter()




    # cv2.ellipse(
    #     frame,
    #     (int(result_pure["center_x"]), int(result_pure["center_y"])),
    #     (int(result_pure["first_ax"] / 2), int(result_pure["second_ax"] / 2)),
    #     int(result_pure["angle"]),
    #     0, 360, (0, 0, 255)
    # )

    # cv2.imshow("debug", frame)
    # cv2.waitKey(1)


    data.append({
        "subject": subject,
        "video": video_id,
        "frame": n,
        "target_x": target[0] / 2,
        "target_y": target[1] / 2,
        # "2d.confidence": result_2d["confidence"],
        # "2d.angle": result_2d["ellipse"]["angle"],
        # "2d.first_ax": result_2d["ellipse"]["axes"][0],
        # "2d.second_ax": result_2d["ellipse"]["axes"][1],
        # "2d.center_x": result_2d["ellipse"]["center"][0],
        # "2d.center_y": result_2d["ellipse"]["center"][1],
        # "2d.time": t2 - t1,
        "confidence": result_pure["confidence"],
        "angle": result_pure["angle"],
        "first_ax": result_pure["first_ax"],
        "second_ax": result_pure["second_ax"],
        "center_x": result_pure["center_x"],
        "center_y": result_pure["center_y"],
        "time": t3 - t2,
        "method": "pure.dirty_canny"
    })

df = pd.DataFrame(data)

df.to_pickle("data.pure.dirty_canny.pkl")

