from pupil_detectors import Detector2D
from pure_detector import PuReDetector
from pure_original import PuReOriginal

import eye_seg

import cv2
import pandas as pd
import numpy as np
import time

import LPW
LPW.LPW_path = "/media/pfa/P L/pfa/datasets/LPW"

detectors = {
    "2d": Detector2D(),
    "pure_pfa": PuReDetector(),
    "pure_orig": PuReOriginal(),
}

for name, detector in detectors.items():
    data = []

    method = f"{name}"

    for subject, video_id, n, target, frame in LPW.video_iterator():
        if n % 100 == 0:
            print(name, subject, video_id, n)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        t1 = time.perf_counter()
        result = detector.detect(gray)
        t2 = time.perf_counter()

        entry = {
            "subject": subject,
            "video_id": video_id,
            "frame": n,
            "time": t2 - t1,
            "method": method,
            "target": target,
            "confidence": result["confidence"],
        }

        if name != "pure_orig":
            entry["center"] = np.array(result["ellipse"]["center"])
            entry["axes"] = np.array(result["ellipse"]["axes"])
            entry["angle"] = result["ellipse"]["angle"]
        else:
            entry["center"] = np.array((result["center_x"], result["center_y"]))
            entry["axes"] = np.array((result["first_ax"], result["second_ax"]))
            entry["angle"] = result["angle"]

        data.append(entry)

    df = pd.DataFrame(data)
    df.to_pickle(f"benchmark.lpw.{method}.pkl")

