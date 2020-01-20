from pupil_detectors import Detector2D
from pure_detector import PuReDetector
# from pure_original import PuReOriginal

import eye_seg

import cv2
import pandas as pd
import numpy as np
import time
import platform

import LPW

detectors = {
    "2d": Detector2D(),
    "pure_pfa": PuReDetector(),
    # "pure_orig": PuReOriginal(),
}
smalls = [
    True,
    # False,
]

computer = platform.node()

for name, detector in detectors.items():
    for small in smalls:

        data = []

        method = ".".join([
            "benchmark",
            "lpw",
            "small" if small else "orig",
            computer,
            name,
        ])

        for subject, video_id, n, target, frame in LPW.video_iterator():
            if n % 100 == 0:
                print(name, subject, video_id, n)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if small:
                gray = cv2.resize(gray, (320, 240), interpolation=cv2.INTER_AREA)

            t1 = time.perf_counter()
            result = detector.detect(gray)
            t2 = time.perf_counter()

            entry = {
                "subject": subject,
                "video_id": video_id,
                "frame": n,
                "time": t2 - t1,
                "method": method,
                "target": np.array(target),
                "confidence": result["confidence"],
            }

            factor = 2.0 if small else 1.0

            if name != "pure_orig":
                entry["center"] = np.array(result["ellipse"]["center"]) * factor
                entry["axes"] = np.array(result["ellipse"]["axes"]) * factor
                entry["angle"] = result["ellipse"]["angle"]
            else:
                entry["center"] = np.array((result["center_x"], result["center_y"])) * factor
                entry["axes"] = np.array((result["first_ax"], result["second_ax"])) * factor
                entry["angle"] = result["angle"]

            data.append(entry)

        df = pd.DataFrame(data)
        df.to_pickle(f"{method}.pkl")

