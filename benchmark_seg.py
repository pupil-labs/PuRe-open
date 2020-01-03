# from pupil_detectors import Detector2D
from pure_detector import PuReDetector
# from pure_original import PuReOriginal

import eye_seg

import cv2
import pandas as pd
import numpy as np
import time


detectors = {
    # "2d": Detector2D(),
    "pure_pfa_orig_canny": PuReDetector(),
    # "pure_orig": PuReOriginal(),
}

for large in [False]:
    for name, detector in detectors.items():
        data = []

        method = f"{name}"
        if large:
            method += ".large"

        for part, n, frame, center, axes, angle in eye_seg.video_iterator():
            if n % 100 == 0:
                print(name, part, n)

            if large:
                scale_factor = min(320 / frame.shape[1], 240 / frame.shape[0])
            else:
                scale_factor = 1.0

            frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            t1 = time.perf_counter()
            result = detector.detect(gray)
            t2 = time.perf_counter()

            entry = {
                "part": part,
                "frame": n,
                "time": t2 - t1,
                "method": method,
                "target_center": np.array(center) * scale_factor,
                "target_axes": np.array(axes) * scale_factor,
                "target_angle": angle,
                "confidence": result["confidence"],
            }

            if name == "2d":
                entry["center"] = np.array(result["ellipse"]["center"])
                entry["axes"] = np.array(result["ellipse"]["axes"])
                entry["angle"] = result["ellipse"]["angle"]
            else:
                entry["center"] = np.array((result["center_x"], result["center_y"]))
                entry["axes"] = np.array((result["first_ax"], result["second_ax"]))
                entry["angle"] = result["angle"]

            data.append(entry)

        df = pd.DataFrame(data)
        df.to_pickle(f"eyeseg_benchmark.{method}.pkl")

