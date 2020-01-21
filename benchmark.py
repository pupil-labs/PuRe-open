from pupil_detectors import Detector2D
from pure_detector import PuReDetector
from pure_original import PuReOriginal


import cv2
import pandas as pd
import numpy as np
import time
import platform

import LPW
import eye_seg

def LPW_iterator():
    for subject, video_id, n, target, frame in LPW.video_iterator():
        yield frame, target, f"{subject}.{video_id}.{n}"

def eye_seg_iterator():
    for part, n, raw_frame, center, axes, angle in eye_seg.video_iterator():
        yield raw_frame, center, f"{part}.{n}"

datasets = {
    "lpw": LPW_iterator,
    "eyeseg500k": eye_seg_iterator,
}

detectors = {
    "2d": Detector2D(),
    "pure_pfa": PuReDetector(),
    "pure_orig": PuReOriginal(),
}

sizes = [
    "original",
    "192x192",
    "320x240",
]

skip_patterns = [
    "lpw",
]

computer = platform.node()

for dataset, iterator in datasets.items():
    for name, detector in detectors.items():
        for size in sizes:
            method = ".".join([
                "benchmark",
                dataset,
                size,
                computer,
                name,
            ])

            if any(pattern in method for pattern in skip_patterns):
                print(f"Skipping '{method}'")
                continue

            data = []

            for i, (frame, target, data_key) in enumerate(iterator()):
                if i % 100 == 0:
                    print(method, i, data_key)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                area = gray.shape[0] * gray.shape[1]
                if size == "original":
                    factor = 1.0
                elif size == "192x192":
                    target_area = 192 * 192
                    factor = np.sqrt(target_area / area)
                elif size == "320x240":
                    target_area = 320 * 240
                    factor = np.sqrt(target_area / area)
                    
                if factor > 1.0:
                    gray = cv2.resize(gray, dsize=None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
                elif factor < 1.0:
                    gray = cv2.resize(gray, dsize=None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)

                rescaling_factor = 1.0 / factor


                t1 = time.perf_counter()
                result = detector.detect(gray)
                t2 = time.perf_counter()

                entry = {
                    "data_key": data_key,
                    "time": t2 - t1,
                    "method": method,
                    "target": np.array(target),
                    "confidence": result["confidence"],
                }

                if name != "pure_orig":
                    entry["center"] = np.array(result["ellipse"]["center"]) * rescaling_factor
                    entry["axes"] = np.array(result["ellipse"]["axes"]) * rescaling_factor
                    entry["angle"] = result["ellipse"]["angle"]
                else:
                    entry["center"] = np.array((result["center_x"], result["center_y"])) * rescaling_factor
                    entry["axes"] = np.array((result["first_ax"], result["second_ax"])) * rescaling_factor
                    entry["angle"] = result["angle"]

                data.append(entry)

            df = pd.DataFrame(data)
            df.to_pickle(f"{method}.pkl")

