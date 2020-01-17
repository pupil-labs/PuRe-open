
import cv2

from pathlib import Path
import os
import csv

LPW_path = "../LPW"

def video_iterator(fix_subject=None, fix_video=None):
    for video_file in Path(LPW_path).rglob("*.avi"):
        video_id = video_file.stem
        subject = video_file.parent.name
        if fix_subject is not None and str(fix_subject) != subject:
            continue
        if fix_video is not None and str(fix_video) != video_id:
            continue
        info_file = video_file.with_suffix(".txt")

        with info_file.open() as f:
            targets = list(csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC))
        
        n = 0
        cap = cv2.VideoCapture(str(video_file))
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            if n > len(targets):
                print(f"Error: Just read frame#{n} for only {len(targets)} targets.")
                break
            
            yield subject, video_id, n, targets[n], frame 

            n += 1
        
        if n != len(targets):
            print(f"Error: Read {n} frames for {len(targets)} targets.")


def extract_images():
    for subject, video_id, n, target, frame in video_iterator():
        print(subject, video_id, n)
        img_file = f"{LPW_path}/images/{subject}/{video_id}/{n:04}.png"
        os.makedirs(os.path.dirname(img_file), exist_ok=True)
        cv2.imwrite(img_file, frame)

def get_frame(subject, video_id, frame_n):
    info_file = Path(f"{LPW_path}/{subject}/{video_id}.txt")
    with info_file.open() as f:
        target = list(csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC))[frame_n]
    img_file = f"{LPW_path}/images/{subject}/{video_id}/{frame_n:04}.png"
    return cv2.imread(img_file)


