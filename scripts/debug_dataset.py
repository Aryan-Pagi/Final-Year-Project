import os
import cv2
from collections import defaultdict
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mediapipe_utils import HandDetector

DATASET = "dataset/raw_images"
FAILED_DIR = "dataset/failed_samples"

os.makedirs(FAILED_DIR, exist_ok=True)

detector = HandDetector(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.1,
    min_tracking_confidence=0.1,
)

total = 0
success = 0
failed = 0

class_stats = defaultdict(lambda: {"ok": 0, "fail": 0})

print("=" * 60)
print("CHECKING DATASET")
print("=" * 60)

for label in sorted(os.listdir(DATASET)):

    label_path = os.path.join(DATASET, label)

    if not os.path.isdir(label_path):
        continue

    for root, dirs, files in os.walk(label_path):

        for file in files:

            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            total += 1

            path = os.path.join(root, file)

            image = cv2.imread(path)

            if image is None:
                failed += 1
                class_stats[label]["fail"] += 1
                print(f"Cannot read: {path}")
                continue

            _, results = detector.find_hands(image, draw=False)

            if results.multi_hand_landmarks:

                success += 1
                class_stats[label]["ok"] += 1

            else:

                failed += 1
                class_stats[label]["fail"] += 1

                save_folder = os.path.join(FAILED_DIR, label)
                os.makedirs(save_folder, exist_ok=True)

                cv2.imwrite(
                    os.path.join(save_folder, file),
                    image,
                )

detector.close()

print("\n")
print("=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"Total Images : {total}")
print(f"Detected     : {success}")
print(f"Failed       : {failed}")

if total:
    print(f"Success Rate : {100*success/total:.2f}%")

print("\nPer Class Statistics")
print("-" * 60)

for cls in sorted(class_stats):

    ok = class_stats[cls]["ok"]
    fail = class_stats[cls]["fail"]

    total_cls = ok + fail

    rate = 100 * ok / total_cls if total_cls else 0

    print(f"{cls:15}  {ok:4}/{total_cls:<4}  ({rate:6.2f}%)")

print("\nFailed images have been copied to:")
print(FAILED_DIR)