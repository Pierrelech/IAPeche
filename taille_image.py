import cv2
import os
from collections import Counter

sizes = Counter()

DATASET_DIR = "dataset_qte"

for root, _, files in os.walk(DATASET_DIR):
    for f in files:
        if f.endswith(".png") or f.endswith(".jpg"):
            img = cv2.imread(os.path.join(root, f))
            h, w, c = img.shape
            sizes[(w, h, c)] += 1

print("Tailles trouvées dans le dataset :")
for size, count in sizes.items():
    print(f"{size} -> {count} images")
