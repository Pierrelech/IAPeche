import mss
import numpy as np
import cv2

with mss.mss() as sct:
    for i, mon in enumerate(sct.monitors):
        print(f"Test écran {i}")
        img = sct.grab(mon)
        frame = np.array(img)
        cv2.imshow(f"Ecran {i}", frame)
        cv2.waitKey(5000)

cv2.destroyAllWindows()
