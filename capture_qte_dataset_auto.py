import os
import time
import mss
import numpy as np
import cv2
import keyboard

# -----------------------------
# CONFIG
# -----------------------------
SAVE_DIR = "dataset_qte_raw"
INTERVAL = 0.3        # temps entre 2 captures (en secondes)
MONITOR_ID = 1        # écran à capturer

# Si tu veux capturer seulement une zone :
USE_ZONE = False

# Zone personnalisée (si USE_ZONE = True)
ZONE = {
    "left": 0,
    "top": 0,
    "width": 1920,
    "height": 500
}

# -----------------------------
# INIT
# -----------------------------
os.makedirs(SAVE_DIR, exist_ok=True)

print("Capture automatique lancée.")
print("Appuie sur F10 pour arrêter.")
print("Les images sont sauvegardées dans :", SAVE_DIR)

# -----------------------------
# CAPTURE LOOP
# -----------------------------
with mss.mss() as sct:
    i = 0

    while True:
        if keyboard.is_pressed("f10"):
            print("Arrêt demandé. Fin de la capture.")
            break

        if USE_ZONE:
            img = sct.grab(ZONE)
        else:
            img = sct.grab(sct.monitors[MONITOR_ID])

        frame = np.array(img)[:, :, :3]  # BGR

        filename = f"frame_{i:06d}.png"
        path = os.path.join(SAVE_DIR, filename)

        cv2.imwrite(path, frame)
        print("Sauvegardé :", filename)

        i += 1
        time.sleep(INTERVAL)

print("Dataset terminé ✅")
