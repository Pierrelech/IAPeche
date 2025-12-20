import os
import time

import mss
import numpy as np
import cv2
import keyboard

# -----------------------------
# CONFIG ECRAN + ZONE A LIRE
# -----------------------------
MONITOR_ID = 1
BAND_HEIGHT = 260
MAX_BOXES_PER_FRAME = 3

# -----------------------------
# CONFIG HSV (bouton QTE brun/olive)
# -----------------------------
# Valeurs basées sur ton screenshot (bouton brun/olive avec lettre blanche).
# Ajustables si besoin.
HSV_LOWER = np.array([18, 40, 80], dtype=np.uint8)   # H, S, V min
HSV_UPPER = np.array([35, 170, 200], dtype=np.uint8) # H, S, V max

# Dossier racine du dataset
DATASET_ROOT = "dataset_qte"
QUEUE_DIR = os.path.join(DATASET_ROOT, "_queue")


def ensure_dataset_dirs():
    os.makedirs(DATASET_ROOT, exist_ok=True)
    os.makedirs(QUEUE_DIR, exist_ok=True)
    print(f"[INFO] Dossier dataset       : {DATASET_ROOT}")
    print(f"[INFO] Dossier file d'attente: {QUEUE_DIR}")


def get_zone_for_top_band(sct):
    mon = sct.monitors[MONITOR_ID]
    return {
        "left": mon["left"],
        "top": mon["top"],
        "width": mon["width"],
        "height": BAND_HEIGHT
    }


def detect_qte_boxes_hsv(frame_bgr, frame_vis):
    """
    Détection des boutons QTE via HSV (brun/olive),
    extraction des ROI EN COULEUR pour dataset.
    """
    # 1) HSV + masque couleur
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

    # 2) Nettoyage morphologique
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidate_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        if area < 900 or area > 8000:
            continue

        ratio = w / float(h)
        if ratio < 0.7 or ratio > 1.4:
            continue

        if y < 5:
            continue

        candidate_boxes.append((x, y, w, h))

    candidate_boxes.sort(key=lambda b: b[1], reverse=True)
    boxes_to_use = candidate_boxes[:MAX_BOXES_PER_FRAME]

    rois = []
    boxes = []

    # 3) EXTRACTION ROI COULEUR
    for (x, y, w, h) in boxes_to_use:
        roi_color = frame_bgr[y:y + h, x:x + w]

        # Resize léger (optionnel mais conseillé)
        roi_resized = cv2.resize(
            roi_color, (64, 64), interpolation=cv2.INTER_LINEAR
        )

        rois.append(roi_resized)
        boxes.append((x, y, w, h))

        cv2.rectangle(frame_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return rois, boxes, mask



def enqueue_roi(roi):
    timestamp = int(time.time() * 1000)
    filename = os.path.join(QUEUE_DIR, f"pending_{timestamp}.png")
    cv2.imwrite(filename, roi)
    print(f"[QUEUE] Image ajoutée : {filename}")


def main():
    ensure_dataset_dirs()

    print("=== COLLECTEUR DE DATASET QTE (MODE FILE D'ATTENTE) — HSV brun/olive ===")
    print(" - Capture la bande du haut de l'écran 1.")
    print(" - 1 capture par 0.5 secondes.")
    print(" - Détection des boutons via couleur HSV.")
    print(" - ESC dans la fenêtre debug ou F10 au clavier pour quitter.")
    print()

    time.sleep(2)
    last_capture_time = 0.0

    with mss.mss() as sct:
        while True:
            if keyboard.is_pressed("f10"):
                print("[INFO] F10 -> sortie.")
                break

            now = time.time()
            if now - last_capture_time >= 0.5:
                last_capture_time = now

                zone = get_zone_for_top_band(sct)
                img = sct.grab(zone)
                frame = np.array(img)  # BGRA

                # BGRA -> BGR
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                frame_vis = frame_bgr.copy()

                rois, boxes, mask_debug = detect_qte_boxes_hsv(frame_bgr, frame_vis)

                if rois:
                    print(f"[INFO] {len(rois)} bouton(s) détecté(s) sur cette frame.")
                    for roi in rois:
                        enqueue_roi(roi)

                cv2.imshow("ZONE QTE (debug)", frame_vis)
                cv2.imshow("MASK HSV (debug)", mask_debug)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                print("[INFO] ESC -> sortie.")
                break

            time.sleep(0.01)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
