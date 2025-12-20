import os
import time
import json
from collections import deque, Counter

import pyautogui
import mss
import numpy as np
import cv2
import keyboard
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# -------------------------------------------------
# CONFIG MODELE (NOUVEAU MODELE COULEUR)
# -------------------------------------------------
MODEL_PATH = "qte_mlp_color.pth"        # <-- adapte si besoin
META_PATH  = "qte_meta_color.json"     # <-- adapte si besoin
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# CONFIG ECRAN + ZONE A LIRE
# -------------------------------------------------
MONITOR_ID = 1
BAND_HEIGHT = 680

ALLOWED_CHARS = set("ABCDEFGHIJKLMNPQRSTUVWYZ123456789")
MAX_BOXES_PER_FRAME = 3
DEBUG_WINDOWS = False
CONF_THRESHOLD = 0.6

IMG_SIZE = 64  # nouveau modèle

# -------------------------------------------------
# MODELE MLP COULEUR (même que ton training)
# -------------------------------------------------
class QTEMLPColor(nn.Module):
    def __init__(self, num_classes, img_size=64):
        super().__init__()
        self.flatten = nn.Flatten()
        in_features = 3 * img_size * img_size

        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)

# -------------------------------------------------
# CHARGEMENT MODELE + META
# -------------------------------------------------
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

classes = meta["classes"]
num_classes = len(classes)
idx_to_class = {i: c for i, c in enumerate(classes)}  # FIX (mapping correct)

print("[INFO] Classes du modèle :", classes)

inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),  # RGB -> (3,H,W) en [0,1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

model = QTEMLPColor(num_classes=num_classes, img_size=IMG_SIZE).to(DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

print("[INFO] Modèle chargé sur :", DEVICE)

# -------------------------------------------------
# OUTILS CAPTURE + DETECTION
# -------------------------------------------------
def get_zone_for_top_band(sct):
    mon = sct.monitors[MONITOR_ID]
    return {
        "left": mon["left"],
        "top": mon["top"],
        "width": mon["width"],
        "height": BAND_HEIGHT,
    }

HSV_LOWER = np.array([18, 40, 80], dtype=np.uint8)
HSV_UPPER = np.array([35, 170, 200], dtype=np.uint8)

def detect_qte_boxes_hsv(frame_bgr, frame_vis_bgr):
    """
    Détecte les boutons QTE via HSV (brun/olive) et renvoie:
      - boxes_to_use: liste (x,y,w,h)
      - mask_debug: masque HSV nettoyé (pour debug)
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

    # Nettoyage morphologique (comme collecte)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidate_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        # Filtres (à ajuster si besoin)
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

    # debug rectangles
    for (x, y, w, h) in boxes_to_use:
        cv2.rectangle(frame_vis_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return boxes_to_use, mask

# -------------------------------------------------
# PREDICTION DIRECTEMENT SUR ROI COULEUR (SANS FICHIER)
# -------------------------------------------------
@torch.no_grad()
def predict_from_roi_bgr(roi_bgr):
    """
    roi_bgr: numpy array (H,W,3) en BGR (OpenCV)
    Retourne (label, conf) ou (None, conf)
    """
    # BGR -> RGB
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

    # numpy -> PIL
    img = Image.fromarray(roi_rgb)

    x = inference_transform(img).unsqueeze(0).to(DEVICE)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    conf, pred_idx = torch.max(probs, dim=1)

    label = idx_to_class[int(pred_idx.item())]
    conf = float(conf.item())

    if label in ALLOWED_CHARS and conf >= CONF_THRESHOLD:
        return label, conf
    else:
        return None, conf

def send_keys_from_text(text):
    for c in text:
        if c.isdigit():
            # si ton jeu attend les chiffres "normaux" et pas le pavé num, remplace par pyautogui.press(c)
            key_numpad = f"num{c}"
            print(f"Sending digit (numpad): {key_numpad}")
            pyautogui.press(key_numpad)
        else:
            key = c.lower()
            print(f"Sending letter: {key}")
            keyboard.send(key)
        time.sleep(0.03)

# -------------------------------------------------
# STABILISATION
# -------------------------------------------------
def compute_stable_text(history, min_votes=3):
    if not history:
        return ""
    max_len = max(len(t) for t in history)
    stable_chars = []
    for i in range(max_len):
        chars_at_pos = [t[i] for t in history if len(t) > i]
        if not chars_at_pos:
            continue
        counts = Counter(chars_at_pos)
        char, count = counts.most_common(1)[0]
        if count >= min_votes:
            stable_chars.append(char)
    return "".join(stable_chars)

# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    print("Bot QTE (ROI couleur + modèle couleur) lancé.")
    print("F10 pour arrêter.\n")

    history = deque(maxlen=5)
    last_stable_text = ""

    time.sleep(2)

    with mss.mss() as sct:
        while True:
            if keyboard.is_pressed("f10"):
                print("Arrêt demandé. Bye.")
                break

            zone = get_zone_for_top_band(sct)
            img = sct.grab(zone)

            frame_bgra = np.array(img)
            frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

            # pour contours seulement


            # détection des boîtes
            boxes, mask_debug = detect_qte_boxes_hsv(frame_bgr, frame_bgr)


            detected_chars = []

            for (x, y, w, h) in boxes:
                # ✅ ROI COULEUR (pas de binaire)
                roi_bgr = frame_bgr[y:y + h, x:x + w]

                label, conf = predict_from_roi_bgr(roi_bgr)

                if label is not None:
                    detected_chars.append((x, label))
                    cv2.putText(
                        frame_bgr,
                        f"{label} ({conf:.2f})",
                        (x, max(0, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

            detected_chars.sort(key=lambda t: t[0])
            text = "".join(c for _, c in detected_chars)

            history.append(text)
            stable_text = compute_stable_text(history, min_votes=3)

            if stable_text and stable_text != last_stable_text:
                print(f"Brut : {text} | Stable : {stable_text}")
                send_keys_from_text(stable_text)
                last_stable_text = stable_text

            if DEBUG_WINDOWS:
                cv2.imshow("Bande brute + boxes", frame_bgr)
                cv2.imshow("MASK HSV (debug)", mask_debug)

                if cv2.waitKey(1) == 27:
                    break
            else:
                cv2.waitKey(1)

            time.sleep(0.05)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
