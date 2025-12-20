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
# CONFIG MODELE
# -------------------------------------------------

MODEL_PATH = "qte_cnn.pth"
META_PATH = "qte_meta.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dossier temporaire pour les images de QTE
TEMP_DIR = "tmp_qte"

# -------------------------------------------------
# CONFIG ECRAN + ZONE A LIRE
# -------------------------------------------------

MONITOR_ID = 1
BAND_HEIGHT = 260

# touches autorisées
ALLOWED_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

# Limite de cases par frame
MAX_BOXES_PER_FRAME = 3

# Debug visuel
DEBUG_WINDOWS = False

# Seuil de confiance
CONF_THRESHOLD = 0.6

# -------------------------------------------------
# MODELE MLP (même que train_qte_cnn.py)
# -------------------------------------------------

class QTEMLP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.flatten = nn.Flatten()
        in_features = 1 * 32 * 32

        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.flatten(x)  # (B,1,32,32) -> (B,1024)
        x = self.net(x)
        return x


# -------------------------------------------------
# CHARGEMENT MODELE + META
# -------------------------------------------------

with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

classes = meta["classes"]
class_to_idx = meta["class_to_idx"]
idx_to_class = {v: k for v, k in enumerate(classes)}

num_classes = len(classes)

print("[INFO] Classes du modèle :", classes)

inference_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

model = QTEMLP(num_classes=num_classes).to(DEVICE)

state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

print("[INFO] Modèle chargé sur :", DEVICE)

os.makedirs(TEMP_DIR, exist_ok=True)

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


def detect_qte_boxes(gray, frame_vis):
    """
    Retourne (rois, boxes, thresh)
    rois = images binaires des cases
    boxes = (x,y,w,h)
    """

    _, thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidate_boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        if area < 1000 or area > 8000:
            continue

        ratio = w / float(h)
        if ratio < 0.75 or ratio > 1.3:
            continue

        candidate_boxes.append((x, y, w, h))

    candidate_boxes.sort(key=lambda b: b[1], reverse=True)
    boxes_to_use = candidate_boxes[:MAX_BOXES_PER_FRAME]

    rois = []

    for (x, y, w, h) in boxes_to_use:
        roi = thresh[y:y + h, x:x + w]
        roi = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        rois.append(roi)

        cv2.rectangle(frame_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return rois, boxes_to_use, thresh


# -------------------------------------------------
# SAUVEGARDE TEMP + PREDICTION SUR FICHIER
# -------------------------------------------------

def save_temp_roi(roi):
    """Sauvegarde le ROI dans TEMP_DIR et retourne le chemin du fichier."""
    ts = int(time.time() * 1000)
    filename = f"qte_{ts}.png"
    path = os.path.join(TEMP_DIR, filename)
    cv2.imwrite(path, roi)
    return path


def predict_from_file(path):
    """
    Charge une image disque, applique le modèle
    et retourne (label, conf) ou (None, conf).
    """
    img = Image.open(path).convert("L")
    x = inference_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

        label = classes[pred_idx.item()]
        conf = float(conf.item())

        if label in ALLOWED_CHARS and conf >= CONF_THRESHOLD:
            return label, conf
        else:
            return None, conf


def send_keys_from_text(text):
    for c in text:
        if c.isdigit():
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
    print("Bot QTE (fichiers + modèle) lancé.")
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

            frame = np.array(img)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

            rois, boxes, thresh = detect_qte_boxes(gray, frame)

            detected_chars = []

            for roi, (x, y, w, h) in zip(rois, boxes):

                path = save_temp_roi(roi)
                label, conf = predict_from_file(path)

                try:
                    os.remove(path)
                except OSError:
                    pass

                if label is not None:
                    detected_chars.append((x, label))

                    cv2.putText(
                        frame,
                        f"{label} ({conf:.2f})",
                        (x, y - 5),
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
                cv2.imshow("Bande brute + contours", frame)
                cv2.imshow("Bande traitée", thresh)

                if cv2.waitKey(1) == 27:
                    break
            else:
                cv2.waitKey(1)

            time.sleep(0.01)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
