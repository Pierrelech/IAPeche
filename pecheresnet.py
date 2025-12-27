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
from torchvision import transforms, models
from PIL import Image

# -------------------------------------------------
# CONFIG MODELE
# -------------------------------------------------
MODEL_PATH = "qte_resnet18_color.pth"
META_PATH  = "qte_meta_color.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_FP16 = True if torch.cuda.is_available() else False # Fast mode for NVIDIA GPUs

# -------------------------------------------------
# CONFIG ECRAN + ZONE A LIRE
# -------------------------------------------------
MONITOR_ID = 1
BAND_HEIGHT = 680
ALLOWED_CHARS = set("ABCDEFGHIJKLMNPQRSTUVWYZ123456789")
MAX_BOXES_PER_FRAME = 3
DEBUG_WINDOWS = False
CONF_THRESHOLD = 0.85

IMG_SIZE = 64

# -------------------------------------------------
# CHARGEMENT MODELE + META
# -------------------------------------------------
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

classes = meta["classes"]
num_classes = len(classes)
idx_to_class = {i: c for i, c in enumerate(classes)}

# Initialize ResNet18
model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Load weights and optimize
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

if USE_FP16:
    model.half() # Instant speed boost on GPU

# Transform (Match training exactly)
inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print(f"[INFO] Modèle ResNet18 prêt sur {DEVICE} (FP16={USE_FP16})")

# -------------------------------------------------
# OPTIMIZED BATCH PREDICTION
# -------------------------------------------------
@torch.no_grad()
def predict_batch_resnet(boxes, frame_bgr):
    """Predicts all detected boxes in a single GPU pass (much faster)"""
    if not boxes:
        return []

    tensors = []
    for (x, y, w, h) in boxes:
        roi = frame_bgr[y:y+h, x:x+w]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(roi_rgb)
        tensors.append(inference_transform(img_pil))

    # Stack into a single batch [Batch_Size, 3, 64, 64]
    batch_tensor = torch.stack(tensors).to(DEVICE)
    
    if USE_FP16:
        batch_tensor = batch_tensor.half()

    # One single model call for all boxes
    outputs = model(batch_tensor)
    probs = torch.softmax(outputs, dim=1)
    confs, pred_indices = torch.max(probs, dim=1)

    results = []
    for i in range(len(boxes)):
        label = idx_to_class[int(pred_indices[i].item())]
        conf = float(confs[i].item())
        if conf >= CONF_THRESHOLD:
            results.append((boxes[i][0], label)) # (x_coord, label)
    
    return results

# -------------------------------------------------
# UTILS
# -------------------------------------------------
def get_zone_for_top_band(sct):
    mon = sct.monitors[MONITOR_ID]
    return {"left": mon["left"], "top": mon["top"], "width": mon["width"], "height": BAND_HEIGHT}

HSV_LOWER = np.array([18, 40, 80], dtype=np.uint8)
HSV_UPPER = np.array([35, 170, 200], dtype=np.uint8)

def detect_qte_boxes_hsv(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    # Fast cleaning
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 900 < (w * h) < 8000:
            if 0.7 < (w / h) < 1.4:
                boxes.append((x, y, w, h))
    
    return boxes[:MAX_BOXES_PER_FRAME], mask

def send_keys(text):
    for c in text:
        if c.isdigit():
            pyautogui.press(f"num{c}")
        else:
            keyboard.send(c.lower())
        time.sleep(0.01) # Small delay to avoid game ghosting

# -------------------------------------------------
# MAIN LOOP
# -------------------------------------------------
def main():
    print("Bot QTE Turbo (ResNet18 Batch Mode) lancé. F10 pour arrêter.")
    history = deque(maxlen=4)
    last_sent_text = ""

    with mss.mss() as sct:
        zone = get_zone_for_top_band(sct)
        
        while True:
            if keyboard.is_pressed("f10"): break

            # Optimized capture
            sct_img = sct.grab(zone)
            frame_bgr = np.frombuffer(sct_img.bgra, dtype=np.uint8).reshape(sct_img.height, sct_img.width, 4)
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2BGR)

            boxes, mask = detect_qte_boxes_hsv(frame_bgr)
            
            if boxes:
                # Fast Batch Prediction
                predictions = predict_batch_resnet(boxes, frame_bgr)
                
                # Sort by X coordinate to read left-to-right
                predictions.sort(key=lambda x: x[0])
                current_text = "".join(p[1] for p in predictions)
                
                history.append(current_text)
                
                # Check for stability across 3 frames
                if len(history) >= 3:
                    counts = Counter(history)
                    stable_text, count = counts.most_common(1)[0]
                    
                    if count >= 3 and stable_text != last_sent_text and stable_text != "":
                        print(f"[ACTION] Envoi : {stable_text}")
                        send_keys(stable_text)
                        last_sent_text = stable_text
            else:
                # Clear history if no boxes found (QTE ended)
                if last_sent_text != "":
                    history.clear()
                    last_sent_text = ""

            if DEBUG_WINDOWS:
                cv2.imshow("Debug", frame_bgr)
                if cv2.waitKey(1) == 27: break
            
            # Minor sleep to free up CPU
            time.sleep(0.001)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()