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
MODEL_PATH = "qte_resnet18_color2.pth"      # Nouveau modèle ResNet18
META_PATH  = "qte_meta_color2.json"         # Métadonnées
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_FP16 = True if torch.cuda.is_available() else False # Utiliser FP16 uniquement si GPU dispo

# -------------------------------------------------
# CONFIG ECRAN + ZONE A LIRE
# -------------------------------------------------
MONITOR_ID = 1      # ID du moniteur à capturer (1 = premier moniteur)
BAND_HEIGHT = 680   # Hauteur de la bande en haut de l'écran à capturer
ALLOWED_CHARS = set("ABCDEFGHIJKLMNPQRSTUVWYZ123456789")    # Caractères autorisés dans les QTE (exclut O et 0 pour éviter confusion et x car non utilisé)
MAX_BOXES_PER_FRAME = 3 # Nombre max de boîtes QTE à détecter par frame
DEBUG_WINDOWS = False   # Affiche les fenêtres de debug (capture + masque)
CONF_THRESHOLD = 0.85   # Seuil de confiance pour accepter une prédiction

IMG_SIZE = 64        # Taille d'entrée du modèle ResNet18

# -------------------------------------------------
# CHARGEMENT MODELE + META
# -------------------------------------------------
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

classes = meta["classes"]   # Liste des classes
num_classes = len(classes)  # Nombre de classes
idx_to_class = {i: c for i, c in enumerate(classes)}    # Mapping index -> classe pour permettre la prédiction car sinon on n'a que l'index en sortie du modèle

# Initialize ResNet18
model = models.resnet18()                               # Utilisation d'un modèle ResNet18 pré-entraîné sur ImageNet
num_ftrs = model.fc.in_features                         # Remplace la dernière couche fully connected pour correspondre au nombre de classes de mon dataset
model.fc = nn.Linear(num_ftrs, num_classes)             # Permet de faire des prédictions sur mes classes de QTE

# Charge les poids du modèle entraîné
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))  # permet de charger le modèle sur CPU ou GPU selon dispo
model.to(DEVICE)                                                    # envoie le modèle sur le device approprié
model.eval()                                                        # mode évaluation           

if USE_FP16:
    model.half() # accelère les inférences sur GPU compatible FP16

# Transform (Match training exactly)
inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),    # redimensionne à la taille attendue par le modèle
    transforms.ToTensor(),                      # convertit en tenseur PyTorch
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),    # normalisation comme lors de l'entraînement
])

print(f"[INFO] Modèle ResNet18 prêt sur {DEVICE} (FP16={USE_FP16})")

# -------------------------------------------------
# OPTIMIZED BATCH PREDICTION
# -------------------------------------------------

#Prédit toutes les boîtes détectées en un seul passage GPU (beaucoup plus rapide) car le problème que j'avais avant était le temps de prédiction par boîte individuelle
@torch.no_grad()
def predict_batch_resnet(boxes, frame_bgr):
    if not boxes:
        return []

    tensors = []    # Prépare les tenseurs pour chaque boîte détectée
    for (x, y, w, h) in boxes:
        roi = frame_bgr[y:y+h, x:x+w]   # extrait la ROI de l'image complète afin de minimiser les copies mémoire
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)  # convertit BGR -> RGB pour PIL
        img_pil = Image.fromarray(roi_rgb)              # convertit en image PIL
        tensors.append(inference_transform(img_pil))    # applique les transforms et ajoute à la liste

    # Stack into a single batch [Batch_Size, 3, 64, 64]
    batch_tensor = torch.stack(tensors).to(DEVICE)      # envoie le batch sur le device
    
    if USE_FP16:
        batch_tensor = batch_tensor.half()

    # Propagation avant
    outputs = model(batch_tensor)                       # sorties brutes du modèle [Batch_Size, Num_Classes]
    probs = torch.softmax(outputs, dim=1)               # convertit en probabilités
    confs, pred_indices = torch.max(probs, dim=1)       # Récupère les indices des classes prédites et leurs confiances affin de filtrer selon le seuil

    results = []    # Prépare les résultats finaux
    for i in range(len(boxes)):
        label = idx_to_class[int(pred_indices[i].item())]   # Récupère le label prédit
        conf = float(confs[i].item())                       # Récupère la confiance associée à cette prédiction
        if conf >= CONF_THRESHOLD:                          # n'accepte la prédiction que si au-dessus du seuil
            results.append((boxes[i][0], label)) # renvoie le X de la boîte et le label prédit
    
    return results

# -------------------------------------------------
# UTILS ET FONCTIONS PRINCIPALES
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