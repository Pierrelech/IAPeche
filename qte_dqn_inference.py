import os
import time
import json
from collections import deque, Counter, deque as Deque

import pyautogui
import mss
import numpy as np
import cv2
import keyboard
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pytesseract  # OCR pour la zone de résultat


# -------------------------------------------------
# CONFIG TESSERACT
# -------------------------------------------------
# Si Tesseract n'est pas dans le PATH, décommente et adapte :
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# -------------------------------------------------
# CONFIG MODELE QTE / RL
# -------------------------------------------------
MODEL_PATH = "qte_dqn.pth"
META_PATH = "qte_meta.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEMP_DIR = "tmp_qte"

# -------- RL HYPERPARAMS --------
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 64

REPLAY_CAPACITY = 20000
START_LEARNING_AFTER = 500       # nb de transitions avant de commencer les updates
TARGET_UPDATE_EVERY = 500        # steps avant copie du réseau cible

EPS_START = 0.5                  # prob d’explo au début
EPS_END = 0.05
EPS_DECAY = 20000                # + grand => décroissance plus lente

QTE_ACTION_COOLDOWN = 0.25       # s entre deux appuis de touche (pour éviter le spam)


# -------------------------------------------------
# CONFIG ECRAN + ZONES A LIRE
# -------------------------------------------------
MONITOR_ID = 1

# Bande en haut pour les QTE
BAND_HEIGHT = 260

# ----- Zone résultat (message bas-gauche) -----
RESULT_ZONE_LEFT_FRAC = 0.0
RESULT_ZONE_TOP_FRAC = 0.6
RESULT_ZONE_WIDTH_FRAC = 0.20
RESULT_ZONE_HEIGHT_FRAC = 0.3

# touches autorisées (doivent correspondre aux classes de ton dataset)
ALLOWED_CHARS = set("ABCDEFGHIJKLMNPQRSTUVWYZ123456789")

# Limite de cases QTE par frame
MAX_BOXES_PER_FRAME = 3

# Debug visuel
DEBUG_WINDOWS = True

# Mots-clés pour OCR de résultat
SUCCESS_KEYWORDS = ["succès", "succes", "réussi", "reussi", "attrapé", "attrape", "fis"]
FAIL_KEYWORDS = ["échec", "echec", "raté", "rate", "failed", "fail"]


# -------------------------------------------------
# MODELE MLP QTE (DEVENU RÉSEAU Q)
# -------------------------------------------------
class QTEMLP(nn.Module):
    """
    Même archi que ton modèle de classification, mais on l'interprète
    comme un réseau Q : chaque sortie = Q(s, action_i).
    """
    def __init__(self, num_actions):
        super().__init__()
        self.flatten = nn.Flatten()
        in_features = 1 * 32 * 32

        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.net(x)
        return x


# -------------------------------------------------
# CHARGEMENT META + MODELES RL
# -------------------------------------------------
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

classes = meta["classes"]            # ex: ['1','2',...,'Z']
class_to_idx = meta["class_to_idx"]
num_actions = len(classes)

print("[INFO] Classes / actions :", classes)
print("[INFO] nb_actions =", num_actions)

inference_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# Réseau Q principal + cible
q_net = QTEMLP(num_actions=num_actions).to(DEVICE)
target_net = QTEMLP(num_actions=num_actions).to(DEVICE)

# On charge les poids initiaux depuis ton modèle supervisé si dispo
if os.path.exists(MODEL_PATH):
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    q_net.load_state_dict(state_dict, strict=False)
    target_net.load_state_dict(q_net.state_dict())
    print("[INFO] Poids initiaux chargés depuis :", MODEL_PATH)
else:
    target_net.load_state_dict(q_net.state_dict())
    print("[WARN] Aucun modèle initial trouvé, RL part de zéro.")

target_net.eval()

optimizer = optim.Adam(q_net.parameters(), lr=LR)

os.makedirs(TEMP_DIR, exist_ok=True)


# -------------------------------------------------
# REPLAY BUFFER
# -------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = Deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        # s, s2 : Tensors CPU (1,1,32,32)
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        s   = torch.cat(s, dim=0)   # (B,1,32,32)
        s2  = torch.cat(s2, dim=0)
        a   = torch.tensor(a, dtype=torch.long)
        r   = torch.tensor(r, dtype=torch.float32)
        d   = torch.tensor(d, dtype=torch.float32)
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buffer)


replay = ReplayBuffer(REPLAY_CAPACITY)


# -------------------------------------------------
# RL UTILS
# -------------------------------------------------
import random


def epsilon_by_step(step_idx: int) -> float:
    # décroissance exponentielle
    eps = EPS_END + (EPS_START - EPS_END) * \
        torch.exp(torch.tensor(-step_idx / EPS_DECAY))
    return float(eps.item())


def select_action(state_tensor: torch.Tensor, step_idx: int) -> (int, float):
    """
    state_tensor : (1,1,32,32) sur DEVICE
    Retourne (action_index, epsilon_utilisé)
    """
    eps = epsilon_by_step(step_idx)
    if random.random() < eps:
        # exploration
        action = random.randrange(num_actions)
    else:
        with torch.no_grad():
            q_values = q_net(state_tensor)  # (1, num_actions)
            action = int(torch.argmax(q_values, dim=1).item())
    return action, eps


def compute_dqn_loss(batch):
    states, actions, rewards, next_states, dones = batch

    states      = states.to(DEVICE)
    next_states = next_states.to(DEVICE)
    actions     = actions.to(DEVICE)
    rewards     = rewards.to(DEVICE)
    dones       = dones.to(DEVICE)

    # Q(s,a)
    q_values = q_net(states)                      # (B, num_actions)
    q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # max_a' Q_target(s', a')
    with torch.no_grad():
        next_q_values = target_net(next_states)
        next_q_max    = next_q_values.max(1)[0]
        target = rewards + (1.0 - dones) * GAMMA * next_q_max

    loss = F.mse_loss(q_sa, target)
    return loss


def roi_to_state(roi: np.ndarray) -> torch.Tensor:
    """
    roi : image numpy 2D (binaire ou niveaux de gris)
    Retourne un tensor (1,1,32,32) sur DEVICE.
    """
    pil = Image.fromarray(roi)
    x = inference_transform(pil).unsqueeze(0)
    return x.to(DEVICE)


def send_key_for_action(action_idx: int):
    """
    Convertit l’index d’action en touche réelle (lettre / chiffre) et l’envoie.
    """
    char = classes[action_idx]

    if char.isdigit():
        key_numpad = f"num{char}"
        print(f"[KEY] digit -> {key_numpad}")
        pyautogui.press(key_numpad)
    else:
        key = char.lower()
        print(f"[KEY] letter -> {key}")
        keyboard.send(key)

    time.sleep(0.03)


# -------------------------------------------------
# OUTILS CAPTURE QTE (comme ta version)
# -------------------------------------------------
def get_zone_for_top_band(sct):
    mon = sct.monitors[MONITOR_ID]
    return {
        "left": mon["left"],
        "top": mon["top"],
        "width": mon["width"],
        "height": BAND_HEIGHT
    }


def detect_qte_boxes(gray, frame_vis):
    """
    Retourne (rois, boxes, thresh) pour les QTE en haut.
    rois = images binaires des cases
    boxes = (x,y,w,h)
    """
    
    # --- MODIFICATION: Seuillage pour Texte Sombre sur Fond Clair ---
    # Nous voulons que l'objet (texte noir) soit blanc pour la détection des contours.
    # Un seuil fixe ou adaptatif peut être plus précis que OTSU ici.
    # Pour l'exemple, restons sur OTSU mais retirons l'inversion qui efface le caractère.
    
    # Isoler la forme extérieure du QTE (blanc) sur fond (sombre)
    _, thresh_qte_shape = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Le texte (lettre) est probablement perdu ici car il est sombre.
    # Nous allons conserver l'inversion pour la détection de CONTOURS (car on veut détecter les BORDURES CLAIRES des QTEs)
    thresh = cv2.bitwise_not(thresh_qte_shape) 
    
    # Dilatation/Ouverture pour l'isolation des formes
    kernel_open = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=2) 
    kernel_dilate = np.ones((2, 2), np.uint8)
    thresh = cv2.dilate(thresh, kernel_dilate, iterations=1)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # ... (le reste du filtrage des contours est conservé) ...
    candidate_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        
        # Filtres de taille et ratio conservés
        if area < 1200 or area > 6000:
            continue
        ratio = w / float(h)
        if ratio < 0.7 or ratio > 1.4:
            continue
        if y < 10: 
            continue

        candidate_boxes.append((x, y, w, h))

    candidate_boxes.sort(key=lambda b: b[1], reverse=True)
    boxes_to_use = candidate_boxes[:MAX_BOXES_PER_FRAME]

    rois = []
    for (x, y, w, h) in boxes_to_use:
        # --- MODIFICATION MAJEURE ICI ---
        # Au lieu d'utiliser le 'thresh' des contours (qui n'a pas la lettre), 
        # on doit re-seuiller la zone d'intérêt (ROI) pour capturer le caractère noir.
        roi_original_gray = gray[y:y + h, x:x + w]
        
        # Seuil inversé pour la classification (met le caractère noir en blanc)
        # 1. Utiliser un seuil local pour le caractère.
        # 2. Utiliser un seuil simple et inverser.
        # Option 2 : Seuil à 120 (ou autre valeur estimée pour séparer le noir du blanc du QTE)
        _, roi_final = cv2.threshold(roi_original_gray, 120, 255, cv2.THRESH_BINARY_INV) 

        rois.append(roi_final) 
        
        cv2.rectangle(frame_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    # On renvoie le thresh_qte_shape pour le debug visuel
    return rois, boxes_to_use, thresh_qte_shape 


# -------------------------------------------------------------------
# A NOTER :
# Dans la boucle principale, vous affichiez cv2.imshow("QTE - thresh", thresh_qte),
# j'ai changé le nom de la variable de retour pour correspondre à cette ligne.
# -------------------------------------------------------------------


# -------------------------------------------------------------------
# A NOTER :
# Dans la boucle principale, vous affichiez cv2.imshow("QTE - thresh", thresh_qte),
# j'ai changé le nom de la variable de retour pour correspondre à cette ligne.
# -------------------------------------------------------------------


# -------------------------------------------------
# ZONE RESULTAT (SUCCES / ECHEC) - même logique qu’avant
# -------------------------------------------------
def get_result_zone(sct):
    mon = sct.monitors[MONITOR_ID]
    left = mon["left"] + int(mon["width"] * RESULT_ZONE_LEFT_FRAC)
    top = mon["top"] + int(mon["height"] * RESULT_ZONE_TOP_FRAC)
    width = int(mon["width"] * RESULT_ZONE_WIDTH_FRAC)
    height = int(mon["height"] * RESULT_ZONE_HEIGHT_FRAC)

    return {
        "left": left,
        "top": top,
        "width": width,
        "height": height,
    }


def read_result_zone(sct):
    """
    Lit la zone de résultat en bas à gauche et essaie de détecter
    un message de succès / échec avec un prétraitement plus fin.
    Renvoie :
      - "success", frame_dbg, thresh_dbg
      - "fail", frame_dbg, thresh_dbg
      - None, frame_dbg, thresh_dbg
    """
    zone = get_result_zone(sct)
    img = sct.grab(zone)
    frame = np.array(img)

    # 1) gris + léger flou
    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 2) Augmenter le contraste local (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 3) Agrandir l'image pour aider Tesseract
    up = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

    # 4) Seuil adaptatif
    thresh = cv2.adaptiveThreshold(
        up,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5
    )

    # 5) OCR
    config = r"--oem 3 --psm 6"
    text = pytesseract.image_to_string(thresh, lang="fra+eng", config=config)
    text_low = text.lower().strip()

    # 6) Recherche de mots-clés
    if any(k in text_low for k in SUCCESS_KEYWORDS):
        return "success", frame, thresh

    if any(k in text_low for k in FAIL_KEYWORDS):
        return "fail", frame, thresh

    return None, frame, thresh


# -------------------------------------------------
# MAIN : RL ONLINE
# -------------------------------------------------
# -------------------------------------------------
# MAIN : RL ONLINE (CORRIGÉ POUR COOLDOWN DE 7 SECONDES)
# -------------------------------------------------
# -------------------------------------------------
# MAIN : RL ONLINE (GESTION DE SÉQUENCE ET RÉCOMPENSE DIFFÉRÉE)
# -------------------------------------------------
def main():
    print("Bot QTE RL online (DQN + lecture résultat) lancé.")
    print("F10 pour arrêter.\n")

    total_success = 0
    total_fail = 0
    last_result = None
    last_result_time = 0.0
    RESULT_COOLDOWN = 7.0 # secondes (ATTENTE MINIMALE entre deux événements de récompense)

    last_action_time = 0.0 # anti-spam des touches
    global_step = 0  # compteur pour epsilon / target update

    # --- NOUVEAUX ÉLÉMENTS POUR LA GESTION DE SÉQUENCE ---
    # Buffer pour stocker les transitions (s, a) jusqu'à la réception de la récompense finale.
    # Chaque élément est un tuple: (state_tensor, action_idx)
    sequence_buffer = deque(maxlen=5) 
    # ----------------------------------------------------

    time.sleep(2)

    with mss.mss() as sct:
        while True:
            if keyboard.is_pressed("f10"):
                print("Arrêt demandé. Bye.")
                break

            current_time = time.time()

            # --------- QTE EN HAUT -> CHOIX ACTION + ENVOI ---------
            zone = get_zone_for_top_band(sct)
            img = sct.grab(zone)
            frame = np.array(img)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

            # NOTE : Utilise la fonction detect_qte_boxes corrigée de la conversation précédente
            rois, boxes, thresh_qte = detect_qte_boxes(gray, frame)

            # Si on voit au moins une case QTE et qu'on n'a pas spammé récemment
            if rois and (current_time - last_action_time) > QTE_ACTION_COOLDOWN:
                main_roi = rois[0]
                state_tensor = roi_to_state(main_roi) # (1,1,32,32) sur DEVICE
                
                # --- Étape 1 : Sélection d'action ---
                action_idx, eps = select_action(state_tensor, global_step)
                char = classes[action_idx]

                print(f"[STEP] eps={eps:.3f} | action={action_idx} ({char})")

                send_key_for_action(action_idx)

                # --- Étape 2 : Stockage de l'action dans le buffer de séquence ---
                # On stocke le couple (état, action) joué
                # On utilise .cpu() car le ReplayBuffer stocke sur CPU
                sequence_buffer.append((state_tensor.detach().cpu(), action_idx))

                last_action_time = current_time
                global_step += 1

            # --------- LECTURE RESULTAT ----------

            result, res_frame, thresh_res = read_result_zone(sct)
            now = time.time()

            # Vérification du Cooldown avant de traiter le résultat
            cooldown_expired = (now - last_result_time) >= RESULT_COOLDOWN

            if result is not None and cooldown_expired:
                
                # --- Étape 3 : Attribution des récompenses et construction des transitions ---
                transitions_to_push = []
                
                if sequence_buffer:
                    if result == "success":
                        # Logique 'SUCCESS' : Chaque action réussie vaut +1.0 (selon la demande)
                        
                        actions_to_reward = list(sequence_buffer) # On récompense toutes les actions stockées
                        
                        for i in range(len(actions_to_reward)):
                            s, a = actions_to_reward[i]
                            
                            # Récompense positive forte pour chaque action réussie
                            r = 1.0 
                            
                            # Détermination de l'état suivant (s') et du 'done'
                            if i + 1 < len(actions_to_reward):
                                # QTE réussi intermédiaire (n'est pas la fin de l'épisode)
                                s_prime, _ = actions_to_reward[i+1] # s' = état du prochain QTE
                                done_flag = 0.0 
                            else:
                                # Dernier QTE de la série réussie (fin de l'épisode)
                                s_prime = s.clone()
                                done_flag = 1.0 
                            
                            transitions_to_push.append((s, a, r, s_prime, done_flag))
                        
                        print(f"[RL_SUCCESS] Attribué +1.0 à {len(actions_to_reward)} actions réussies.")


                    elif result == "fail":
                        # Logique 'FAIL' : Nouveau comportement demandé par l'utilisateur
                        
                        # On sépare la dernière action (l'échec) du reste de la séquence (les réussites)
                        s_last, a_last = sequence_buffer.pop() # L'action qui a causé le FAIL
                        
                        # 1. Attribuer +1.0 à toutes les actions précédentes (réussies)
                        # Pour chaque QTE réussi dans le buffer de séquence
                        actions_succeed = list(sequence_buffer)
                        
                        for i in range(len(actions_succeed)):
                            s, a = actions_succeed[i]
                            
                            # s_prime est l'état du QTE suivant qui a aussi été réussi
                            s_prime, _ = actions_succeed[i+1] if i + 1 < len(actions_succeed) else s_last.clone()

                            # Récompense positive pour l'action qui était correcte
                            r = 1.0 
                            transitions_to_push.append((s, a, r, s_prime, 0.0)) # done = 0.0 car l'échec n'est pas encore arrivé
                        
                        # 2. Attribuer -1.0 à la dernière action (l'échec)
                        # S' du dernier QTE = s_last (car l'épisode se termine ici)
                        transitions_to_push.append((s_last, a_last, -1.0, s_last.clone(), 1.0)) # done = 1.0 : fin de l'épisode
                        
                        print(f"[RL_FAIL] Attribué +1.0 à {len(actions_succeed)} actions et -1.0 à la dernière.")

                    # --- Étape 4 : Injection dans le ReplayBuffer et Nettoyage ---
                    for s, a, r, s_prime, d in transitions_to_push:
                        replay.push(s, a, r, s_prime, d)
                    
                    # On vide le buffer de séquence après le traitement du FAIL/SUCCESS
                    sequence_buffer.clear()
                    
                    # CORRECTION ICI: Remplacer {final_reward:+.1f} par la somme des récompenses
                    # ou une valeur fixe pour indiquer le résultat.
                    
                    # Calculer la récompense totale de la séquence pour l'affichage :
                    total_r_sequence = sum(r for s, a, r, s_prime, d in transitions_to_push)
                    
                    print(f"[RL] New sequence of {len(transitions_to_push)} transitions pushed. Total R={total_r_sequence:+.1f} | buffer={len(replay)}")
                    
                    # --- Update DQN ---
                    if len(replay) >= max(BATCH_SIZE, START_LEARNING_AFTER):
                        batch = replay.sample(BATCH_SIZE)
                        loss = compute_dqn_loss(batch)

                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(q_net.parameters(), 5.0)
                        optimizer.step()

                        print(f"[RL] DQN update | loss={loss.item():.4f}")

                        if global_step % TARGET_UPDATE_EVERY == 0:
                            target_net.load_state_dict(q_net.state_dict())
                            print(f"[RL] TARGET update @ step {global_step}")
                            
                # --- Mise à jour des Stats et du Cooldown (Indépendant du RL) ---
                if result == "success":
                    total_success += 1
                elif result == "fail":
                    total_fail += 1

                last_result = result
                last_result_time = now 
                
                print(f"[RESULT] {result.upper()} | Succès = {total_success} "
                      f"| Fails = {total_fail} | Next reward in {RESULT_COOLDOWN}s")
            
            # --------- DEBUG VISUEL ----------
            if DEBUG_WINDOWS:
                cv2.imshow("QTE - bande brute + contours", frame)
                cv2.imshow("QTE - thresh", thresh_qte)
                cv2.imshow("Result zone", res_frame)
                cv2.imshow("Result thresh", thresh_res)
                if cv2.waitKey(1) == 27:
                    break
            else:
                cv2.waitKey(1)

            time.sleep(0.01)

    cv2.destroyAllWindows()

    # Sauvegarde finale du réseau Q
    torch.save(q_net.state_dict(), "qte_dqn_online.pth")
    print("[FIN] RL online terminé, modèle sauvegardé -> qte_dqn_online.pth")


if __name__ == "__main__":
    # Assurez-vous que la fonction detect_qte_boxes utilise la version corrigée 
    # de la conversation précédente pour la détection des lettres.
    # ... (code de la fonction detect_qte_boxes)
    main()