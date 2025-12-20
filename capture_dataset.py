import os
import time

import mss
import numpy as np
import cv2
import keyboard

# -----------------------------
# CONFIG ECRAN + ZONE A LIRE
# -----------------------------
MONITOR_ID = 1          # écran 1
BAND_HEIGHT = 260       # hauteur de la zone à analyser en haut
MAX_BOXES_PER_FRAME = 3 # Limite de cases QTE par frame

# Dossier racine du dataset
DATASET_ROOT = "dataset_qte"
QUEUE_DIR = os.path.join(DATASET_ROOT, "_queue") # file d'attente d'images brutes


def ensure_dataset_dirs():
    """Crée les dossiers racine + file d'attente si besoin."""
    os.makedirs(DATASET_ROOT, exist_ok=True)
    os.makedirs(QUEUE_DIR, exist_ok=True)
    print(f"[INFO] Dossier dataset      : {DATASET_ROOT}")
    print(f"[INFO] Dossier file attente : {QUEUE_DIR}")


def get_zone_for_top_band(sct):
    """Zone correspondant à la bande du haut de l'écran 1."""
    mon = sct.monitors[MONITOR_ID]
    zone = {
        "left": mon["left"],
        "top": mon["top"],
        "width": mon["width"],
        "height": BAND_HEIGHT
    }
    return zone


def detect_qte_boxes(frame_gray, frame_vis):
    """
    À partir d'une image en niveaux de gris, détecte les cases QTE avec des contours.
    Retourne une liste de ROI (images) + leurs bounding boxes + l'image thresh.
    
    LOGIQUE ADAPTÉE : Utilisation de OTSU et re-seuillage du ROI pour isoler le caractère.
    """
    # 1. Détection des BORDURES des QTE (utilisation de OTSU)
    # Isoler la forme extérieure du QTE sur fond sombre (OTSU)
    _, thresh_qte_shape = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Inversion pour la détection de CONTOURS (les bordures sont blanches sur fond noir)
    thresh = cv2.bitwise_not(thresh_qte_shape) 
    
    # Opérations morphologiques pour nettoyer les formes
    kernel_open = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=2) 
    kernel_dilate = np.ones((2, 2), np.uint8)
    thresh = cv2.dilate(thresh, kernel_dilate, iterations=1)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 2. Filtrage des contours
    candidate_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        
        # Filtres de taille et ratio
        if area < 1200 or area > 6000:
            continue
        ratio = w / float(h)
        if ratio < 0.7 or ratio > 1.4:
            continue
        if y < 10: 
            continue

        candidate_boxes.append((x, y, w, h))

    # Limiter le nombre de boîtes à traiter
    candidate_boxes.sort(key=lambda b: b[1], reverse=True) 
    boxes_to_use = candidate_boxes[:MAX_BOXES_PER_FRAME]


    rois = []
    boxes = []
    
    # 3. Extraction et Seillage du ROI (pour isoler le caractère)
    for (x, y, w, h) in boxes_to_use:
        
        # Extraction de la zone d'intérêt originale en niveaux de gris
        roi_original_gray = frame_gray[y:y + h, x:x + w]
        
        # Seuil inversé pour la classification (met le caractère sombre en blanc)
        _, roi_final = cv2.threshold(roi_original_gray, 120, 255, cv2.THRESH_BINARY_INV) 

        # Agrandir le ROI 
        roi_resized = cv2.resize(
            roi_final, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR
        )

        rois.append(roi_resized)
        boxes.append((x, y, w, h))
        
        # Visualisation des boîtes sur l'image globale
        cv2.rectangle(frame_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    # On renvoie le thresh_qte_shape pour le debug visuel
    return rois, boxes, thresh_qte_shape


def enqueue_roi(roi):
    """
    Sauvegarde le ROI dans le dossier de file d'attente.
    Pas encore de label : juste un ID unique.
    """
    timestamp = int(time.time() * 1000)
    filename = os.path.join(QUEUE_DIR, f"pending_{timestamp}.png")
    cv2.imwrite(filename, roi)
    print(f"[QUEUE] Image ajoutée : {filename}")


def main():
    ensure_dataset_dirs()

    print("=== COLLECTEUR DE DATASET QTE (MODE FILE D'ATTENTE) ===")
    print(" - Capture la bande du haut de l'écran 1.")
    print(" - 1 capture par 0.5 secondes.")
    print(" - Si une case QTE est détectée, chaque ROI est ajouté dans la file d'attente :")
    print(f"     {QUEUE_DIR}")
    print(" - Tu feras un 2e script pour dépiler et labeller ces images plus tard.")
    print(" - ESC dans la fenêtre debug ou F10 au clavier pour quitter.")
    print()

    time.sleep(2)

    last_capture_time = 0.0

    with mss.mss() as sct:
        while True:
            # sortie d'urgence par clavier
            if keyboard.is_pressed("f10"):
                print("[INFO] F10 -> sortie.")
                break

            now = time.time()

            # On ne capture qu'une fois par 0.5 secondes
            if now - last_capture_time >= 0.5:
                last_capture_time = now

                zone = get_zone_for_top_band(sct)
                img = sct.grab(zone)
                frame = np.array(img)

                # On garde une copie couleur pour la visualisation globale
                frame_vis = frame.copy()

                # Conversion en niveau de gris
                gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)

                # Détection des cases
                # 'thresh' contient maintenant l'image OTSU pour le debug
                rois, boxes, thresh = detect_qte_boxes(gray, frame_vis)

                # Ajout des ROIs détectés dans la file d'attente
                if rois:
                    print(f"[INFO] {len(rois)} case(s) détectée(s) sur cette frame.")
                    for roi in rois:
                        enqueue_roi(roi)

                # Affichage de la zone + thresh pour debug (optionnel)
                cv2.imshow("ZONE QTE (debug)", frame_vis)
                cv2.imshow("THRESH (debug)", thresh)

            # Gestion des fenêtres OpenCV + sortie ESC
            key = cv2.waitKey(1) & 0xFF
            if key == 27: # ESC
                print("[INFO] ESC dans la fenêtre debug -> sortie.")
                break

            # petite pause pour ne pas faire tourner à 100% CPU
            time.sleep(0.01)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()