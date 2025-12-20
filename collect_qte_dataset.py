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
BAND_HEIGHT = 700       # hauteur de la zone à analyser en haut

# Jeu de caractères qu'on accepte comme labels
ALLOWED_CHARS = set("ABCDEFGHIJKLMNPQRSTUVWYZ123456789")

# Dossier racine du dataset
DATASET_ROOT = "dataset_qte"


def ensure_dataset_dirs():
    """Crée le dossier racine si besoin."""
    if not os.path.exists(DATASET_ROOT):
        os.makedirs(DATASET_ROOT)
        print(f"[INFO] Création du dossier dataset : {DATASET_ROOT}")


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
    Retourne une liste de ROI (images) + leurs bounding boxes.
    """
    # Binarisation inversée : cases claires sur fond plus sombre
    _, thresh = cv2.threshold(frame_gray, 170, 255, cv2.THRESH_BINARY_INV)

    # Petit nettoyage morphologique
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    rois = []
    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h

        # Filtres à adapter si besoin (taille des cases)
        if area < 1000 or area > 8000:
            continue

        ratio = w / float(h)
        if ratio < 0.75 or ratio > 1.3:
            continue

        roi = thresh[y:y + h, x:x + w]

        # Optionnel : agrandir un peu pour aider l'œil humain (et plus tard le CNN)
        roi_resized = cv2.resize(
            roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR
        )

        rois.append(roi_resized)
        boxes.append((x, y, w, h))

        # Visualisation des boîtes sur l'image globale
        cv2.rectangle(frame_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return rois, boxes, thresh


def ask_label_for_roi(roi):
    """
    Affiche le ROI dans une petite fenêtre et demande à l'utilisateur
    de taper la vraie lettre/chiffre.

    Retourne :
      - un caractère (str) si label valide
      - None pour ignorer
      - 'QUIT' pour sortir du programme
    """
    cv2.imshow("ROI - tape la lettre/chiffre (ESC pour quitter, SPACE pour ignorer)", roi)

    while True:
        key = cv2.waitKey(0) & 0xFF

        # ESC -> quitter complètement
        if key == 27:
            return "QUIT"

        # Espace -> ignorer cette image
        if key == 32:
            return None

        char = chr(key).upper()

        if char in ALLOWED_CHARS:
            print(f"[LABEL] -> {char}")
            return char
        else:
            print(f"[WARN] Touche '{char}' non valide. "
                  f"Utilise A-Z / 0-9, ESC pour quitter, SPACE pour ignorer.")


def save_roi(roi, label):
    """
    Sauvegarde le ROI dans le sous-dossier correspondant au label.
    Format compatible avec torchvision.datasets.ImageFolder.
    """
    label_dir = os.path.join(DATASET_ROOT, label)
    os.makedirs(label_dir, exist_ok=True)

    timestamp = int(time.time() * 1000)
    filename = os.path.join(label_dir, f"{label}_{timestamp}.png")

    cv2.imwrite(filename, roi)
    print(f"[SAVE] {filename}")


def main():
    ensure_dataset_dirs()

    print("=== COLLECTEUR DE DATASET QTE ===")
    print("Instructions :")
    print(" - Le script surveille la bande du haut de l'écran 1.")
    print(" - Dès qu'il détecte une case QTE, il t'affiche la case (ROI).")
    print(" - Tu tapes la lettre/chiffre correspondant (A-Z ou 0-9).")
    print(" - SPACE = ignorer la case.")
    print(" - ESC (dans la fenêtre ROI) = quitter le programme.")
    print()

    time.sleep(2)

    with mss.mss() as sct:
        while True:
            zone = get_zone_for_top_band(sct)
            img = sct.grab(zone)
            frame = np.array(img)

            # On garde une copie couleur pour la visualisation globale
            frame_vis = frame.copy()

            # Conversion en niveau de gris
            gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

            # Détection des cases
            rois, boxes, thresh = detect_qte_boxes(gray, frame_vis)

            # Affichage de la zone + thresh pour debug (optionnel)
            cv2.imshow("ZONE QTE (debug)", frame_vis)
            cv2.imshow("THRESH (debug)", thresh)

            # Pour chaque ROI (case détectée), demander un label
            for roi in rois:
                label = ask_label_for_roi(roi)

                if label == "QUIT":
                    print("[INFO] Sortie demandée. Bye.")
                    cv2.destroyAllWindows()
                    return
                elif label is None:
                    # case ignorée
                    continue
                else:
                    save_roi(roi, label)

            # petite pause pour éviter de surcharger
            if cv2.waitKey(1) & 0xFF == 27:
                print("[INFO] ESC dans la fenêtre de debug -> sortie.")
                break

            time.sleep(0.05)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
