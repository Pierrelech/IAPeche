import os
import cv2
import time
import shutil


QUEUE_DIR = "dataset_qte/_queue"        # là où la capture stocke les images à classer
DATASET_DIR = "dataset_qte"    # ImageFolder PyTorch
ALLOWED_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


def ensure_dirs():
    os.makedirs(QUEUE_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)


def get_next_image():
    """Retourne le chemin d'une image non annotée, ou None si la file est vide."""
    imgs = [f for f in os.listdir(QUEUE_DIR)
            if f.lower().endswith(".png")]

    if not imgs:
        return None

    return os.path.join(QUEUE_DIR, imgs[0])


def save_to_dataset(img_path, label):
    """Déplace une image dans dataset_qte/<LABEL>/"""
    label_dir = os.path.join(DATASET_DIR, label)
    os.makedirs(label_dir, exist_ok=True)

    filename = os.path.basename(img_path)
    dst = os.path.join(label_dir, filename)

    shutil.move(img_path, dst)
    print(f"[SAVE] {img_path} -> {dst}")


def delete_image(img_path):
    """Supprime l'image de la file."""
    os.remove(img_path)
    print(f"[DELETE] {img_path}")


def annotate_loop():
    print("=== ANNOTATEUR QTE ===")
    print("Instructions :")
    print(" - Une image est affichée.")
    print(" - Tape sa lettre (A-Z ou 0-9) pour l'ajouter au dataset.")
    print(" - SPACE = supprimer l'image.")
    print(" - ESC = quitter.")
    print()

    while True:
        img_path = get_next_image()

        if img_path is None:
            print("[INFO] Aucune image dans la file. Attente...")
            time.sleep(1)
            continue

        img = cv2.imread(img_path)
        if img is None:
            delete_image(img_path)
            continue

        cv2.imshow("ANNOTATION (ESC=quit, SPACE=del)", img)

        key = cv2.waitKey(0) & 0xFF

        # ESC → quitter proprement
        if key == 27:
            print("[INFO] Sortie demandée.")
            cv2.destroyAllWindows()
            break

        # SPACE → supprimer l’image
        if key == 32:
            delete_image(img_path)
            cv2.destroyAllWindows()
            continue

        # lettre ou chiffre
        char = chr(key).upper()

        if char in ALLOWED_CHARS:
            save_to_dataset(img_path, char)
        else:
            print(f"[WARN] '{char}' non valide → suppression")
            delete_image(img_path)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    ensure_dirs()
    annotate_loop()

