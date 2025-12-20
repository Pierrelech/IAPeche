import cv2
import json
import os
import shutil
import time

QUEUE_FILE = "queue.json"
RAW_DIR = "dataset/raw"
OUT_DIR = "dataset/labeled"

os.makedirs(OUT_DIR, exist_ok=True)

def load_queue():
    with open(QUEUE_FILE, "r") as f:
        return json.load(f)

def save_queue(q):
    with open(QUEUE_FILE, "w") as f:
        json.dump(q, f, indent=2)

print("🎨 Interface annotation manuelle")
print("🧵 Les images vont défiler une par une.")
print("⌨️ Tape la lettre/chiffre correspondant → entrée pour valider.")
print("❌ Tape X pour supprimer une image.")

while True:
    queue = load_queue()
    if not queue:
        print("✔ Queue vide, en attente...")
        time.sleep(1)
        continue

    entry = queue[0]
    file = entry["file"]
    path = os.path.join(RAW_DIR, file)

    if not os.path.exists(path):
        queue.pop(0)
        save_queue(queue)
        continue

    img = cv2.imread(path)
    cv2.imshow("Annoter l'image", img)
    cv2.waitKey(1)

    label = input(f"Label pour {file} : ").upper().strip()
    cv2.destroyAllWindows()

    if label == "X":
        print("🗑️ Supprimé.")
        os.remove(path)
        queue.pop(0)
        save_queue(queue)
        continue

    # dossier de la classe
    class_dir = os.path.join(OUT_DIR, label)
    os.makedirs(class_dir, exist_ok=True)

    dest = os.path.join(class_dir, file)
    shutil.move(path, dest)

    print(f"✔ Classé : {file} → {label}")

    queue.pop(0)
    save_queue(queue)
