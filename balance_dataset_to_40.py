import os
import random
import shutil

# -----------------------------
# CONFIG
# -----------------------------
DATASET_ROOT = "dataset_qte"
TARGET_COUNT = 40

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def is_image(filename):
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def main():
    print("=== ÉQUILIBRAGE DU DATASET À 40 IMAGES PAR CLASSE ===\n")

    for class_name in os.listdir(DATASET_ROOT):
        class_dir = os.path.join(DATASET_ROOT, class_name)

        if not os.path.isdir(class_dir):
            continue

        images = [f for f in os.listdir(class_dir) if is_image(f)]
        count = len(images)

        print(f"Classe '{class_name}' : {count} images")

        if count == 0:
            print(f"⚠️  Classe '{class_name}' ignorée (aucune image)")
            continue

        if count >= TARGET_COUNT:
            print(f"✅ Déjà suffisante\n")
            continue

        needed = TARGET_COUNT - count
        print(f"➡️  Duplication de {needed} image(s)...")

        for i in range(needed):
            src_name = random.choice(images)
            src_path = os.path.join(class_dir, src_name)

            base, ext = os.path.splitext(src_name)
            new_name = f"{base}_dup_{i}{ext}"
            dst_path = os.path.join(class_dir, new_name)

            shutil.copy(src_path, dst_path)

        final_count = len([f for f in os.listdir(class_dir) if is_image(f)])
        print(f"✅ Classe '{class_name}' maintenant à {final_count} images\n")

    print("=== ÉQUILIBRAGE TERMINÉ ===")


if __name__ == "__main__":
    main()
