import os

DATASET_ROOT = "dataset_qte"

total = 0

print("=== TAILLE DU DATASET ===\n")

for class_name in sorted(os.listdir(DATASET_ROOT)):
    class_path = os.path.join(DATASET_ROOT, class_name)

    if not os.path.isdir(class_path):
        continue

    num_files = len([
        f for f in os.listdir(class_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    total += num_files
    print(f"Classe '{class_name}' : {num_files} images")

print("\n----------------------------")
print(f"TOTAL DATASET : {total} images")
print("----------------------------")
