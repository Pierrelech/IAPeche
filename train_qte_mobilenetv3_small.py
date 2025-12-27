import os
import json
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ==============================
# CONFIG
# ==============================
DATASET_ROOT = "dataset_qte"
MODEL_PATH   = "qte_mobilenetv3small_color.pth"
META_PATH    = "qte_meta_color.json"

IMG_SIZE     = 64
BATCH_SIZE   = 32
NUM_EPOCHS   = 30
LR           = 3e-4          # MobileNet supporte un LR un peu plus élevé
VAL_SPLIT    = 0.2
RANDOM_SEED  = 42

# ==============================
# SEED & DEVICE
# ==============================
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device utilisé : {device}")

# ==============================
# TRANSFORMS
# ==============================

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

if not os.path.isdir(DATASET_ROOT):
    raise FileNotFoundError(f"Le dossier dataset '{DATASET_ROOT}' n'existe pas.")

# ==============================
# DATASET + SPLIT
# ==============================

full_dataset_train = datasets.ImageFolder(root=DATASET_ROOT, transform=train_transform)
full_dataset_val   = datasets.ImageFolder(root=DATASET_ROOT, transform=val_transform)

num_classes = len(full_dataset_train.classes)

indices = list(range(len(full_dataset_train)))
random.shuffle(indices)

split = int(len(indices) * VAL_SPLIT)
train_idx, val_idx = indices[split:], indices[:split]

train_dataset = torch.utils.data.Subset(full_dataset_train, train_idx)
val_dataset   = torch.utils.data.Subset(full_dataset_val, val_idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"[INFO] Classes: {full_dataset_train.classes}")
print(f"[INFO] Train: {len(train_dataset)} | Val: {len(val_dataset)}")

# ==============================
# MobileNetV3 Small (PRETRAINED)
# ==============================

model = models.mobilenet_v3_small(
    weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
)

# Remplacer la tête de classification
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, num_classes)

model = model.to(device)
print("[INFO] MobileNetV3 Small chargé")

# ==============================
# LOSS & OPTIM
# ==============================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=LR)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3
)

# ==============================
# TRAIN / EVAL
# ==============================

def train_one_epoch():
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

@torch.no_grad()
def eval_one_epoch():
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

# ==============================
# TRAIN LOOP
# ==============================

best_val_acc = 0.0

for epoch in range(1, NUM_EPOCHS + 1):
    t_loss, t_acc = train_one_epoch()
    v_loss, v_acc = eval_one_epoch()

    scheduler.step(v_acc)

    print(
        f"Epoch {epoch:02d} | "
        f"Train Acc: {t_acc*100:.2f}% | "
        f"Val Acc: {v_acc*100:.2f}% | "
        f"Val Loss: {v_loss:.4f}"
    )

    if v_acc > best_val_acc:
        best_val_acc = v_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  --> Saved Best Model ({v_acc*100:.2f}%)")

# ==============================
# SAVE META
# ==============================

meta = {
    "model": "mobilenet_v3_small",
    "classes": full_dataset_train.classes,
    "class_to_idx": full_dataset_train.class_to_idx,
    "img_size": IMG_SIZE,
    "normalize": {
        "mean": [0.485, 0.456, 0.406],
        "std":  [0.229, 0.224, 0.225]
    }
}

with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)

print(f"\n[FIN] Entraînement terminé.")
print(f"Meilleure accuracy validation : {best_val_acc*100:.2f}%")
print(f"Modèle -> {MODEL_PATH}")
print(f"Méta   -> {META_PATH}")
