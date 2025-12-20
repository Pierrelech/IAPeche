import os
import json
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms

# ==============================
# CONFIG
# ==============================

DATASET_ROOT = "dataset_qte"
MODEL_PATH   = "qte_mlp_color.pth"
META_PATH    = "qte_meta_color.json"

IMG_SIZE     = 64          # ton dataset est déjà en 64x64x3, donc parfait
BATCH_SIZE   = 64
NUM_EPOCHS   = 50
LR           = 1e-3
VAL_SPLIT    = 0.2
RANDOM_SEED  = 42

# ==============================-
# SEED & DEVICE
# ==============================

torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device utilisé : {device}")

# ==============================
# TRANSFORMS & DATASET (COULEUR)
# ==============================

# ImageFolder charge via PIL -> RGB par défaut (pas besoin de gérer BGR ici)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),  # -> (3, H, W) en [0,1]
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

if not os.path.isdir(DATASET_ROOT):
    raise FileNotFoundError(f"Le dossier dataset '{DATASET_ROOT}' n'existe pas.")

full_dataset = datasets.ImageFolder(root=DATASET_ROOT, transform=transform)

num_classes = len(full_dataset.classes)
print(f"[INFO] Classes trouvées ({num_classes}) : {full_dataset.classes}")

# Split train / val
val_size = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(RANDOM_SEED)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"[INFO] Taille train : {train_size}, taille val : {val_size}")

# ==============================
# MODELE MLP (COULEUR)
# ==============================

class QTEMLPColor(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        in_features = 3 * IMG_SIZE * IMG_SIZE  # 3 canaux

        self.net = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)  # (B, 3, H, W) -> (B, 3*H*W)
        return self.net(x)

model = QTEMLPColor(num_classes=num_classes).to(device)
print(model)

# ==============================
# LOSS & OPTIM
# ==============================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ==============================
# FONCTIONS TRAIN / EVAL
# ==============================

def train_one_epoch(epoch_idx):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)

    print(f"[TRAIN] Epoch {epoch_idx} - Loss: {epoch_loss:.4f} | Acc: {epoch_acc*100:.2f}%")
    return epoch_loss, epoch_acc


@torch.no_grad()
def eval_one_epoch(epoch_idx):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)

    print(f"[VAL]   Epoch {epoch_idx} - Loss: {epoch_loss:.4f} | Acc: {epoch_acc*100:.2f}%")
    return epoch_loss, epoch_acc

# ==============================
# SANITY CHECK (optionnel)
# ==============================

mini_indices = list(range(min(128, len(full_dataset))))
mini_dataset = Subset(full_dataset, mini_indices)
mini_loader = DataLoader(mini_dataset, batch_size=16, shuffle=True)

print("[SANITY] Taille mini_dataset :", len(mini_dataset))

# Décommente si tu veux tester si le modèle peut overfit un mini set
"""
model = QTEMLPColor(num_classes=num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, 11):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in mini_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    print(f"[SANITY] Epoch {epoch} - Loss: {running_loss/total:.4f} | Acc: {100*correct/total:.2f}%")

exit()
"""

# ==============================
# BOUCLE D'ENTRAINEMENT
# ==============================

best_val_acc = 0.0
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\n===== EPOCH {epoch}/{NUM_EPOCHS} =====")

    train_loss, train_acc = train_one_epoch(epoch)
    val_loss, val_acc = eval_one_epoch(epoch)

    scheduler.step()

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"[SAVE] Nouveau meilleur modèle -> {MODEL_PATH} (val_acc={val_acc*100:.2f}%)")

# ==============================
# SAUVEGARDE METADATA
# ==============================

meta = {
    "classes": full_dataset.classes,
    "class_to_idx": full_dataset.class_to_idx,
    "input_size": [3, IMG_SIZE, IMG_SIZE],
    "normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
    "img_size": IMG_SIZE,
}

with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print("\n[FIN] Entraînement terminé.")
print(f"Meilleure accuracy validation : {best_val_acc*100:.2f}%")
print(f"Modèle -> {MODEL_PATH}")
print(f"Méta   -> {META_PATH}")
