import os
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ==============================
# CONFIG
# ==============================

DATASET_ROOT = "dataset_qte"        # dossier que tu as déjà
MODEL_PATH   = "qte_cnn.pth"        # où sauvegarder le modèle
META_PATH    = "qte_meta.json"      # pour sauvegarder les classes, etc.

BATCH_SIZE   = 64
NUM_EPOCHS   = 50
LR           = 1e-3
VAL_SPLIT    = 0.2                  # 80% train, 20% validation

RANDOM_SEED  = 42

# ==============================
# SEED & DEVICE
# ==============================

torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device utilisé : {device}")


# ==============================
# TRANSFORMS & DATASET
# ==============================

# On force en niveau de gris, on redimensionne et on normalise
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),    # taille fixe pour le CNN
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
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
# MODELE MLP
# ==============================

class QTEMLP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        in_features = 1 * 32 * 32   # car 1 canal, 32x32

        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)  # (B, 1, 32, 32) -> (B, 1024)
        x = self.net(x)      # -> (B, num_classes)
        return x


model = QTEMLP(num_classes=num_classes).to(device)
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

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    print(f"[TRAIN] Epoch {epoch_idx} - Loss: {epoch_loss:.4f} | Acc: {epoch_acc*100:.2f}%")
    return epoch_loss, epoch_acc


def eval_one_epoch(epoch_idx):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)

    print(f"[VAL]   Epoch {epoch_idx} - Loss: {epoch_loss:.4f} | Acc: {epoch_acc*100:.2f}%")
    return epoch_loss, epoch_acc



# ==============================
# SANITY CHECK — vérifier si le modèle PEUT apprendre
# ==============================
from torch.utils.data import Subset

# On prend 128 images max du dataset complet
mini_indices = list(range(min(128, len(full_dataset))))
mini_dataset = Subset(full_dataset, mini_indices)
mini_loader = DataLoader(mini_dataset, batch_size=16, shuffle=True)

print("[SANITY] Taille mini_dataset :", len(mini_dataset))

# Nouveau modèle propre
# Nouveau modèle propre
model = QTEMLP(num_classes=num_classes).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

'''
# Entraînement sur ce mini-dataset
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
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    print(f"[SANITY] Epoch {epoch} - Loss: {running_loss/total:.4f} | "
          f"Acc: {100*correct/total:.2f}%")

# On stoppe le script ici pour ne pas lancer l'entraînement normal
exit()
'''
# ==============================
# BOUCLE D'ENTRAINEMENT
# ==============================

best_val_acc = 0.0
history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
}




for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\n===== EPOCH {epoch}/{NUM_EPOCHS} =====")

    train_loss, train_acc = train_one_epoch(epoch)
    val_loss, val_acc = eval_one_epoch(epoch)

    scheduler.step()

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    # Sauvegarde du meilleur modèle
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"[SAVE] Nouveau meilleur modèle sauvegardé -> {MODEL_PATH} "
              f"(val_acc={val_acc*100:.2f}%)")

# ==============================
# SAUVEGARDE METADATA
# ==============================

meta = {
    "classes": full_dataset.classes,
    "class_to_idx": full_dataset.class_to_idx,
    "input_size": [1, 32, 32],
}
with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print("\n[FIN] Entraînement terminé.")
print(f"Meilleure accuracy validation : {best_val_acc*100:.2f}%")
print(f"Modèle -> {MODEL_PATH}")
print(f"Méta   -> {META_PATH}")
