import os
import json
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# ==============================
# CONFIG
# ==============================
DATASET_ROOT = "dataset_qte"
MODEL_PATH   = "qte_resnet18_color2.pth"
META_PATH    = "qte_meta_color2.json"

IMG_SIZE     = 64 
BATCH_SIZE   = 32     # Reduced batch size often helps ResNet generalize better
NUM_EPOCHS   = 30     # ResNet converges faster; 30 is usually plenty
LR           = 1e-4   # Lower learning rate for fine-tuning
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
# TRANSFORMS & DATASET
# ==============================

# Training transforms include Augmentation to prevent overfitting
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1), # Handles lighting changes
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Validation transforms must ONLY resize and normalize
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

if not os.path.isdir(DATASET_ROOT):
    raise FileNotFoundError(f"Le dossier dataset '{DATASET_ROOT}' n'existe pas.")

# Load full dataset twice to apply different transforms
full_dataset_train = datasets.ImageFolder(root=DATASET_ROOT, transform=train_transform)
full_dataset_val = datasets.ImageFolder(root=DATASET_ROOT, transform=val_transform)

num_classes = len(full_dataset_train.classes)
indices = list(range(len(full_dataset_train)))
random.shuffle(indices)

split = int(len(indices) * VAL_SPLIT)
train_idx, val_idx = indices[split:], indices[:split]

train_dataset = torch.utils.data.Subset(full_dataset_train, train_idx)
val_dataset = torch.utils.data.Subset(full_dataset_val, val_idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"[INFO] Classes: {full_dataset_train.classes}")
print(f"[INFO] Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# ==============================
# RESNET18 MODEL SETUP
# ==============================

# Using pre-trained weights is the secret to 100% accuracy
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Replace the last layer (fc) to match your QTE class count
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

model = model.to(device)

# ==============================
# LOSS & OPTIM
# ==============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
# Reduce learning rate when accuracy plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# ==============================
# TRAIN / EVAL FUNCTIONS
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
# MAIN LOOP
# ==============================
best_val_acc = 0.0

for epoch in range(1, NUM_EPOCHS + 1):
    t_loss, t_acc = train_one_epoch()
    v_loss, v_acc = eval_one_epoch()
    
    scheduler.step(v_acc)
    
    print(f"Epoch {epoch:02d} | Train Acc: {t_acc*100:.2f}% | Val Acc: {v_acc*100:.2f}% | Loss: {v_loss:.4f}")
    
    if v_acc > best_val_acc:
        best_val_acc = v_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  --> Saved Best Model ({v_acc*100:.2f}%)")

# ==============================
# SAVE METADATA
# ==============================
meta = {
    "classes": full_dataset_train.classes,
    "class_to_idx": full_dataset_train.class_to_idx,
    "img_size": IMG_SIZE,
    "model_type": "resnet18"
}

with open(META_PATH, "w") as f:
    json.dump(meta, f)

print(f"\n[FIN] Entraînement terminé. Meilleure Accuracy: {best_val_acc*100:.2f}%")