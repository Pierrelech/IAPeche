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
DATASET_ROOT = "dataset_qte"    # Charge mon dataset d'images couleur
MODEL_PATH   = "qte_resnet18_color.pth"     # Nouveau modèle ResNet18
META_PATH    = "qte_meta_color.json"        # Métadonnées

IMG_SIZE     = 64       # ResNet18 fonctionne bien avec 64x64 donc on met 64
BATCH_SIZE   = 32     # Petit batch size pour ResNet18 (plus lourd)
NUM_EPOCHS   = 30     # Nombre d'époques
LR           = 1e-4   # Taux d'apprentissage
VAL_SPLIT    = 0.2    # split des données avec 20% pour validation
RANDOM_SEED  = 42       # Pour reproductibilité

# ==============================
# SEED & DEVICE
# ==============================
torch.manual_seed(RANDOM_SEED)  # rend l'init et certains tirages Pytorch reproductibles
random.seed(RANDOM_SEED)        # rend le shuffle reproductible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device utilisé : {device}")

# ==============================
# TRANSFORMS & DATASET
# ==============================

# Training transforms inclut l'augmentation de données pour éviter l'overfitting
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(10),             # Fais une rotation aléatoire de +/- 10 degrés pour permettre au modèle de mieux généraliser
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Gère les variations de luminosité et de contraste
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Validation transforms sans augmentation
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

if not os.path.isdir(DATASET_ROOT):
    raise FileNotFoundError(f"Le dossier dataset '{DATASET_ROOT}' n'existe pas.")

# charge les datasets avec les transforms appropriés
full_dataset_train = datasets.ImageFolder(root=DATASET_ROOT, transform=train_transform)
full_dataset_val = datasets.ImageFolder(root=DATASET_ROOT, transform=val_transform)

num_classes = len(full_dataset_train.classes)   # nombre de classes
indices = list(range(len(full_dataset_train)))  # indices pour le split pour permettre le shuffle
random.shuffle(indices)                         # shuffle des indices pour permettre un bon split

split = int(len(indices) * VAL_SPLIT)           # point de split pour val afin d'avoir VAL_SPLIT % pour la val
train_idx, val_idx = indices[split:], indices[:split]   # split des indices pour permettre le DataLoader de faire son boulot

train_dataset = torch.utils.data.Subset(full_dataset_train, train_idx)  # subset pour le train afin de gérer le split
val_dataset = torch.utils.data.Subset(full_dataset_val, val_idx)    # subset pour la val

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)   # shuffle pour le train
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)    # pas de shuffle pour la val

print(f"[INFO] Classes: {full_dataset_train.classes}")
print(f"[INFO] Train: {len(train_dataset)}, Val: {len(val_dataset)}")

# ==============================
# RESNET18 MODEL SETUP
# ==============================

# Utilisation d'un modèle ResNet18 pré-entraîné sur ImageNet
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Remplace la dernière couche fully connected pour correspondre au nombre de classes de mon dataset
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

model = model.to(device)

# ==============================
# LOSS & OPTIM
# ==============================
criterion = nn.CrossEntropyLoss()   # Classification multi-classes
optimizer = optim.Adam(model.parameters(), lr=LR)   # Adam optimizer pour de meilleures performances
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)   # Réduit le LR si la val accuracy stagne

# ==============================
# TRAIN / EVAL FUNCTIONS
# ==============================

def train_one_epoch():
    model.train()   # Mode entraînement
    running_loss, correct, total = 0.0, 0, 0    # statistiques
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)   # envoi au device pour accélération
        
        optimizer.zero_grad()               # reset des gradients pour cette itération
        outputs = model(images)             # propagation avant pour obtenir les prédictions
        loss = criterion(outputs, labels)   # compute la loss afin d'avoir les résultats
        loss.backward()                     # retropropagation pour calculer les gradients et adapter les poids
        optimizer.step()                    # update des poids  
        
        running_loss += loss.item() * images.size(0)            # accumulateur de loss
        correct += (outputs.argmax(1) == labels).sum().item()   # accumulateur de bonnes prédictions
        total += labels.size(0)                                 # accumulateur du total d'images
    return running_loss / total, correct / total    

@torch.no_grad()    # Evaluation sans calcul de gradients
def eval_one_epoch():   
    model.eval()        # Mode évaluation
    running_loss, correct, total = 0.0, 0, 0    # statistiques
    for images, labels in val_loader:           
        images, labels = images.to(device), labels.to(device)   # envoi au device pour accélération
        outputs = model(images)                                 # propagation avant pour obtenir les prédictions
        loss = criterion(outputs, labels)                       # compute la loss afin d'avoir les résultats
        running_loss += loss.item() * images.size(0)            # accumulateur de loss
        correct += (outputs.argmax(1) == labels).sum().item()   # accumulateur de bonnes prédictions
        total += labels.size(0)                                 # accumulateur du total d'images
    return running_loss / total, correct / total

# ==============================
# MAIN LOOP
# ==============================
best_val_acc = 0.0

for epoch in range(1, NUM_EPOCHS + 1):
    t_loss, t_acc = train_one_epoch()           # Entraînement
    v_loss, v_acc = eval_one_epoch()            # Évaluation
    
    scheduler.step(v_acc)                   # Scheduler pour ajuster le LR si besoin
    
    print(f"Epoch {epoch:02d} | Train Acc: {t_acc*100:.2f}% | Val Acc: {v_acc*100:.2f}% | Loss: {v_loss:.4f}")
    
    if v_acc > best_val_acc:    # sauvegarde du meilleur modèle si meilleure accuracy
        best_val_acc = v_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  --> Saved Best Model ({v_acc*100:.2f}%)")

# ==============================
# SAVE METADATA
# ==============================
# Sauvegarde des métadonnées nécessaires pour l'inférence
meta = {
    "classes": full_dataset_train.classes,              # Liste des classes du dataset
    "class_to_idx": full_dataset_train.class_to_idx,    # Mapping classe -> index
    "img_size": IMG_SIZE,                           # Taille d'entrée du modèle
    "model_type": "resnet18"                        # Type de modèle utilisé
}

with open(META_PATH, "w") as f:                     # sauvegarde au format JSON
    json.dump(meta, f)

print(f"\n[FIN] Entraînement terminé. Meilleure Accuracy: {best_val_acc*100:.2f}%")