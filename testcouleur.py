import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

# -----------------------
# CONFIG
# -----------------------
DATASET_DIR = "dataset_qte"   # adapte si besoin
IMG_SIZE = 64

# -----------------------
# Dataset simple
# -----------------------
class QTEDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        for label in sorted(os.listdir(root_dir)):
            label_dir = os.path.join(root_dir, label)
            if not os.path.isdir(label_dir):
                continue
            for f in os.listdir(label_dir):
                if f.endswith(".png"):
                    self.samples.append((os.path.join(label_dir, f), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # ⚠️ OpenCV -> BGR
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # FIX CRITIQUE

        if self.transform:
            img = self.transform(img)

        return img, label


# -----------------------
# Transforms (DOIVENT matcher l'entraînement)
# -----------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),                 # [0,1]
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

dataset = QTEDataset(DATASET_DIR, transform=transform)

# -----------------------
# VISUALISATION
# -----------------------
img, label = dataset[0]

# Unnormalize pour affichage
mean = torch.tensor([0.5, 0.5, 0.5]).view(3,1,1)
std  = torch.tensor([0.5, 0.5, 0.5]).view(3,1,1)
img_vis = (img * std + mean).clamp(0, 1)

plt.imshow(img_vis.permute(1,2,0))
plt.title(label)
plt.axis("off")
plt.show()
