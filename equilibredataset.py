from collections import Counter
from torchvision.datasets import ImageFolder

ds = ImageFolder("dataset_qte")  # adapte le chemin
counts = Counter(ds.targets)

# idx -> nom de classe
inv = {v:k for k,v in ds.class_to_idx.items()}

for idx, n in sorted(counts.items(), key=lambda x: x[1], reverse=True):
    print(inv[idx], n)
