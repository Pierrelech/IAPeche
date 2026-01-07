# 🎣 IA de Pêche – Creating DataSet and Training the model

Cette branche contient tous les fichiers pour créer un dataset, l'équilibrer, et entrainer un modèle :
- `train_qte_cnn.py` est utilisé pour les images en nuances de gris
- `train_qte_cnn_couleur.py` est pour les images en **couleur (HSV)**,
- `train_qte_mobilenetv3_small.py` pour les petites configs,
- `train_resnet18_qte_couleur` pour le 100% validation accuracy.

---

## ✅ Prérequis

- **Windows 10 / 11**
- **Python 3.12**
- Jeu lancé en **plein écran fenêtré** ou **fenêtré**
- Clavier avec **pavé numérique** (si le jeu attend les touches numpad)

---


## 🚀 Installation

### 1️⃣ Créer un environnement virtuel (recommandé)

Dans le dossier du projet :

```bash
python -m venv .venv
```

Activation :

**PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
```

**CMD**
```bat
.\.venv\Scripts\activate.bat
```

---

### 2️⃣ Installer les dépendances

#### Installation simple (CPU)

```bash
pip install --upgrade pip
pip install numpy opencv-python mss pyautogui keyboard pillow torch torchvision
```

#### Installation avec GPU (CUDA)

Si tu veux utiliser le GPU :
- Va sur le site officiel PyTorch → *Get Started*
- Sélectionne **Windows / pip / ta version CUDA**
- Exécute la commande fournie

#### Puis installe le reste :

```bash
pip install numpy opencv-python mss pyautogui keyboard pillow torchvision
```

---

## ▶️ Lancer la récupération de data

```bash
python capture_dataset_couleur.py
```
ou alors : 
```bash
python capture_dataset.py
```

Fonctionnement :
- lancer la peche sur le jeu
- vous pouvez jouer en même temps (pêcher)
- Les QTE sont détectés et mis dans le dossier dataset_qte/_queue sous cette structure : 

```
.
├─ dataset_qte
      └─ _queue
```

### ⛔ Arrêter la capture
- Appuyer sur **F10**

---

## ▶️ classer les datas


Vous pouvez classer les datas via vos dossiers, ou bien via 

```bash
python anotation_dataset.py
```
ou alors : 
```bash
python annotate_dataset.py
```

Fonctionnement :
- les images apparaissent une par une
- appuyez sur la touche correspondante
- si l'image ne correspond est une mauvaise capture, mettez la dans un dossier non utilisable (exemple x) puis supprimez ce dossier après l'annotation

D'autres fichiers sont disponibles pour équilibrer votre dataset

## ▶️ Entrainer le modèle

Vous pouvez entrainer votre modele via : 

Vous pouvez les faire via vos dossiers, ou bien via 

```bash
python train_qte_cnn.py
```
ou : 
```bash
python train_qte_cnn_couleur.py
```
ou : 
```bash
python train_qte_mobilenetv3_small.py
```
ou pour le 100%: 
```bash
python train_resnet18_qte_couleur2.py
```



## ✨ Auteur

Développé par **Pierre Lechevalier**  
EFREI Paris – Technologies Immersives & IA
