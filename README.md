# 🎣 IA de Pêche – Bot QTE (Couleur + MLP PyTorch)

Ce projet exécute automatiquement les **QTE** (Quick Time Events) d’un mini-jeu de pêche en :
- capturant l’écran en temps réel,
- détectant les boutons QTE via leur **couleur (HSV)**,
- reconnaissant lettres et chiffres avec un **MLP PyTorch** entraîné,
- envoyant automatiquement les touches clavier correspondantes.

---

## ✅ Prérequis

- **Windows 10 / 11**
- **Python 3.10 ou supérieur**
- Jeu lancé en **plein écran fenêtré** ou **fenêtré**
- Clavier avec **pavé numérique** (si le jeu attend les touches numpad)

⚠️ Ce script envoie automatiquement des touches clavier.  
À utiliser uniquement dans un cadre **autorisé** (respect des règles du serveur / jeu).

---

## 📁 Structure minimale du projet

Le dossier doit contenir au minimum :

```
.
├─ pechecolor.py
├─ qte_mlp_color.pth
└─ qte_meta_color.json
```

- `pechecolor.py` : script principal
- `qte_mlp_color.pth` : poids du modèle IA
- `qte_meta_color.json` : métadonnées (classes reconnues)

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
- Puis installe le reste :

```bash
pip install numpy opencv-python mss pyautogui keyboard pillow torchvision
```

---

## ▶️ Lancer l’IA

```bash
python pechecolor.py
```

Fonctionnement :
- Attendre ~2 secondes après le lancement
- Mettre le jeu au premier plan
- Les QTE sont détectés et exécutés automatiquement

### ⛔ Arrêter le bot
- Appuyer sur **F10**

---

## ⚙️ Réglages importants (dans `pechecolor.py`)

### 🎯 Zone de capture écran

```python
MONITOR_ID = 1
BAND_HEIGHT = 680
```

- `MONITOR_ID` : numéro de l’écran (1 = écran principal)
- `BAND_HEIGHT` : hauteur capturée depuis le haut de l’écran

👉 Ajuster si les QTE ne sont pas détectés.

---

### 🎚️ Seuil de confiance du modèle

```python
CONF_THRESHOLD = 0.6
```

- Augmenter si des erreurs sont envoyées
- Diminuer si certaines touches ne sont pas reconnues

---

### 🎨 Détection couleur (HSV)

```python
HSV_LOWER = np.array([18, 40, 80], dtype=np.uint8)
HSV_UPPER = np.array([35, 170, 200], dtype=np.uint8)
```

Ces valeurs correspondent aux boutons QTE brun/olive.  
À ajuster si la luminosité, un reshade ou la météo du jeu change.

---

### 🔢 Nombre max de QTE détectés

```python
MAX_BOXES_PER_FRAME = 3
```

Augmenter si plus de boutons apparaissent en même temps.

---

### 🐞 Mode debug visuel

```python
DEBUG_WINDOWS = False
```

Passer à `True` pour afficher :
- la bande capturée
- les rectangles détectés
- le masque HSV

Très utile pour calibrer.

---

## ⌨️ Gestion des touches clavier

- **Chiffres** : envoyés via le pavé numérique (`num0` à `num9`)
- **Lettres** : envoyées via le clavier classique

Code actuel :
```python
pyautogui.press(f"num{c}")
```

👉 Si ton jeu attend les chiffres normaux (ligne du haut), remplace par :
```python
pyautogui.press(c)
```

---

## 🧪 Problèmes courants

### ❌ `ModuleNotFoundError`
➡️ Le venv n’est pas activé ou dépendances manquantes

```bash
pip install keyboard
```

---

### ❌ Aucun QTE détecté
- Vérifier `MONITOR_ID`
- Vérifier `BAND_HEIGHT`
- Activer `DEBUG_WINDOWS = True`

---

### ❌ Touches incorrectes ou instables
- Augmenter `CONF_THRESHOLD`
- Le script stabilise sur plusieurs frames (vote majoritaire)

---

### ❌ Les touches ne sont pas envoyées
- Le jeu doit être **au premier plan**
- Lancer le terminal **en administrateur**
- Désactiver certains overlays (Discord, GeForce, etc.)

---

## 🔐 Notes techniques

- Modèle utilisé : **MLP couleur**
- Taille d’entrée : **64×64 RGB**
- Classes reconnues :
```python
A–Z et 0–9
```

---

## 📜 Licence

À définir (MIT, GPL, privée…).  
Par défaut : **All rights reserved**.

---

## ✨ Auteur

Développé par **Pierre Lechevalier**  
EFREI Paris – Technologies Immersives & IA
