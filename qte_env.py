# qte_env.py

import random
from typing import Tuple, Dict, Any

import torch
from torch import Tensor
from torchvision import datasets, transforms


class QTEEnv:
    """
    Environnement RL pour QTE basé sur ton dataset ImageFolder.

    - Observation : image (1, 32, 32) en niveaux de gris, normalisée
    - Action : int dans [0, num_actions-1] => index dans self.classes
    - Reward : +1 si action == label, -1 sinon
    - Chaque step tire une nouvelle image indépendamment (épisode de longueur 1)
    """

    def __init__(
        self,
        dataset_root: str = "dataset_qte",
        device: torch.device | None = None,
        seed: int = 42,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # mêmes transforms que pour ton MLP (32x32 + grayscale + normalisation)
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        self.max_steps_per_episode = 10
        self.steps_in_episode = 0


        self.dataset = datasets.ImageFolder(
            root=dataset_root,
            transform=self.transform
        )
        self.classes = self.dataset.classes
        self.n_actions = len(self.classes)

        print(f"[QTEEnv] Classes ({self.n_actions}) : {self.classes}")

        self.rng = random.Random(seed)

        # état courant
        self.current_obs: Tensor | None = None
        self.current_label_idx: int | None = None

    # -----------------------------
    # utils internes
    # -----------------------------
    def _sample_obs_and_label(self) -> Tuple[Tensor, int]:
        """
        Tire aléatoirement une image du dataset et renvoie :
        - obs : Tensor (1,1,32,32) sur self.device
        - label_idx : int
        Cette fonction NE renvoie JAMAIS None.
        """
        while True:
            idx = self.rng.randrange(len(self.dataset))
            img, label_idx = self.dataset[idx]   # img déjà transformée

            if not isinstance(img, torch.Tensor):
                # normallement jamais ici, mais on sécurise
                img = self.transform(img)

            # img est (1,32,32) => on ajoute une dimension batch
            obs = img.unsqueeze(0).to(self.device)  # (1,1,32,32)
            return obs, label_idx

    # -----------------------------
    # API style Gym
    # -----------------------------
    def reset(self):
        self.steps_in_episode = 0
        obs, label_idx = self._sample_obs_and_label()
        self.current_obs = obs
        self.current_label_idx = label_idx
        return obs


    def step(self, action: int):
        """
        action : int dans [0, n_actions-1]
        Renvoie :
        next_obs : Tensor (1,1,32,32)
        reward   : float
        done     : bool
        info     : dict
        """
        assert self.current_obs is not None
        assert self.current_label_idx is not None

        # --- Récompense ---
        correct = (action == self.current_label_idx)
        reward = 1.0 if correct else -1.0

        # --- Compteur de steps ---
        self.steps_in_episode += 1

        # épisode terminé si on atteint max_steps
        done = (self.steps_in_episode >= self.max_steps_per_episode)

        info = {
            "correct_label_idx": self.current_label_idx,
            "correct_label": self.classes[self.current_label_idx],
            "success": correct,
        }

        # On prépare déjà le prochain état (sera utilisé au step suivant)
        next_obs, next_label_idx = self._sample_obs_and_label()
        self.current_obs = next_obs
        self.current_label_idx = next_label_idx

        return next_obs, reward, done, info

