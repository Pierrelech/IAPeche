import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from qte_env import QTEEnv   # ⚠️ importe bien ton env


# ==============================
# CONFIG
# ==============================

DATASET_ROOT = "dataset_qte"

NUM_EPISODES = 2000          # nb "parties" RL
MAX_STEPS_PER_EPISODE = 20   # nombre de QTE consécutives par épisode

GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64

REPLAY_CAPACITY = 10000
START_LEARNING_AFTER = 500    # nb de transitions avant d’update
TARGET_UPDATE_EVERY = 200     # steps avant copie du réseau cible

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 5000              # plus grand = décroissance plus lente

RANDOM_SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] Device :", device)


# ==============================
# RESEAU Q (CNN)
# ==============================

class QNetCNN(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        # input: (B,1,32,32)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # -> (32,32,32)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                 # -> (32,16,16)

            nn.Conv2d(32, 64, 3, padding=1), # -> (64,16,16)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                 # -> (64,8,8)

            nn.Conv2d(64, 128, 3, padding=1),# -> (128,8,8)
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))     # -> (128,1,1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),                    # -> (128,)
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x


# ==============================
# REPLAY BUFFER
# ==============================

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        # On stocke tout en CPU; on déplacera sur device au moment de l’update
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        # s, s2 sont des tenseurs (1,1,32,32) -> on les concatène
        s   = torch.cat(s, dim=0)    # (B,1,32,32)
        s2  = torch.cat(s2, dim=0)   # (B,1,32,32)
        a   = torch.tensor(a, dtype=torch.long)
        r   = torch.tensor(r, dtype=torch.float32)
        d   = torch.tensor(d, dtype=torch.float32)
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buffer)


# ==============================
# FONCTIONS RL
# ==============================

def epsilon_by_step(step_idx):
    # décroissance exponentielle lissée
    eps = EPS_END + (EPS_START - EPS_END) * \
        torch.exp(torch.tensor(-step_idx / EPS_DECAY))
    return float(eps.item())


def select_action(q_net, state, step_idx, num_actions):
    eps = epsilon_by_step(step_idx)
    if random.random() < eps:
        # exploration
        action = random.randrange(num_actions)
        greedy = False
    else:
        # exploitation
        q_values = q_net(state.to(device))  # (1, num_actions)
        action = int(torch.argmax(q_values, dim=1).item())
        greedy = True
    return action, eps, greedy


def compute_dqn_loss(batch, q_net, target_net):
    states, actions, rewards, next_states, dones = batch

    states      = states.to(device)
    next_states = next_states.to(device)
    actions     = actions.to(device)
    rewards     = rewards.to(device)
    dones       = dones.to(device)

    # Q(s,a)
    q_values = q_net(states)                     # (B, num_actions)
    q_sa     = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # max_a' Q_target(s’, a’)
    with torch.no_grad():
        next_q_values = target_net(next_states)
        next_q_max = next_q_values.max(1)[0]
        target = rewards + (1.0 - dones) * GAMMA * next_q_max

    loss = F.mse_loss(q_sa, target)
    return loss


# ==============================
# MAIN TRAIN LOOP
# ==============================

def main():
    # ----- ENV -----
    env = QTEEnv(dataset_root=DATASET_ROOT, device=device)
    if hasattr(env, "num_actions"):
        num_actions = env.num_actions
    else:
        num_actions = len(env.classes)
    print("[INFO] Nombre d’actions :", num_actions)

    # ----- MODELES -----
    q_net      = QNetCNN(num_actions=num_actions).to(device)
    target_net = QNetCNN(num_actions=num_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    replay    = ReplayBuffer(REPLAY_CAPACITY)

    global_step = 0

    for episode in range(1, NUM_EPISODES + 1):
        # 1) RESET
        obs = env.reset()
        if isinstance(obs, torch.Tensor):
            state = obs.detach().to(device) # State S (sur DEVICE)
        else:
            state = torch.tensor(obs, dtype=torch.float32).to(device)

        episode_reward = 0.0

        for t in range(MAX_STEPS_PER_EPISODE):
            global_step += 1

            # 2) Choisir action A pour l'état S
            action, eps, greedy = select_action(q_net, state, global_step, num_actions)

            # 3) Jouer l'action A (interagit avec l'environnement)
            next_obs, reward, done, info = env.step(action) # Renvoie S', R, Done

            # 4) Préparer l'état suivant S'
            if isinstance(next_obs, torch.Tensor):
                next_state = next_obs.detach().to(device)
            elif next_obs is None:
                # Épisode terminé ou échec de sampling
                next_state = torch.zeros_like(state)
            else:
                next_state = torch.tensor(next_obs, dtype=torch.float32).to(device)
            
            # 5) Stocker transition (S, A, R, S', Done) dans le buffer (sur CPU)
            replay.push(state.cpu(), action, reward, next_state.cpu(), float(done))

            episode_reward += reward
            state = next_state # S' devient le nouvel état S

            # 6) Optimisation DQN
            if len(replay) >= max(BATCH_SIZE, START_LEARNING_AFTER):
                batch = replay.sample(BATCH_SIZE)
                loss = compute_dqn_loss(batch, q_net, target_net)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 5.0)
                optimizer.step()

            # 7) Update réseau cible de temps en temps
            if global_step % TARGET_UPDATE_EVERY == 0:
                target_net.load_state_dict(q_net.state_dict())
                print(f"[TARGET] update à step {global_step}")

            # 8) Fin d'épisode ?
            if done:
                break

        print(f"Episode {episode:4d} | steps={t+1:2d} | R={episode_reward:5.1f} | eps={eps:.3f}")

    # Sauvegarde du Q-Net
    torch.save(q_net.state_dict(), "qte_dqn.pth")
    print("[FIN] Entraînement terminé, modèle sauvegardé -> qte_dqn.pth")

if __name__ == "__main__":
    main()