from qte_env import QTEEnv

def main():
    env = QTEEnv(
        dataset_root="dataset_qte",
        device="cpu",      # ou "cpu"
        episode_len=5,
        img_size=32,
    )

    obs = env.reset()
    print("Obs shape :", obs.shape)   # ex: torch.Size([1, 1, 32, 32])

    done = False
    step = 0
    while not done:
        action = env.sample_random_action()  # pour l'instant : politique random
        next_obs, reward, done, info = env.step(action)

        print(f"Step {step} | action={action} ({env.idx_to_class[action]}) | "
              f"target={info['target_class']} | reward={reward}")

        obs = next_obs
        step += 1

if __name__ == "__main__":
    main()
