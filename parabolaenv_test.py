import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import os
from pilco import PILCO
from parabolaenv import RollingBallParabolaEnv  # <- deine Environment-Klasse importieren

# --- Setze Seed für Reproduzierbarkeit ---
SEED = 1337
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

LOGDIR = "pilco_parabola_logs"
os.makedirs(LOGDIR, exist_ok=True)

EPISODES = 5
ROLLOUT_STEPS = 40
POLICY_FEAT = 5

# --- Reward-Funktion ---
def reward_fn(x):
    return torch.exp(-0.5 * x[..., 0]**2)  # Bestrafe nur Abweichung in x

# --- Evaluation ---
def evaluate_policy(env, pilco, episodes=5, steps=40):
    rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        ep_reward = 0
        for _ in range(steps):
            state_torch = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                action = pilco.policy(state_torch).numpy().squeeze()
            action = np.array([action], dtype=np.float32)
            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break
        rewards.append(ep_reward)
    return np.mean(rewards), np.std(rewards)

# --- Training ---
if __name__ == "__main__":
    env = RollingBallParabolaEnv()
    pilco = PILCO(state_dim=2, action_dim=1, policy_feat=POLICY_FEAT)
    pilco.reward = reward_fn

    avg_returns = []
    std_returns = []
    policy_params_list = []

    # Erstes Rollout (random Policy)
    pilco.rollout(env, steps=ROLLOUT_STEPS)
    pilco.train_model()

    for epoch in range(EPISODES):
        print(f"== Training Epoch {epoch + 1}/{EPISODES} ==")

        # Policy-Optimierung
        init_mean = np.zeros(2)
        init_cov = np.eye(2) * 0.2
        pilco.optimize_policy(init_mean, init_cov, horizon=20, lr=1e-2, steps=30)

        # Evaluation
        mean_ret, std_ret = evaluate_policy(env, pilco, episodes=5, steps=ROLLOUT_STEPS)
        print(f"Policy Test-Return: {mean_ret:.2f} ± {std_ret:.2f}")
        avg_returns.append(mean_ret)
        std_returns.append(std_ret)

        # Policy-Parameter speichern
        params = [p.clone().detach().cpu().numpy() for p in pilco.policy.parameters()]
        policy_params_list.append(params)

        # Neue Daten sammeln und Modell updaten
        pilco.rollout(env, steps=ROLLOUT_STEPS)
        pilco.train_model()

        # Speichern
        if (epoch + 1) % 10 == 0:
            torch.save(pilco.policy.state_dict(), os.path.join(LOGDIR, f"policy_epoch_{epoch + 1}.pt"))

    # Final speichern
    torch.save(pilco.policy.state_dict(), os.path.join(LOGDIR, "policy_final.pt"))
    np.save(os.path.join(LOGDIR, "avg_returns.npy"), np.array(avg_returns))
    np.save(os.path.join(LOGDIR, "std_returns.npy"), np.array(std_returns))
    np.save(os.path.join(LOGDIR, "policy_params.npy"), np.array(policy_params_list, dtype=object))
    print(f"Final policy saved to {os.path.join(LOGDIR, 'policy_final.pt')}")

    # Plot: Returns
    plt.figure(figsize=(8, 5))
    plt.plot(avg_returns, label="Mean Return (eval)")
    plt.fill_between(range(len(avg_returns)),
                     np.array(avg_returns) - np.array(std_returns),
                     np.array(avg_returns) + np.array(std_returns), alpha=0.3, label="±1 Std")
    plt.xlabel("Epoch")
    plt.ylabel("Episode Return")
    plt.title("PILCO ParabolaEnv Training Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(LOGDIR, "training_return.png"))
    plt.close()
    print(f"Training plot saved to {os.path.join(LOGDIR, 'training_return.png')}")

    # Plot: Policy weights evolution
    fig, axs = plt.subplots(len(policy_params_list[0]), 1, figsize=(8, 2 * len(policy_params_list[0])))
    for i, p in enumerate(policy_params_list[0]):
        weights = [params[i].flatten() for params in policy_params_list]
        weights = np.stack(weights)
        for j in range(weights.shape[1]):
            axs[i].plot(weights[:, j], label=f"W{i}[{j}]")
        axs[i].set_ylabel(f"Param {i}")
        axs[i].legend()
    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(LOGDIR, "policy_weights.png"))
    plt.close()
    print(f"Policy weights plot saved to {os.path.join(LOGDIR, 'policy_weights.png')}")

    env.close()
    print("Training abgeschlossen. Ergebnisse und Modelle gespeichert.")
