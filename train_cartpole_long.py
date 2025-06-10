import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import os
from environment import CartPoleEnv
from pilco import PILCO

# --- Settings ---
SEED = 1337
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

LOGDIR = "pilco_logs"
os.makedirs(LOGDIR, exist_ok=True)

EPISODES = 20      # Anzahl der Trainingszyklen, ruhig hoch stellen
ROLLOUT_STEPS = 200
POLICY_FEAT = 10


def evaluate_policy(env, pilco, episodes=5, steps=200):
    rewards = []
    for _ in range(episodes):
        state = env.reset()
        ep_reward = 0
        for _ in range(steps):
            state_torch = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                action = pilco.policy(state_torch).numpy().squeeze()
            # Diskrete Policy für Gym-CartPole:
            action = int(action >= 0)
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
    return np.mean(rewards), np.std(rewards)

if __name__ == "__main__":
    env = CartPoleEnv(mode="sim")
    pilco = PILCO(state_dim=4, action_dim=1, policy_feat=POLICY_FEAT)

    # Logging
    avg_returns = []
    std_returns = []
    policy_params_list = []

    # Erstes Rollout (zum initialisieren)
    pilco.rollout(env, steps=ROLLOUT_STEPS)
    pilco.train_model()

    for epoch in range(EPISODES):
        print(f"== Training Epoch {epoch+1}/{EPISODES} ==")

        # Policy-Optimierung
        init_mean = np.zeros(4)
        init_cov = np.eye(4)*0.1
        pilco.optimize_policy(init_mean, init_cov, horizon=100, lr=1e-2, steps=50)

        # Evaluate policy (mehrere Rollouts, Mittelwert/Std loggen)
        mean_ret, std_ret = evaluate_policy(env, pilco, episodes=5, steps=ROLLOUT_STEPS)
        print(f"Policy Test-Return: {mean_ret:.2f} ± {std_ret:.2f}")
        avg_returns.append(mean_ret)
        std_returns.append(std_ret)

        # Policy-Parameter speichern
        params = [p.clone().detach().cpu().numpy() for p in pilco.policy.parameters()]
        policy_params_list.append(params)

        # Neue Daten sammeln
        pilco.rollout(env, steps=ROLLOUT_STEPS)
        pilco.train_model()

        # (Optional: nach jedem n-ten Zyklus Policy speichern)
        if (epoch+1) % 10 == 0:
            torch.save(pilco.policy.state_dict(), os.path.join(LOGDIR, f"policy_epoch_{epoch+1}.pt"))

    # Save final Policy
    torch.save(pilco.policy.state_dict(), os.path.join(LOGDIR, "policy_final.pt"))
    print(f"Final policy saved to {os.path.join(LOGDIR, 'policy_final.pt')}")

    # Save logs
    np.save(os.path.join(LOGDIR, "avg_returns.npy"), np.array(avg_returns))
    np.save(os.path.join(LOGDIR, "std_returns.npy"), np.array(std_returns))
    np.save(os.path.join(LOGDIR, "policy_params.npy"), np.array(policy_params_list, dtype=object))

    # Plot returns
    plt.figure(figsize=(8,5))
    plt.plot(avg_returns, label="Mean Return (eval)")
    plt.fill_between(range(len(avg_returns)),
                     np.array(avg_returns)-np.array(std_returns),
                     np.array(avg_returns)+np.array(std_returns), alpha=0.3, label="±1 Std")
    plt.xlabel("Epoch")
    plt.ylabel("Episode Return")
    plt.title("PILCO Training Performance (CartPole)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(LOGDIR, "training_return.png"))
    plt.close()
    print(f"Training plot saved to {os.path.join(LOGDIR, 'training_return.png')}")

    # Optionally: Plot policy weights evolution (as a “fingerprint”)
    fig, axs = plt.subplots(len(policy_params_list[0]), 1, figsize=(8,2*len(policy_params_list[0])))
    for i, p in enumerate(policy_params_list[0]):
        weights = [params[i].flatten() for params in policy_params_list]
        weights = np.stack(weights)
        for j in range(weights.shape[1]):
            axs[i].plot(weights[:,j], label=f"W{i}[{j}]")
        axs[i].set_ylabel(f"Param {i}")
        axs[i].legend()
    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(LOGDIR, "policy_weights.png"))
    plt.close()
    print(f"Policy weights plot saved to {os.path.join(LOGDIR, 'policy_weights.png')}")

    env.close()
    print("Training abgeschlossen. Ergebnisse und Modelle gespeichert.")
