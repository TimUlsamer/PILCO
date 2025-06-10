import numpy as np
from pilco import PILCO
import torch

# --- Einfaches künstliches Environment ---
class SyntheticEnv:
    """
    x_{t+1} = x_t + 0.5 * sin(a) + noise
    """
    def __init__(self):
        self.state_dim = 1
        self.action_dim = 1
        self.state = None

    def reset(self):
        self.state = np.random.uniform(-1, 1, size=(1,))
        return self.state

    def step(self, action):
        noise = np.random.normal(0, 0.05, size=(1,))
        next_state = self.state + 0.5 * np.sin(action) + noise
        reward = -np.square(next_state).sum()  # Ziel: zum Ursprung
        done = False
        self.state = next_state
        return next_state, reward, done, {}

    def close(self):
        pass

# --- Reward (wie in reward.py, aber für 1D) ---
def synthetic_reward(x):
    return torch.exp(-0.5 * x.pow(2).sum(-1))

# --- Main Test-Anwendung ---
if __name__ == "__main__":
    env = SyntheticEnv()
    pilco = PILCO(state_dim=1, action_dim=1, policy_feat=5)

    # Ersetze Reward-Funktion für dieses Experiment
    pilco.reward = synthetic_reward

    for iteration in range(10):
        print(f"Iteration {iteration}")

        # 1. Sammle initiale Daten
        print("Initial rollout...")
        pilco.rollout(env, steps=40)

        # 2. Trainiere GP-Model
        print("Trainiere GP...")
        pilco.train_model()

        # 3. Policy-Optimierung
        init_mean = np.zeros(1)
        init_cov = np.eye(1)*0.2
        print("Policy-Optimierung...")
        pilco.optimize_policy(init_mean, init_cov, horizon=20, lr=1e-2, steps=30)

    # 4. Teste neue Policy (Rollout)
    print("Rollout mit neuer Policy:")
    state = env.reset()
    returns = 0
    for t in range(20):
        state_torch = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action = pilco.policy(state_torch).numpy().squeeze()
        state, reward, done, _ = env.step(action)
        returns += reward
        print(f"Step {t}: state={state}, action={action}, reward={reward:.3f}")

    print(f"Gesamter Return im Test: {returns:.2f}")
    env.close()
