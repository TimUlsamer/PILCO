import numpy as np
import torch
import random
from pilco import PILCO

# --- Setze festen Seed für Reproduzierbarkeit ---
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

class SyntheticEnv:
    def __init__(self):
        self.state_dim = 1
        self.action_dim = 1
        self.state = None

    def reset(self):
        self.state = np.random.uniform(-1, 1, size=(1,))
        return self.state

    def step(self, action):
        # Weniger Noise!
        noise = np.random.normal(0, 0.01, size=(1,))
        next_state = self.state + 0.5 * np.sin(action) + noise
        reward = -np.square(next_state).sum()
        done = False
        self.state = next_state
        return next_state, reward, done, {}

    def close(self):
        pass

def synthetic_reward(x):
    return torch.exp(-0.5 * x.pow(2).sum(-1))

def test_pilco_learns():
    env = SyntheticEnv()
    pilco = PILCO(state_dim=1, action_dim=1, policy_feat=15)
    pilco.reward = synthetic_reward

    init_mean = np.zeros(1)
    init_cov = np.eye(1)*0.2

    # --- Vor Training: Rollout mit Random Policy ---
    pilco.rollout(env, steps=100)
    before_train_states = []
    state = env.reset()
    for _ in range(10):   # Kürzerer Test-Rollout!
        state_torch = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action = pilco.policy(state_torch).numpy().squeeze()
        state, reward, done, _ = env.step(action)
        before_train_states.append(abs(state[0]))
    initial_mean_abs_state = np.mean(before_train_states)

    # --- Trainingsschleife ---
    for iteration in range(10):
        pilco.train_model()
        pilco.optimize_policy(init_mean, init_cov, horizon=10, lr=1e-2, steps=25)
        pilco.rollout(env, steps=100)

    # --- Nach Training: Rollout mit gelernter Policy ---
    after_train_states = []
    state = env.reset()
    for _ in range(10):  # Gleich wie oben
        state_torch = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action = pilco.policy(state_torch).numpy().squeeze()
        state, reward, done, _ = env.step(action)
        after_train_states.append(abs(state[0]))
    final_mean_abs_state = np.mean(after_train_states)

    print(f"Vor Training: mittlerer |state| = {initial_mean_abs_state:.3f}")
    print(f"Nach Training: mittlerer |state| = {final_mean_abs_state:.3f}")
    assert final_mean_abs_state < initial_mean_abs_state * 0.95, "PILCO hat nicht erkennbar gelernt!"

def test_policy_changes_over_time():
    env = SyntheticEnv()
    pilco = PILCO(state_dim=1, action_dim=1, policy_feat=15)
    pilco.reward = synthetic_reward

    init_mean = np.zeros(1)
    init_cov = np.eye(1)*0.2

    # Initial Policy Kopie
    initial_params = [p.clone().detach() for p in pilco.policy.parameters()]

    for iteration in range(7):
        pilco.rollout(env, steps=100)
        pilco.train_model()
        pilco.optimize_policy(init_mean, init_cov, horizon=10, lr=1e-2, steps=25)

    trained_params = [p.clone().detach() for p in pilco.policy.parameters()]
    diffs = [torch.norm(tp - ip).item() for tp, ip in zip(trained_params, initial_params)]
    print(f"Policy Parameter Änderungen: {diffs}")
    assert any(d > 1e-3 for d in diffs), "Policy hat sich über Training nicht relevant verändert!"

def test_expected_return_improves():
    env = SyntheticEnv()
    pilco = PILCO(state_dim=1, action_dim=1, policy_feat=15)
    pilco.reward = synthetic_reward

    init_mean = np.zeros(1)
    init_cov = np.eye(1)*0.2

    pilco.rollout(env, steps=100)
    pilco.train_model()
    before = pilco.expected_return(init_mean, init_cov, horizon=10)

    for iteration in range(7):
        pilco.optimize_policy(init_mean, init_cov, horizon=10, lr=1e-2, steps=25)
        pilco.rollout(env, steps=100)
        pilco.train_model()

    after = pilco.expected_return(init_mean, init_cov, horizon=10)
    print(f"Expected Return vor Training: {before:.2f} nach Training: {after:.2f}")
    assert after > before, "Expected Return steigt nicht über Training!"

if __name__ == "__main__":
    print("Starte PILCO Synthetic-Tests (robust)...")
    test_pilco_learns()
    test_policy_changes_over_time()
    test_expected_return_improves()
    print("Alle Tests erfolgreich!")
