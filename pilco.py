import torch
from buffer import DataBuffer
from policy import RBFPolicy
from dynamics_gpytorch import MultiOutputGP
from reward import pilco_reward

class PILCO:
    def __init__(self, state_dim, action_dim, policy_feat=10):
        self.dynamics = MultiOutputGP(state_dim + action_dim, state_dim)
        self.policy = RBFPolicy(state_dim, action_dim, num_features=policy_feat)
        self.data = DataBuffer()

    def expected_return(self, init_mean, init_cov, horizon=10, n_samples=20):
        # Monte-Carlo Erwartungswert aus Anfangsverteilung und aktuellem Modell/Policy
        init_dist = torch.distributions.MultivariateNormal(
            torch.tensor(init_mean, dtype=torch.float32),
            torch.tensor(init_cov, dtype=torch.float32)
        )
        total_return = 0.
        gamma = 0.99
        for _ in range(n_samples):
            state = init_dist.sample()
            traj_return = 0.
            for t in range(horizon):
                action = self.policy(state.unsqueeze(0)).squeeze(0)
                xu = torch.cat([state, action])
                next_state = state + self.dynamics.predict(xu.unsqueeze(0))[0]
                traj_return += gamma ** t * self.reward(next_state.unsqueeze(0))
                state = next_state
            total_return += traj_return
        return (total_return / n_samples).item()

    def rollout(self, env, steps=200):
        state, _ = env.reset()
        for _ in range(steps):
            state_torch = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                action = self.policy(state_torch).numpy().squeeze()
            next_state, reward, terminated, truncated, done = env.step(action)
            self.data.add(state, action, next_state)
            state = next_state
            if done:
                break

    def train_model(self):
        X, U, Y = self.data.get()
        inputs = torch.cat([X, U], dim=1)
        deltas = Y - X
        self.dynamics.train(inputs, deltas)

    def optimize_policy(self, init_mean, init_cov, horizon=100, lr=1e-2, steps=50, n_samples=20):
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        for step in range(steps):
            optimizer.zero_grad()
            # Monte Carlo Sampling aus der Anfangsverteilung
            init_dist = torch.distributions.MultivariateNormal(
                torch.tensor(init_mean, dtype=torch.float32),
                torch.tensor(init_cov, dtype=torch.float32)
            )
            total_return = 0.
            for _ in range(n_samples):
                state = init_dist.sample()
                traj_return = 0.
                gamma = 0.99
                for t in range(horizon):
                    action = self.policy(state.unsqueeze(0)).squeeze(0)
                    xu = torch.cat([state, action])
                    next_state = state + self.dynamics.predict(xu.unsqueeze(0))[0]  # [state_dim]
                    traj_return += gamma**t * self.reward(next_state.unsqueeze(0))
                    state = next_state
                total_return += traj_return
            expected_return = total_return / n_samples
            loss = -expected_return
            loss.backward()
            optimizer.step()
            if step % 10 == 0:
                print(f"Policy Opt Step {step}: Expected Return {expected_return.item():.3f}")
