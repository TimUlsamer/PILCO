from environment import CartPoleEnv
from pilco import PILCO
import numpy as np

if __name__ == "__main__":
    # -- FÜR VIRTUELLEN CARTPOLE --
    env = CartPoleEnv(mode="sim")
    pilco = PILCO(state_dim=4, action_dim=1, policy_feat=10)

    # 1. Rollout auf virtuellem CartPole (kann auch viele Durchläufe sein)
    pilco.rollout(env, steps=200)

    # 2. GP-Modell trainieren auf gesammelten Daten
    pilco.train_model()

    # 3. Policy-Optimierung
    init_mean = np.zeros(4)
    init_cov = np.eye(4)*0.1
    pilco.optimize_policy(init_mean, init_cov, horizon=100, lr=1e-2, steps=50)

    # 4. Rollout mit neuer Policy
    pilco.rollout(env, steps=200)

    # 5. (Optional) Repeat: Daten werden im Buffer gesammelt und Modell/Policy immer besser

    env.close()
