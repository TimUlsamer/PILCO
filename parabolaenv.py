import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt


class RollingBallParabolaEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        self.a = 1.0      # Parabel y = a * x^2
        self.g = 9.81
        self.dt = 0.02

        self.state = None  # [x, v]
        self.t = 0.0

        # Beobachtungs- und Aktionsraum
        high = np.array([1.5, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)

        # Rendering
        self.fig, self.ax = None, None
        self.ball_dot, self.trail_line = None, None
        self.x_traj, self.y_traj = [], []

    def mass_eff(self, x):
        return 1 + (2/5)*(1 + 4 * self.a**2 * x**2)

    def dynamics(self, x, v, u):
        M = self.mass_eff(x)
        a_int = -(16/5)*self.a**2 * x * v**2 / M - (2*self.a*self.g * x) / M
        return a_int + u / M

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        x0 = self.np_random.uniform(low=0.3, high=0.6)
        v0 = self.np_random.uniform(low=-2.0, high=-0.5)
        self.state = np.array([x0, v0], dtype=np.float32)
        self.t = 0.0
        self.x_traj, self.y_traj = [x0], [self.a * x0**2]
        return self.state.copy(), {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)[0]
        x, v = self.state
        a = self.dynamics(x, v, action)
        v += self.dt * a
        x += self.dt * v
        self.state = np.array([x, v], dtype=np.float32)
        self.t += self.dt

        self.x_traj.append(x)
        self.y_traj.append(self.a * x**2)

        terminated = abs(x) > 1.5 or self.t > 10.0
        reward = -x**2
        return self.state.copy(), reward, terminated, False, {}

    def render(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            x_range = np.linspace(-1.5, 1.5, 400)
            self.ax.plot(x_range, self.a * x_range**2, '--', color='gray')
            self.ball_dot, = self.ax.plot([], [], 'ro')
            self.trail_line, = self.ax.plot([], [], '-', color='blue')
            self.ax.set_xlim(-1.5, 1.5)
            self.ax.set_ylim(0, 2.5)
            self.ax.set_title("Ball in Parabola")
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("y")
            self.ax.grid(True)

        self.ball_dot.set_data([self.state[0]], [self.a * self.state[0]**2])
        self.trail_line.set_data(self.x_traj, self.y_traj)
        self.trail_line.set_alpha(0.5)
        plt.pause(0.001)

    def close(self):
        if self.fig:
            plt.close(self.fig)
