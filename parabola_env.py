import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# --- Parameters ---
a, g = 1.0, 9.81

def mass_eff(x):
    return 1 + (2/5)*(1 + 4*a**2*x**2)

def rolling_ball(t, z):
    x, v = z
    M = mass_eff(x)
    dv = -(16/5)*a**2 * x * v**2 / M - (2*a*g * x) / M
    return [v, dv]

# Initial conditions
z0 = [0.3, -1.5]
t_span = (0, 10)
t_eval = np.linspace(*t_span, 500)

sol = solve_ivp(rolling_ball, t_span, z0, t_eval=t_eval, rtol=1e-8, atol=1e-8)
x_traj, y_traj = sol.y[0], a * sol.y[0]**2

# --- Plot setup ---
fig, ax = plt.subplots()
x_full = np.linspace(-0.6, 0.6, 400)
ax.plot(x_full, a*x_full**2, '--', color='gray', label='Parabola')

ball_dot, = ax.plot([], [], 'ro')
trail_line, = ax.plot([], [], '-', color='blue', linewidth=2)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.axis('equal')
ax.set_title('Animated Rolling Ball on Parabola')
ax.grid(True)
ax.set_xlim(-0.6, 0.6)
ax.set_ylim(0, 0.4)

# --- Animation function ---
def animate(i):
    ball_dot.set_data([x_traj[i]], [y_traj[i]])  # FIX: wrap in list
    trail_line.set_data(x_traj[:i+1], y_traj[:i+1])
    trail_line.set_alpha(0.3 + 0.7 * (i / len(x_traj)))  # optional fading effect
    return ball_dot, trail_line

anim = FuncAnimation(fig, animate, frames=len(t_eval), interval=20, blit=True)

plt.show()
