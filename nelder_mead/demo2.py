import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize
from numpy.polynomial.chebyshev import Chebyshev

# Generate a Chebyshev polynomial landscape
def chebyshev_landscape(x, coeffs):
    T = Chebyshev(coeffs, domain=[0, 1])
    return T(x)

# Gradient-free landscape function to minimize (height at given x)
def landscape_func(x, landscape_x, landscape_y):
    x = np.clip(x[0], 0, 1)
    idx = np.searchsorted(landscape_x, x)
    return landscape_y[min(idx, len(landscape_y) - 1)]

# Simulation settings
num_drops = 20
coeffs = np.random.randn(10) * 0.3  # Randomized landscape
landscape_x = np.linspace(0, 1, 1000)
landscape_y = chebyshev_landscape(landscape_x, coeffs)

# Initial drop positions (fall vertically)
drops_x = np.random.rand(num_drops)
drops_y = np.ones(num_drops) * 1.2  # Start above screen

# Status flags
drop_reached_land = [False] * num_drops
drop_target_x = [None] * num_drops
drop_target_y = [None] * num_drops

# Plotting setup
fig, ax = plt.subplots(figsize=(10, 4))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.5)
landscape_line, = ax.plot(landscape_x, landscape_y, 'k', lw=2)
drops_plot, = ax.plot([], [], 'bo', markersize=5)

def init():
    drops_plot.set_data([], [])
    return drops_plot,

def update(frame):
    global drops_x, drops_y, drop_reached_land, drop_target_x, drop_target_y

    new_drops_x = []
    new_drops_y = []

    for i in range(num_drops):
        x = drops_x[i]
        y = drops_y[i]

        # Find ground height at x
        idx = np.searchsorted(landscape_x, x)
        ground_y = landscape_y[min(idx, len(landscape_y) - 1)]

        if not drop_reached_land[i]:
            # Drop is still falling
            y -= 0.02
            if y <= ground_y:
                y = ground_y
                drop_reached_land[i] = True

                # Use Nelder-Mead to minimize landscape height nearby
                res = minimize(landscape_func, [x],
                               args=(landscape_x, landscape_y),
                               method='Nelder-Mead',
                               options={'xatol': 1e-4, 'disp': False})
                x_target = res.x[0]
                y_target = landscape_func([x_target], landscape_x, landscape_y)

                drop_target_x[i] = x_target
                drop_target_y[i] = y_target

        else:
            # Slide toward the minimum using linear interpolation
            tx = drop_target_x[i]
            ty = drop_target_y[i]

            dx = tx - x
            dy = ty - y
            step_size = 0.01
            norm = np.hypot(dx, dy)

            if norm > 1e-3:
                x += (dx / norm) * step_size
                y = landscape_func([x], landscape_x, landscape_y)
            else:
                x = tx
                y = ty

        new_drops_x.append(x)
        new_drops_y.append(y)

    drops_x[:] = new_drops_x
    drops_y[:] = new_drops_y
    drops_plot.set_data(drops_x, drops_y)
    return drops_plot,

# Run animation
ani = FuncAnimation(fig, update, frames=300, init_func=init, blit=True, interval=40)
plt.title("Chebyshev Landscape with Nelder-Mead Sliding Water Drops")
plt.show()
