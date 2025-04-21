import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import Chebyshev

# Define a Chebyshev-based landscape function
coeffs = np.random.uniform(-1, 1, size=5)  # random coefficients for T0..T4
landscape = Chebyshev(coeffs, domain=[0, 1])

# Nelder-Mead optimizer for 1D functions
def nelder_mead_1d(f, x0, delta=0.05, alpha=1, gamma=2, beta=0.5, shrink=0.5, tol=1e-5, max_iter=200):
    # Initialize simplex: two points in 1D
    x = np.array([x0, x0 + delta])
    fvals = f(x)
    for _ in range(max_iter):
        # Order points
        idx = np.argsort(fvals)
        x = x[idx]
        fvals = fvals[idx]
        # Convergence?
        if abs(x[1] - x[0]) < tol:
            break
        # Centroid of best point (just x[0])
        centroid = x[0]
        # Reflection
        xr = centroid + alpha * (centroid - x[1])
        fr = f(np.array([xr]))[0]
        if fvals[0] <= fr < fvals[1]:
            x[1], fvals[1] = xr, fr
            continue
        # Expansion
        if fr < fvals[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = f(np.array([xe]))[0]
            if fe < fr:
                x[1], fvals[1] = xe, fe
            else:
                x[1], fvals[1] = xr, fr
            continue
        # Contraction
        if fr < fvals[1]:
            xc = centroid + beta * (xr - centroid)
        else:
            xc = centroid - beta * (centroid - x[1])
        fc = f(np.array([xc]))[0]
        if fc < min(fr, fvals[1]):
            x[1], fvals[1] = xc, fc
            continue
        # Shrink
        x[1] = x[0] + shrink * (x[1] - x[0])
        fvals[1] = f(np.array([x[1]]))[0]
    return x[0]

# Prepare landscape plot
xs = np.linspace(0, 1, 500)
ys = landscape(xs)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(xs, ys, label='Landscape')
ax.set_xlabel('x')
ax.set_ylabel('Height')
ax.set_title('Chebyshev Landscape and Water Drop Paths')

# Simulate water drops
drops = np.random.uniform(0, 1, size=5)
top_y = max(ys) + 0.2

for xi in drops:
    # vertical drop to landscape
    yi = landscape(xi)
    ax.plot([xi, xi], [top_y, yi], linestyle='--')
    # flow to local minimum via Nelder-Mead
    xmin = nelder_mead_1d(lambda x: landscape(x), xi)
    ymin = landscape(xmin)
    ax.annotate('', xy=(xmin, ymin), xytext=(xi, yi), arrowprops=dict(arrowstyle='->'))
    ax.scatter([xi], [yi], marker='o')
    ax.scatter([xmin], [ymin], marker='x')

ax.set_ylim(bottom=min(ys) - 0.1, top=top_y + 0.1)
ax.legend()
plt.show()
