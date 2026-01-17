import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

def generate_algebraic_numbers(max_degree, max_coeff, resolution=1000):
    """
    Generates a set of algebraic numbers and maps them to a complex plane grid.
    """
    x_range = (-2, 2)
    y_range = (-2, 2)
    Z = np.zeros((resolution, resolution))

    # Iterate through degrees
    for deg in range(1, max_degree + 1):
        # Sample polynomials
        # For low degrees, we can be more exhaustive
        if deg == 1:
            # ax + b = 0 => x = -b/a
            for a in range(-max_coeff, max_coeff + 1):
                if a == 0: continue
                for b in range(-max_coeff, max_coeff + 1):
                    root = -b/a
                    if x_range[0] <= root <= x_range[1]:
                        ix = int((root - x_range[0]) / (x_range[1] - x_range[0]) * (resolution - 1))
                        iy = resolution // 2
                        Z[iy, ix] += 10 # Weight rational points more
        else:
            # Random sampling for higher degrees to show the 'field'
            num_samples = 50000 // deg
            for _ in range(num_samples):
                poly_coeffs = np.random.randint(-max_coeff, max_coeff + 1, deg + 1)
                if poly_coeffs[-1] == 0: poly_coeffs[-1] = 1
                
                roots = np.roots(poly_coeffs[::-1])
                for root in roots:
                    re, im = root.real, root.imag
                    if x_range[0] <= re <= x_range[1] and y_range[0] <= im <= y_range[1]:
                        ix = int((re - x_range[0]) / (x_range[1] - x_range[0]) * (resolution - 1))
                        iy = int((im - y_range[0]) / (y_range[1] - y_range[0]) * (resolution - 1))
                        Z[iy, ix] += 1

    return Z

def visualize_field_interference():
    print("Generating algebraic number distribution...")
    max_degree = 5
    max_coeff = 20
    res = 1000
    
    density = generate_algebraic_numbers(max_degree, max_coeff, resolution=res)
    
    # Apply blur to create the 'interference' field effect
    density_blurred = gaussian_filter(density, sigma=1.2)

    fig, ax = plt.subplots(figsize=(12, 12), facecolor='black')
    
    # Use a more 'interference' like colormap
    im = ax.imshow(density_blurred, extent=[-2, 2, -2, 2], origin='lower', 
                   cmap='inferno', norm=LogNorm(vmin=0.1, vmax=density_blurred.max()))
    
    # Add markings for the 'Base Field' (Rational points)
    for q in range(1, 10):
        for p in range(-2*q, 2*q + 1):
            val = p/q
            if -2 <= val <= 2:
                ax.plot(val, 0, 'wo', markersize=1, alpha=0.3)

    ax.set_title("Field Interference: Algebraic Number Density in $\mathbb{C}$\n"
                 "Resonance between $\mathbb{Q}$ and Higher Polynomial Structures", 
                 color='white', fontsize=18, pad=20)
    ax.set_xlabel("Re(z)", color='white', fontsize=12)
    ax.set_ylabel("Im(z)", color='white', fontsize=12)
    ax.tick_params(colors='white', labelsize=10)
    ax.set_facecolor('black')
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_edgecolor('#444444')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Density of Algebraic Roots (Log Scale)', color='white', fontsize=12)
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')

    plt.tight_layout()
    plt.savefig('./field_interference_v2.png', dpi=300, facecolor='black')
    print("Visualization saved to ./field_interference_v2.png")

if __name__ == "__main__":
    visualize_field_interference()
