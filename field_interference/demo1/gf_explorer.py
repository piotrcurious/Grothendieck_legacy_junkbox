import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import sympy as sp

class GFExplorer:
    def __init__(self, interactive=True):
        self.p = 3
        self.n = 2
        self.interactive = interactive
        self.fig, self.ax = plt.subplots(figsize=(12, 10), facecolor='#0a0a0a')
        
        if self.interactive:
            plt.subplots_adjust(bottom=0.2)
            ax_p = plt.axes([0.25, 0.1, 0.5, 0.03], facecolor='#222222')
            self.slider_p = Slider(ax_p, 'Prime p ', 2, 11, valinit=3, valstep=[2, 3, 5, 7, 11])
            
            ax_n = plt.axes([0.25, 0.05, 0.5, 0.03], facecolor='#222222')
            self.slider_n = Slider(ax_n, 'Degree n ', 1, 3, valinit=2, valstep=1)
            
            self.slider_p.on_changed(self.update)
            self.slider_n.on_changed(self.update)
        
        self.update(None)

    def get_field_elements(self, p, n):
        """
        Represent GF(p^n) as a vector space over GF(p).
        We map the basis {1, α, α^2, ..., α^(n-1)} to symmetric directions in the plane.
        """
        elements = []
        labels = []
        
        # Generate all p^n combinations of coefficients
        for i in range(p**n):
            coeffs = []
            temp = i
            for _ in range(n):
                coeffs.append(temp % p)
                temp //= p
            
            # Map to 2D plane using a clear vector space basis
            val = 0
            label_parts = []
            for j, c in enumerate(coeffs):
                if n == 2:
                    # For degree 2, use 1 and i as basis for a clear grid
                    basis_vec = complex(1, 0) if j == 0 else complex(0, 1)
                    val += c * basis_vec
                elif n > 2:
                    # For higher degrees, use roots of unity
                    angle = 2 * np.pi * j / n
                    val += c * np.exp(1j * angle)
                else:
                    val = complex(c, 0)
                
                if c != 0:
                    if j == 0: label_parts.append(f"{c}")
                    elif j == 1: label_parts.append(f"{c if c!=1 else ''}α")
                    else: label_parts.append(f"{c if c!=1 else ''}α^{j}")
            
            elements.append(val)
            labels.append(" + ".join(label_parts) if label_parts else "0")
            
        return np.array(elements), labels

    def update(self, val):
        p = int(self.slider_p.val) if self.interactive else self.p
        n = int(self.slider_n.val) if self.interactive else self.n
        
        self.ax.clear()
        self.ax.set_facecolor('#0a0a0a')
        
        elements, labels = self.get_field_elements(p, n)
        
        # Plotting logic
        re = [e.real for e in elements]
        im = [e.imag for e in elements]
        
        # Draw the "Interference" lattice (additive structure)
        # We connect elements that differ by a single basis vector
        for i in range(len(elements)):
            for j in range(i + 1, len(elements)):
                diff = elements[i] - elements[j]
                # If the difference is a basis vector (magnitude 1)
                if np.isclose(np.abs(diff), 1.0, atol=0.1):
                    self.ax.plot([elements[i].real, elements[j].real], 
                                 [elements[i].imag, elements[j].imag], 
                                 color='#00d4ff', alpha=0.15, lw=0.8, zorder=1)

        # Scatter plot
        # Color by "norm" or distance from origin to show structure
        norms = np.abs(elements)
        scatter = self.ax.scatter(re, im, c=norms, cmap='cool', s=120, 
                                  edgecolors='white', linewidth=0.5, zorder=3)
        
        # Highlight the Base Field GF(p)
        base_indices = [i for i in range(p)]
        self.ax.scatter([re[i] for i in base_indices], [im[i] for i in base_indices], 
                        color='yellow', s=180, edgecolors='red', linewidth=1.5, 
                        zorder=4, label=f'Base Field GF({p})')
        
        # Annotations
        for i, txt in enumerate(labels):
            self.ax.annotate(txt, (re[i], im[i]), xytext=(4, 4), 
                             textcoords='offset points', color='white', 
                             fontsize=9, alpha=0.8, fontweight='bold')

        self.ax.set_title(f"Finite Field Interference: $GF({p}^{n})$\n"
                          f"Structural Coupling of $GF({p})$ and Degree {n} Extension", 
                          color='white', fontsize=16, pad=20)
        
        self.ax.legend(facecolor='#1a1a1a', edgecolor='white', labelcolor='white', loc='upper right')
        self.ax.grid(True, color='#222222', linestyle=':', alpha=0.5)
        
        # Adjust limits
        max_dim = np.max(np.abs(np.concatenate([re, im]))) + 1
        self.ax.set_xlim(-max_dim, max_dim)
        self.ax.set_ylim(-max_dim, max_dim)
        self.ax.set_aspect('equal')
        
        if self.interactive:
            self.fig.canvas.draw_idle()

if __name__ == "__main__":
    import sys
    if "--static" in sys.argv:
        explorer = GFExplorer(interactive=False)
        explorer.p = 5
        explorer.n = 2
        explorer.update(None)
        plt.savefig('/home/ubuntu/gf_extension_visualization.png', dpi=300, facecolor='#0a0a0a')
        print("Static visualization saved to /home/ubuntu/gf_extension_visualization.png")
    else:
        explorer = GFExplorer(interactive=True)
        plt.show()
