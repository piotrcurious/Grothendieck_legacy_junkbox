import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm

class UnifiedFieldExplorer:
    def __init__(self, interactive=True):
        self.interactive = interactive
        self.mode = 'Algebraic' # 'Algebraic' or 'Finite'
        
        # Algebraic Params
        self.max_degree = 3
        self.max_coeff = 5
        self.sigma = 1.0
        
        # Finite Params
        self.p = 3
        self.n = 2
        
        self.fig = plt.figure(figsize=(14, 10), facecolor='#050505')
        self.ax = self.fig.add_axes([0.1, 0.25, 0.8, 0.65], facecolor='black')
        
        if self.interactive:
            self.setup_widgets()
        
        self.update(None)

    def setup_widgets(self):
        # Mode Selection
        ax_mode = self.fig.add_axes([0.02, 0.8, 0.1, 0.1], facecolor='#111111')
        self.radio = RadioButtons(ax_mode, ('Algebraic', 'Finite'), activecolor='#00d4ff')
        for label in self.radio.labels:
            label.set_color('white')
        self.radio.on_clicked(self.change_mode)
        
        # Sliders (shared space, visibility toggled by mode)
        self.ax_s1 = self.fig.add_axes([0.25, 0.12, 0.5, 0.02], facecolor='#222222')
        self.s1 = Slider(self.ax_s1, 'Degree/p ', 1, 11, valinit=3, valstep=1)
        
        self.ax_s2 = self.fig.add_axes([0.25, 0.08, 0.5, 0.02], facecolor='#222222')
        self.s2 = Slider(self.ax_s2, 'Coeff/n ', 1, 20, valinit=5, valstep=1)
        
        self.ax_s3 = self.fig.add_axes([0.25, 0.04, 0.5, 0.02], facecolor='#222222')
        self.s3 = Slider(self.ax_s3, 'Blur ', 0.1, 3.0, valinit=1.0)
        
        self.s1.on_changed(self.update)
        self.s2.on_changed(self.update)
        self.s3.on_changed(self.update)

    def change_mode(self, label):
        self.mode = label
        if self.mode == 'Algebraic':
            self.s1.label.set_text('Max Degree ')
            self.s2.label.set_text('Max Coeff ')
            self.ax_s3.set_visible(True)
        else:
            self.s1.label.set_text('Prime p ')
            self.s2.label.set_text('Degree n ')
            self.ax_s3.set_visible(False)
        self.update(None)

    def draw_algebraic(self, deg, coeff, sigma):
        res = 400
        Z = np.zeros((res, res))
        x_range = (-2, 2)
        for d in range(1, deg + 1):
            num_samples = 10000 // d
            for _ in range(num_samples):
                poly = np.random.randint(-coeff, coeff + 1, d + 1)
                if poly[-1] == 0: poly[-1] = 1
                roots = np.roots(poly[::-1])
                for r in roots:
                    re, im = r.real, r.imag
                    if x_range[0] <= re <= x_range[1] and x_range[0] <= im <= x_range[1]:
                        ix = int((re - x_range[0]) / 4 * (res - 1))
                        iy = int((im - x_range[0]) / 4 * (res - 1))
                        Z[iy, ix] += 1
        Z_blur = gaussian_filter(Z, sigma=sigma)
        self.ax.imshow(Z_blur, extent=[-2, 2, -2, 2], origin='lower', cmap='magma', norm=LogNorm(vmin=0.1))
        self.ax.set_title("Field Interference: Algebraic Number Density in $\mathbb{C}$", color='white', fontsize=15)

    def draw_finite(self, p, n):
        # Ensure p is prime (simple check for visualization)
        primes = [2, 3, 5, 7, 11, 13, 17, 19]
        p = min(primes, key=lambda x:abs(x-p))
        
        elements = []
        for i in range(p**n):
            val = 0
            temp = i
            for j in range(n):
                c = temp % p
                temp //= p
                if n == 2:
                    val += c * (1 if j == 0 else 1j)
                else:
                    val += c * np.exp(1j * 2 * np.pi * j / n)
            elements.append(val)
        
        re = [e.real for e in elements]
        im = [e.imag for e in elements]
        self.ax.scatter(re, im, c=np.abs(elements), cmap='cool', s=100, edgecolors='white', zorder=3)
        self.ax.scatter(re[:p], im[:p], color='yellow', s=150, edgecolors='red', label=f'Base Field GF({p})')
        
        # Lattice lines
        for i in range(len(elements)):
            for j in range(i + 1, len(elements)):
                if np.isclose(np.abs(elements[i] - elements[j]), 1.0, atol=0.1):
                    self.ax.plot([elements[i].real, elements[j].real], [elements[i].imag, elements[j].imag], 
                                 color='#00d4ff', alpha=0.2, lw=0.5)
        
        self.ax.set_title(f"Finite Field Extension: $GF({p}^{n})$ over $GF({p})$", color='white', fontsize=15)
        self.ax.legend(facecolor='#111111', labelcolor='white')
        self.ax.set_aspect('equal')

    def update(self, val):
        self.ax.clear()
        self.ax.set_facecolor('black')
        v1 = int(self.s1.val) if self.interactive else 3
        v2 = int(self.s2.val) if self.interactive else 5
        v3 = self.s3.val if self.interactive else 1.0
        
        if self.mode == 'Algebraic':
            self.draw_algebraic(v1, v2, v3)
        else:
            self.draw_finite(v1, v2)
            
        self.ax.tick_params(colors='white')
        if self.interactive:
            self.fig.canvas.draw_idle()

if __name__ == "__main__":
    import sys
    if "--static" in sys.argv:
        explorer = UnifiedFieldExplorer(interactive=False)
        plt.savefig('./unified_field_interference.png', dpi=300, facecolor='#050505')
    else:
        explorer = UnifiedFieldExplorer(interactive=True)
        plt.show()
