import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, TextBox
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm

class TranscendentalFieldExplorer:
    def __init__(self, interactive=True):
        self.interactive = interactive
        self.mode = 'Algebraic' # 'Algebraic', 'Finite', 'Transcendental'
        
        # Base values
        self.bases = {
            'pi': np.pi,
            'e': np.e,
            'phi': (1 + 5**0.5) / 2,
            'pi + i': np.pi + 1j,
            'e^i': np.exp(1j),
            'i': 1j
        }
        self.current_base_val = np.pi
        self.current_base_name = 'pi'
        
        # Params
        self.max_degree = 3
        self.max_coeff = 5
        self.sigma = 1.0
        
        self.fig = plt.figure(figsize=(14, 10), facecolor='#050505')
        self.ax = self.fig.add_axes([0.15, 0.25, 0.75, 0.65], facecolor='black')
        
        if self.interactive:
            self.setup_widgets()
        
        self.update(None)

    def setup_widgets(self):
        # Mode Selection
        ax_mode = self.fig.add_axes([0.02, 0.8, 0.1, 0.1], facecolor='#111111')
        self.radio = RadioButtons(ax_mode, ('Algebraic', 'Finite', 'Transcendental'), activecolor='#00d4ff')
        for label in self.radio.labels:
            label.set_color('white')
        self.radio.on_clicked(self.change_mode)
        
        # Base Selection (for Transcendental mode)
        self.ax_base = self.fig.add_axes([0.02, 0.6, 0.1, 0.15], facecolor='#111111')
        self.base_radio = RadioButtons(self.ax_base, list(self.bases.keys()), activecolor='#ff007f')
        for label in self.base_radio.labels:
            label.set_color('white')
        self.base_radio.on_clicked(self.change_base)
        self.ax_base.set_visible(False)
        
        # Sliders
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
        self.ax_base.set_visible(self.mode == 'Transcendental')
        if self.mode == 'Algebraic':
            self.s1.label.set_text('Max Degree ')
            self.s2.label.set_text('Max Coeff ')
            self.ax_s3.set_visible(True)
        elif self.mode == 'Finite':
            self.s1.label.set_text('Prime p ')
            self.s2.label.set_text('Degree n ')
            self.ax_s3.set_visible(False)
        else: # Transcendental
            self.s1.label.set_text('Extension Deg ')
            self.s2.label.set_text('Coeff Range ')
            self.ax_s3.set_visible(True)
        self.update(None)

    def change_base(self, label):
        self.current_base_name = label
        self.current_base_val = self.bases[label]
        self.update(None)

    def draw_transcendental(self, deg, coeff_range, sigma):
        """
        Visualizes the 'interference' of a transcendental base.
        We generate elements of the form: sum_{i=0}^{deg} c_i * base^i
        """
        res = 500
        Z = np.zeros((res, res))
        # Adjust range based on base magnitude
        mag = np.abs(self.current_base_val)
        limit = (mag ** deg) * coeff_range * 0.5
        x_range = (-limit, limit)
        y_range = (-limit, limit)
        
        num_samples = 50000
        base = self.current_base_val
        
        for _ in range(num_samples):
            coeffs = np.random.randint(-coeff_range, coeff_range + 1, deg + 1)
            val = np.polyval(coeffs, base)
            
            re, im = val.real, val.imag
            if x_range[0] <= re <= x_range[1] and y_range[0] <= im <= y_range[1]:
                ix = int((re - x_range[0]) / (x_range[1] - x_range[0]) * (res - 1))
                iy = int((im - y_range[0]) / (y_range[1] - y_range[0]) * (res - 1))
                Z[iy, ix] += 1
                
        Z_blur = gaussian_filter(Z, sigma=sigma)
        self.ax.imshow(Z_blur, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], 
                        origin='lower', cmap='plasma', norm=LogNorm(vmin=0.1))
        self.ax.set_title(f"Transcendental Field Interference: $\mathbb{{Q}}({self.current_base_name})$\n"
                          f"Structure of Polynomials in {self.current_base_name}", color='white', fontsize=14)

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
        primes = [2, 3, 5, 7, 11, 13, 17, 19]
        p = min(primes, key=lambda x:abs(x-p))
        elements = []
        for i in range(p**n):
            val = 0
            temp = i
            for j in range(n):
                c = temp % p
                temp //= p
                if n == 2: val += c * (1 if j == 0 else 1j)
                else: val += c * np.exp(1j * 2 * np.pi * j / n)
            elements.append(val)
        re, im = [e.real for e in elements], [e.imag for e in elements]
        self.ax.scatter(re, im, c=np.abs(elements), cmap='cool', s=100, edgecolors='white', zorder=3)
        self.ax.set_title(f"Finite Field Extension: $GF({p}^{n})$", color='white', fontsize=15)
        self.ax.set_aspect('equal')

    def update(self, val):
        self.ax.clear()
        self.ax.set_facecolor('black')
        v1 = int(self.s1.val) if self.interactive else 3
        v2 = int(self.s2.val) if self.interactive else 5
        v3 = self.s3.val if self.interactive else 1.0
        
        if self.mode == 'Algebraic':
            self.draw_algebraic(v1, v2, v3)
        elif self.mode == 'Finite':
            self.draw_finite(v1, v2)
        else:
            self.draw_transcendental(v1, v2, v3)
            
        self.ax.tick_params(colors='white')
        if self.interactive:
            self.fig.canvas.draw_idle()

if __name__ == "__main__":
    import sys
    if "--static" in sys.argv:
        explorer = TranscendentalFieldExplorer(interactive=False)
        explorer.mode = 'Transcendental'
        explorer.current_base_name = 'pi'
        explorer.current_base_val = np.pi
        explorer.update(None)
        plt.savefig('./transcendental_interference_static.png', dpi=300, facecolor='#050505')
    else:
        explorer = TranscendentalFieldExplorer(interactive=True)
        plt.show()
