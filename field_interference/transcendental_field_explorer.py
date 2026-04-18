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
        self.colorbar = None
        
        # Params
        self.max_degree = 3
        self.max_coeff = 5
        self.sigma = 1.0
        self.view_zoom = 1.0
        
        self.fig = plt.figure(figsize=(14, 10), facecolor='#050505')
        self.ax = self.fig.add_axes([0.15, 0.25, 0.75, 0.65], facecolor='black')
        
        if self.interactive:
            self.setup_widgets()
        
        self.update(None)

    def setup_widgets(self):
        # Custom Base Input
        self.ax_text = self.fig.add_axes([0.02, 0.45, 0.1, 0.05], facecolor='#111111')
        self.text_box = TextBox(self.ax_text, 'Custom: ', initial='pi', color='white', hovercolor='#333333')
        self.text_box.on_submit(self.submit_custom_base)
        self.ax_text.set_visible(False)

        # Mode Selection
        ax_mode = self.fig.add_axes([0.02, 0.75, 0.12, 0.15], facecolor='#111111')
        self.radio = RadioButtons(ax_mode, ('Algebraic', 'Finite', 'Transcendental', 'Complexity'), activecolor='#00d4ff')
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

        self.ax_s4 = self.fig.add_axes([0.25, 0.16, 0.5, 0.02], facecolor='#222222')
        self.s4 = Slider(self.ax_s4, 'Zoom ', 0.1, 10.0, valinit=1.0)

        self.ax_s5 = self.fig.add_axes([0.25, 0.20, 0.5, 0.02], facecolor='#222222')
        self.s5 = Slider(self.ax_s5, 'Samples ', 1e4, 1e6, valinit=3e5, valstep=1e4)
        
        self.s1.on_changed(self.update)
        self.s2.on_changed(self.update)
        self.s3.on_changed(self.update)
        self.s4.on_changed(self.update)
        self.s5.on_changed(self.update)

    def change_mode(self, label):
        self.mode = label
        is_trans = self.mode in ['Transcendental', 'Complexity']
        self.ax_base.set_visible(is_trans)
        self.ax_text.set_visible(is_trans)
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

    def submit_custom_base(self, text):
        try:
            # Safe eval for mathematical expressions
            allowed_names = {
                'pi': np.pi, 'e': np.e, 'phi': (1 + 5**0.5) / 2, 'i': 1j,
                'j': 1j, 'exp': np.exp, 'sin': np.sin, 'cos': np.cos,
                'sqrt': np.sqrt, 'log': np.log
            }
            val = eval(text, {"__builtins__": None}, allowed_names)
            self.current_base_name = text
            self.current_base_val = complex(val)
            self.update(None)
        except Exception as e:
            print(f"Error evaluating custom base: {e}")

    def get_convergents(self, x, n=8):
        convergents = []
        if np.iscomplex(x):
            return convergents
        a = int(np.floor(x))
        convergents.append(a)
        rem = x - a
        for _ in range(n):
            if abs(rem) < 1e-10: break
            inv = 1.0 / rem
            a = int(np.floor(inv))
            convergents.append(a)
            rem = inv - a

        # Build p/q
        results = []
        for i in range(1, len(convergents) + 1):
            p, q = 1, 0
            for j in reversed(range(i)):
                p, q = convergents[j] * p + q, p
            results.append(p / q if q != 0 else p)
        return results

    def draw_transcendental(self, deg, coeff_range, sigma, show_complexity=False, zoom=1.0, num_samples=300000):
        """
        Visualizes the 'interference' of a transcendental base using vectorized computation.
        We generate elements of the form: sum_{i=0}^{deg} c_i * base^i
        """
        res = 600
        base = self.current_base_val

        # Vectorized coefficient generation
        coeffs = np.random.randint(-coeff_range, coeff_range + 1, (num_samples, deg + 1))
        powers = base ** np.arange(deg + 1)
        vals = coeffs @ powers

        limit = np.percentile(np.abs(vals), 95) / zoom
        x_range = (-limit, limit)
        y_range = (-limit, limit)
        
        title_base = self.current_base_name.replace('_', '\\_')
        
        if show_complexity:
            complexity = np.sum(np.abs(coeffs), axis=1)
            # Map vals to pixel coords
            ix = ((vals.real - x_range[0]) / (x_range[1] - x_range[0]) * (res - 1)).astype(int)
            iy = ((vals.imag - y_range[0]) / (y_range[1] - y_range[0]) * (res - 1)).astype(int)
            mask = (ix >= 0) & (ix < res) & (iy >= 0) & (iy < res)

            # Efficiently find min complexity per pixel
            Z_comp = np.full((res, res), np.inf)
            np.minimum.at(Z_comp, (iy[mask], ix[mask]), complexity[mask])
            Z_comp[np.isinf(Z_comp)] = np.nan

            im = self.ax.imshow(Z_comp, extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                                origin='lower', cmap='viridis_r')
            if self.colorbar:
                self.colorbar.remove()
            self.colorbar = self.fig.colorbar(im, ax=self.ax, label='Min L1 Coefficient Complexity')
            self.colorbar.ax.yaxis.label.set_color('white')
            self.colorbar.ax.tick_params(colors='white')
            
            self.ax.set_title(rf"Transcendental Complexity: $\mathbb{{Q}}({title_base})$" + "\n" +
                              rf"Simplest representation for each neighborhood",
                              color='white', fontsize=14)
        else:
            if self.colorbar:
                self.colorbar.remove()
                self.colorbar = None
                
            H, xedges, yedges = np.histogram2d(vals.real, vals.imag, bins=res,
                                               range=[[x_range[0], x_range[1]], [y_range[0], y_range[1]]])
            Z_blur = gaussian_filter(H.T, sigma=sigma)
            self.ax.imshow(Z_blur, extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                            origin='lower', cmap='plasma', norm=LogNorm(vmin=1.0))
            self.ax.set_title(rf"Transcendental Field Interference: $\mathbb{{Q}}({title_base})$" + "\n" +
                              rf"Density of elements $\sum_{{i=0}}^{{{deg}}} c_i \cdot {title_base}^i$",
                              color='white', fontsize=14)

        # Plot Rational Convergents if base is real
        if not np.iscomplex(base):
            convs = self.get_convergents(base)
            self.ax.scatter(convs, np.zeros_like(convs), color='red', s=50, marker='x',
                            label='Rational Convergents', zorder=5)
            self.ax.legend(facecolor='#111111', labelcolor='white')

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
        self.ax.set_title(r"Field Interference: Algebraic Number Density in $\mathbb{C}$", color='white', fontsize=15)

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
        v4 = self.s4.val if self.interactive else 1.0
        v5 = int(self.s5.val) if self.interactive else 300000
        
        if self.mode == 'Algebraic':
            self.draw_algebraic(v1, v2, v3)
        elif self.mode == 'Finite':
            self.draw_finite(v1, v2)
        elif self.mode == 'Transcendental':
            self.draw_transcendental(v1, v2, v3, show_complexity=False, zoom=v4, num_samples=v5)
        else: # Complexity
            self.draw_transcendental(v1, v2, v3, show_complexity=True, zoom=v4, num_samples=v5)
            
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
