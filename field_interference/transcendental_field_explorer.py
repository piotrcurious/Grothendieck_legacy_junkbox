import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, TextBox
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm

def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0: return False
    return True

def gf_inv(n, p): return pow(n % p, p - 2, p)

def gf_poly_mod(a, g, p):
    n = len(g) - 1; inv_lead = gf_inv(g[-1], p)
    a = list(a)
    while len(a) > n:
        f = (a[-1] * inv_lead) % p; d = len(a) - 1 - n
        for i in range(len(g)): a[i+d] = (a[i+d] - f * g[i]) % p
        while len(a) > 0 and a[-1] == 0: a.pop()
    return [x % p for x in a] if a else [0]

def gf_multiply(a, b, g, p):
    if not a or not b or (len(a)==1 and a[0]==0) or (len(b)==1 and b[0]==0): return [0]
    res = [0] * (len(a) + len(b) - 1)
    for i, va in enumerate(a):
        for j, vb in enumerate(b): res[i+j] = (res[i+j] + va * vb) % p
    return gf_poly_mod(res, g, p)

def gf_poly_gcd(a, b, p):
    while len(b) > 1 or (len(b)==1 and b[0] != 0): a, b = b, gf_poly_mod(a, b, p)
    if a:
        inv = gf_inv(a[-1], p); a = [(x * inv) % p for x in a]
    return a

def gf_is_irreducible(poly, p):
    n = len(poly) - 1
    if n < 1: return False
    if n == 1: return True
    if not is_prime(p): return False
    rem_x_pow = [0, 1]
    for i in range(1, n // 2 + 1):
        next_rem = [1]; base = list(rem_x_pow); exp = p
        while exp > 0:
            if exp % 2 == 1: next_rem = gf_multiply(next_rem, base, poly, p)
            base = gf_multiply(base, base, poly, p); exp //= 2
        rem_x_pow = next_rem
        diff = list(rem_x_pow)
        if len(diff) < 2: diff = diff + [0] * (2 - len(diff))
        diff[1] = (diff[1] - 1) % p
        if len(gf_poly_gcd(poly, diff, p)) > 1: return False
    return True

def gf_find_irreducible(p, n):
    if n == 1: return [0, 1]
    poly = [0] * (n + 1); poly[n] = 1
    for _ in range(1000):
        for j in range(n): poly[j] = np.random.randint(0, p)
        if gf_is_irreducible(poly, p): return poly
    return poly

class TranscendentalFieldExplorer:
    def __init__(self, interactive=True):
        self.interactive = interactive
        self.mode = 'Algebraic'
        self.bases = {
            'pi': np.pi, 'e': np.e, 'phi': (1 + 5**0.5) / 2,
            'pi + i': np.pi + 1j, 'e^i': np.exp(1j), 'i': 1j
        }
        self.current_base_val = np.pi
        self.current_base_name = 'pi'
        self.fig = plt.figure(figsize=(14, 10), facecolor='#050505')
        self.ax = self.fig.add_axes([0.15, 0.25, 0.75, 0.65], facecolor='black')
        if self.interactive: self.setup_widgets()
        self.update(None)

    def setup_widgets(self):
        ax_mode = self.fig.add_axes([0.02, 0.85, 0.12, 0.1], facecolor='#111111')
        self.radio = RadioButtons(ax_mode, ('Algebraic', 'Finite', 'Transcendental'), activecolor='#00d4ff')
        for label in self.radio.labels: label.set_color('white')
        self.radio.on_clicked(self.change_mode)

        ax_map = self.fig.add_axes([0.02, 0.7, 0.12, 0.12], facecolor='#111111')
        self.map_radio = RadioButtons(ax_map, ('Standard', 'Log-Polar', 'Mobius', 'Joukowsky', 'Reciprocal'), activecolor='#00ff88')
        for label in self.map_radio.labels: label.set_color('white')
        self.map_radio.on_clicked(self.update)

        self.ax_base = self.fig.add_axes([0.02, 0.5, 0.12, 0.15], facecolor='#111111')
        self.base_radio = RadioButtons(self.ax_base, list(self.bases.keys()), activecolor='#ff007f')
        for label in self.base_radio.labels: label.set_color('white')
        self.base_radio.on_clicked(self.change_base)
        self.ax_base.set_visible(False)

        self.ax_s1 = self.fig.add_axes([0.25, 0.12, 0.5, 0.02], facecolor='#222222')
        self.s1 = Slider(self.ax_s1, 'Degree/p ', 1, 11, valinit=3, valstep=1)
        self.ax_s2 = self.fig.add_axes([0.25, 0.08, 0.5, 0.02], facecolor='#222222')
        self.s2 = Slider(self.ax_s2, 'Coeff/n ', 1, 20, valinit=5, valstep=1)
        self.ax_s3 = self.fig.add_axes([0.25, 0.04, 0.5, 0.02], facecolor='#222222')
        self.s3 = Slider(self.ax_s3, 'Blur ', 0.1, 3.0, valinit=1.0)
        for s in [self.s1, self.s2, self.s3]: s.on_changed(self.update)

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
        else:
            self.s1.label.set_text('Extension Deg ')
            self.s2.label.set_text('Coeff Range ')
            self.ax_s3.set_visible(True)
        self.update(None)

    def apply_mapping(self, z, mapping):
        if mapping == 'Log-Polar':
            return np.log(np.abs(z) + 1e-15) + 1j * np.angle(z)
        elif mapping == 'Mobius':
            return (z - 1j) / (z + 1j + 1e-15)
        elif mapping == 'Joukowsky':
            return z + 1.0 / (z + 1e-15)
        elif mapping == 'Reciprocal':
            return 1.0 / (z + 1e-15)
        return z

    def change_base(self, label):
        self.current_base_name = label
        self.current_base_val = self.bases[label]
        self.update(None)

    def draw_transcendental(self, deg, coeff_range, sigma, mapping):
        res = 500; Z = np.zeros((res, res))
        limit = (np.abs(self.current_base_val) ** deg) * coeff_range * 0.5
        x_range = (-limit, limit); y_range = (-limit, limit)
        num_samples = 100000; base = self.current_base_val

        # Vectorized generation
        coeffs = np.random.randint(-coeff_range, coeff_range + 1, (num_samples, deg + 1))
        powers = base ** np.arange(deg + 1)
        vals = coeffs @ powers
        mvals = self.apply_mapping(vals, mapping)

        mask = (mvals.real >= x_range[0]) & (mvals.real <= x_range[1]) & \
               (mvals.imag >= y_range[0]) & (mvals.imag <= y_range[1])
        valid_vals = mvals[mask]

        ix = ((valid_vals.real - x_range[0]) / (x_range[1] - x_range[0]) * (res - 1)).astype(int)
        iy = ((valid_vals.imag - y_range[0]) / (y_range[1] - y_range[0]) * (res - 1)).astype(int)
        np.add.at(Z, (iy, ix), 1)

        Z_blur = gaussian_filter(Z, sigma=sigma)
        self.ax.imshow(Z_blur, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], origin='lower', cmap='plasma', norm=LogNorm(vmin=0.1))
        self.ax.set_title(rf"Transcendental Field Interference ({mapping}): $\mathbb{{Q}}({self.current_base_name})$", color='white')

    def draw_algebraic(self, deg, coeff, sigma, mapping):
        res = 400; Z = np.zeros((res, res))
        for d in range(1, deg + 1):
            num_samples = 20000 // d
            all_roots = []
            for _ in range(num_samples):
                poly = np.random.randint(-coeff, coeff + 1, d + 1)
                if poly[-1] == 0: poly[-1] = 1
                all_roots.extend(np.roots(poly[::-1]))

            roots = np.array(all_roots)
            mrs = self.apply_mapping(roots, mapping)
            mask = (mrs.real >= -2) & (mrs.real <= 2) & (mrs.imag >= -2) & (mrs.imag <= 2)
            valid = mrs[mask]
            ix = ((valid.real + 2) / 4 * (res - 1)).astype(int)
            iy = ((valid.imag + 2) / 4 * (res - 1)).astype(int)
            np.add.at(Z, (iy, ix), 1)

        Z_blur = gaussian_filter(Z, sigma=sigma)
        self.ax.imshow(Z_blur, extent=[-2, 2, -2, 2], origin='lower', cmap='magma', norm=LogNorm(vmin=0.1))
        self.ax.set_title(f"Field Interference ({mapping}): Algebraic Number Density", color='white')

    def draw_finite(self, p, n, mapping):
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]; p = min(primes, key=lambda x:abs(x-p))
        elements = []
        for i in range(min(p**n, 5000)):
            v = 0; temp = i
            for j in range(n):
                c = temp % p; temp //= p
                if n == 2:
                    v += c * (1 if j==0 else 1j)
                else:
                    v += c * (np.exp(1j * 2 * np.pi * j / n))
            elements.append(self.apply_mapping(v, mapping))
        re, im = [e.real for e in elements], [e.imag for e in elements]
        self.ax.scatter(re, im, c=np.abs(elements), cmap='cool', s=100, edgecolors='white', zorder=3)
        self.ax.set_title(f"Finite Field Extension ({mapping}): $GF({p}^{n})$", color='white'); self.ax.set_aspect('equal')

    def update(self, val):
        self.ax.clear(); self.ax.set_facecolor('black')
        v1 = int(self.s1.val) if self.interactive else 3
        v2 = int(self.s2.val) if self.interactive else 5
        v3 = self.s3.val if self.interactive else 1.0
        mapping = self.map_radio.val if self.interactive else 'Standard'
        if self.mode == 'Algebraic': self.draw_algebraic(v1, v2, v3, mapping)
        elif self.mode == 'Finite': self.draw_finite(v1, v2, mapping)
        else: self.draw_transcendental(v1, v2, v3, mapping)
        self.ax.tick_params(colors='white')
        if self.interactive: self.fig.canvas.draw_idle()

if __name__ == "__main__":
    import sys
    explorer = TranscendentalFieldExplorer(interactive="--static" not in sys.argv)
    if "--static" in sys.argv: plt.savefig('./transcendental_interference_static.png', dpi=300, facecolor='#050505')
    else: plt.show()
