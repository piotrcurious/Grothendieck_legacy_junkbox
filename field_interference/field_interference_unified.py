import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm

# --- Finite Field Math (Python Implementation) ---

def gf_is_irreducible(poly, p):
    n = len(poly) - 1
    if n < 1: return False
    if n == 1: return True
    for i in range(p):
        val = 0
        x_pow = 1
        for c in poly:
            val = (val + c * x_pow) % p
            x_pow = (x_pow * i) % p
        if val == 0: return False
    if n == 4:
        for a in range(p):
            for b in range(p):
                has_root = False
                for i in range(p):
                    if (i*i + a*i + b) % p == 0:
                        has_root = True; break
                if has_root: continue
                rem = list(poly)
                for i in range(4, 1, -1):
                    factor = rem[i]
                    if factor == 0: continue
                    rem[i-1] = (rem[i-1] - factor * a) % p
                    rem[i-2] = (rem[i-2] - factor * b) % p
                    rem[i] = 0
                if rem[0] == 0 and rem[1] == 0: return False
    return True

def gf_find_irreducible(p, n):
    if n == 1: return [0, 1]
    poly = [0] * (n + 1); poly[n] = 1
    for i in range(1, p**n):
        temp = i
        for j in range(n):
            poly[j] = temp % p; temp //= p
        if gf_is_irreducible(poly, p): return poly
    return poly

def gf_multiply(a, b, g, p):
    n = len(g) - 1; res = [0] * (2 * n)
    for i in range(len(a)):
        for j in range(len(b)):
            res[i+j] = (res[i+j] + a[i] * b[j]) % p
    for i in range(len(res)-1, n-1, -1):
        if res[i] == 0: continue
        factor = res[i]
        for j in range(n + 1):
            res[i-n+j] = (res[i-n+j] - factor * g[j]) % p
    return [x % p for x in res[:n]]

def gf_find_primitive(p, n, g):
    q = p**n; q_minus_1 = q - 1; factors = []; temp = q_minus_1; d = 2
    while d * d <= temp:
        if temp % d == 0:
            factors.append(d)
            while temp % d == 0: temp //= d
        d += 1
    if temp > 1: factors.append(temp)
    for i in range(1, q):
        alpha = [0] * n; temp = i
        for j in range(n):
            alpha[j] = temp % p; temp //= p
        if all(x == 0 for x in alpha): continue
        is_primitive = True
        for f in factors:
            exp = q_minus_1 // f; res = [0] * n; res[0] = 1; base = list(alpha); curr_exp = exp
            while curr_exp > 0:
                if curr_exp % 2 == 1: res = gf_multiply(res, base, g, p)
                base = gf_multiply(base, base, g, p); curr_exp //= 2
            if all(x == (1 if j == 0 else 0) for j, x in enumerate(res)):
                is_primitive = False; break
        if is_primitive: return alpha
    return [0, 1] + [0]*(n-2) if n > 1 else [1]

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
        self.ax_s1 = self.fig.add_axes([0.25, 0.14, 0.5, 0.015], facecolor='#222222')
        self.s1 = Slider(self.ax_s1, 'Degree/p ', 1, 31, valinit=3, valstep=1)
        
        self.ax_s2 = self.fig.add_axes([0.25, 0.11, 0.5, 0.015], facecolor='#222222')
        self.s2 = Slider(self.ax_s2, 'Coeff/n ', 1, 20, valinit=5, valstep=1)
        
        self.ax_s3 = self.fig.add_axes([0.25, 0.08, 0.5, 0.015], facecolor='#222222')
        self.s3 = Slider(self.ax_s3, 'Blur ', 0.1, 3.0, valinit=1.0)

        self.ax_s4 = self.fig.add_axes([0.25, 0.05, 0.5, 0.015], facecolor='#222222')
        self.s4 = Slider(self.ax_s4, 'Samples ', 1e4, 1e6, valinit=3e5, valstep=1e4)
        
        self.s1.on_changed(self.update)
        self.s2.on_changed(self.update)
        self.s3.on_changed(self.update)
        self.s4.on_changed(self.update)

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

    def draw_algebraic(self, deg, coeff, sigma, num_samples):
        res = 600
        all_roots = []
        for d in range(1, deg + 1):
            ns = int(num_samples // (2 * d))
            polys = np.random.randint(-coeff, coeff + 1, (ns, d + 1)).astype(float)
            polys[polys[:, -1] == 0, -1] = 1

            if d == 1:
                all_roots.extend(-polys[:, 0] / polys[:, 1])
            else:
                monic = polys[:, :-1] / polys[:, -1:]
                matrices = np.zeros((ns, d, d))
                matrices[:, 1:, :-1] = np.eye(d - 1)
                matrices[:, :, -1] = -monic
                roots = np.linalg.eigvals(matrices)
                all_roots.extend(roots.flatten())

        all_roots = np.array(all_roots)
        mask = (np.abs(all_roots.real) <= 2.5) & (np.abs(all_roots.imag) <= 2.5)
        roots = all_roots[mask]

        H, _, _ = np.histogram2d(roots.real, roots.imag, bins=res, range=[[-2.5, 2.5], [-2.5, 2.5]])
        Z_blur = gaussian_filter(H.T, sigma=sigma)
        vmin = 0.1
        vmax = max(vmin + 1.0, np.max(Z_blur))
        self.ax.imshow(Z_blur, extent=[-2.5, 2.5, -2.5, 2.5], origin='lower', cmap='magma', norm=LogNorm(vmin=vmin, vmax=vmax))
        self.ax.set_title(r"Field Interference: Algebraic Number Density in $\mathbb{C}$", color='white', fontsize=15)

    def draw_finite(self, p, n):
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
        p = min(primes, key=lambda x:abs(x-p))
        q = p**n
        g = gf_find_irreducible(p, n)
        alpha_gen = gf_find_primitive(p, n, g)
        
        elements = []
        for i in range(q):
            val = 0; temp = i; coeffs = []
            for j in range(n):
                c = temp % p; temp //= p; coeffs.append(c)
                if n == 2: val += c * (1 if j == 0 else 1j)
                else: val += c * np.exp(1j * 2 * np.pi * j / n)
            elements.append((val, coeffs))
        
        re = [e[0].real for e in elements]; im = [e[0].imag for e in elements]
        
        # Additive Lattice
        for i in range(q):
            for j in range(i + 1, q):
                diff_count = 0
                for k in range(n):
                    if elements[i][1][k] != elements[j][1][k]: diff_count += 1
                if diff_count == 1:
                    self.ax.plot([re[i], re[j]], [im[i], im[j]], color='#00d4ff', alpha=0.15, lw=0.8, zorder=1)

        # Multiplicative Orbit
        if q > 1:
            curr = [0] * n; curr[0] = 1; orbit = []
            for _ in range(q):
                val = 0
                for j in range(n):
                    if n == 2: val += curr[j] * (1 if j == 0 else 1j)
                    else: val += curr[j] * np.exp(1j * 2 * np.pi * j / n)
                orbit.append(val); curr = gf_multiply(curr, alpha_gen, g, p)
                if all(x == (1 if j == 0 else 0) for j, x in enumerate(curr)):
                    orbit.append(orbit[0]); break
            self.ax.plot([z.real for z in orbit], [z.imag for z in orbit], color='yellow', alpha=0.6, lw=1.5, zorder=2, label='Multiplicative Orbit')

        self.ax.scatter(re, im, c=np.abs([e[0] for e in elements]), cmap='cool', s=100, edgecolors='white', zorder=3)
        self.ax.scatter(re[:p], im[:p], color='yellow', s=150, edgecolors='red', label=f'Base Field GF({p})', zorder=4)
        
        self.ax.set_title(f"Finite Field Extension: $GF({p}^{n})$ over $GF({p})$", color='white', fontsize=15)
        self.ax.legend(facecolor='#111111', labelcolor='white')
        self.ax.set_aspect('equal')

    def update(self, val):
        self.ax.clear()
        self.ax.set_facecolor('black')
        v1 = int(self.s1.val) if self.interactive else 3
        v2 = int(self.s2.val) if self.interactive else 5
        v3 = self.s3.val if self.interactive else 1.0
        v4 = int(self.s4.val) if self.interactive else 100000
        
        if self.mode == 'Algebraic':
            self.draw_algebraic(v1, v2, v3, v4)
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
