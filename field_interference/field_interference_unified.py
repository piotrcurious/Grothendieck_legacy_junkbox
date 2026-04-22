import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm

# --- Robust Finite Field Math ---

def is_prime(n):
    if n < 2: return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0: return False
    return True

def gf_inv(n, p): return pow(n % p, p - 2, p)

def gf_poly_mod(a, g, p):
    n = len(g) - 1; inv_lead = gf_inv(g[-1], p)
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
    for i in range(1, n // 2 + 1):
        base = [0, 1]; exp = p**i; rem_x_pow = [1]
        temp_base = list(base)
        while exp > 0:
            if exp % 2 == 1: rem_x_pow = gf_multiply(rem_x_pow, temp_base, poly, p)
            temp_base = gf_multiply(temp_base, temp_base, poly, p); exp //= 2
        diff = list(rem_x_pow)
        if len(diff) < 2: diff = diff + [0] * (2 - len(diff))
        diff[1] = (diff[1] - 1) % p
        g = gf_poly_gcd(poly, diff, p)
        if len(g) > 1: return False
    return True

def gf_find_irreducible(p, n):
    if n == 1: return [0, 1]
    poly = [0] * (n + 1); poly[n] = 1
    for i in range(1, min(p**n, 10000)):
        temp = i
        for j in range(n): poly[j] = temp % p; temp //= p
        if gf_is_irreducible(poly, p): return poly
    return poly

def gf_find_primitive(p, n, g):
    q = p**n; q_m_1 = q - 1; factors = []; temp = q_m_1; d = 2
    while d * d <= temp:
        if temp % d == 0:
            factors.append(d);
            while temp % d == 0: temp //= d
        d += 1
    if temp > 1: factors.append(temp)
    for i in range(1, min(q, 10000)):
        alpha = [0] * n; temp = i
        for j in range(n): alpha[j] = temp % p; temp //= p
        if all(x == 0 for x in alpha): continue
        is_p = True
        for f in factors:
            exp = q_m_1 // f; res = [1]; base = list(alpha); cur_e = exp
            while cur_e > 0:
                if cur_e % 2 == 1: res = gf_multiply(res, base, g, p)
                base = gf_multiply(base, base, g, p); cur_e //= 2
            if len(res)==1 and res[0]==1: is_p = False; break
        if is_p: return alpha
    return [0, 1] + [0]*(n-2) if n > 1 else [1]

class UnifiedFieldExplorer:
    def __init__(self, interactive=True):
        self.interactive = interactive
        self.mode = 'Algebraic'
        self.fig = plt.figure(figsize=(14, 10), facecolor='#050505')
        self.ax = self.fig.add_axes([0.1, 0.25, 0.8, 0.65], facecolor='black')
        self.prime_warn_text = self.fig.text(0.02, 0.95, '', color='red', fontweight='bold')
        if self.interactive: self.setup_widgets()
        self.update(None)

    def setup_widgets(self):
        ax_mode = self.fig.add_axes([0.02, 0.8, 0.1, 0.1], facecolor='#111111')
        self.radio = RadioButtons(ax_mode, ('Algebraic', 'Finite'), activecolor='#00d4ff')
        for label in self.radio.labels: label.set_color('white')
        self.radio.on_clicked(self.change_mode)
        self.ax_s1 = self.fig.add_axes([0.25, 0.14, 0.5, 0.015], facecolor='#222222')
        self.s1 = Slider(self.ax_s1, 'Degree/p ', 1, 31, valinit=3, valstep=1)
        self.ax_s2 = self.fig.add_axes([0.25, 0.11, 0.5, 0.015], facecolor='#222222')
        self.s2 = Slider(self.ax_s2, 'Coeff/n ', 1, 20, valinit=5, valstep=1)
        self.ax_s3 = self.fig.add_axes([0.25, 0.08, 0.5, 0.015], facecolor='#222222')
        self.s3 = Slider(self.ax_s3, 'Blur ', 0.1, 3.0, valinit=1.0)
        self.ax_s4 = self.fig.add_axes([0.25, 0.05, 0.5, 0.015], facecolor='#222222')
        self.s4 = Slider(self.ax_s4, 'Samples ', 1e4, 1e6, valinit=3e5, valstep=1e4)
        for s in [self.s1, self.s2, self.s3, self.s4]: s.on_changed(self.update)

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
        res = 600; all_roots = []
        for d in range(1, deg + 1):
            ns = int(num_samples // (2 * d))
            polys = np.random.randint(-coeff, coeff + 1, (ns, d + 1)).astype(float)
            polys[polys[:, -1] == 0, -1] = 1
            if d == 1: all_roots.extend(-polys[:, 0] / polys[:, 1])
            else:
                monic = polys[:, :-1] / polys[:, -1:]; matrices = np.zeros((ns, d, d))
                matrices[:, 1:, :-1] = np.eye(d - 1); matrices[:, :, -1] = -monic
                roots = np.linalg.eigvals(matrices); all_roots.extend(roots.flatten())
        all_roots = np.array(all_roots)
        mask = (np.abs(all_roots.real) <= 2.5) & (np.abs(all_roots.imag) <= 2.5)
        roots = all_roots[mask]
        H, _, _ = np.histogram2d(roots.real, roots.imag, bins=res, range=[[-2.5, 2.5], [-2.5, 2.5]])
        Z_blur = gaussian_filter(H.T, sigma=sigma)
        self.ax.imshow(Z_blur, extent=[-2.5, 2.5, -2.5, 2.5], origin='lower', cmap='magma', norm=LogNorm(vmin=0.1, vmax=max(1.1, np.max(Z_blur))))
        self.ax.set_title("Field Interference: Algebraic Number Density in Complex Plane", color='white')

    def draw_finite(self, p, n):
        if not is_prime(p): self.prime_warn_text.set_text(f"Warning: {p} is not prime")
        else: self.prime_warn_text.set_text("")
        q = p**n; g = gf_find_irreducible(p, n); alpha_gen = gf_find_primitive(p, n, g)
        elements = []
        for i in range(min(q, 5000)):
            v = 0; temp = i; cs = []
            for j in range(n):
                c = temp % p; temp //= p; cs.append(c)
                v += c * (np.exp(1j * 2 * np.pi * j / n) if n > 2 else (1 if j==0 else 1j))
            elements.append((v, cs))
        re = [e[0].real for e in elements]; im = [e[0].imag for e in elements]
        if q < 1500:
            for i in range(len(elements)):
                for j in range(i + 1, len(elements)):
                    if sum(elements[i][1][k] != elements[j][1][k] for k in range(n)) == 1:
                        self.ax.plot([re[i], re[j]], [im[i], im[j]], color='#00d4ff', alpha=0.15, lw=0.8)
        if q > 1 and q < 5000:
            curr = [0] * n; curr[0] = 1; orbit = []
            for _ in range(q):
                v = 0
                for j in range(n): v += curr[j] * (np.exp(1j*2*np.pi*j/n) if n>2 else (1 if j==0 else 1j))
                orbit.append(v); curr = gf_multiply(curr, alpha_gen, g, p)
                if all(x == (1 if j==0 else 0) for j, x in enumerate(curr)): orbit.append(orbit[0]); break
            self.ax.plot([z.real for z in orbit], [z.imag for z in orbit], color='yellow', alpha=0.6, lw=1.5, label='Multiplicative Orbit')
        self.ax.scatter(re, im, c=np.abs([e[0] for e in elements]), cmap='cool', s=100, edgecolors='white', zorder=3)
        self.ax.set_title(f"Finite Field Extension: $GF({p}^{n})$", color='white')
        self.ax.set_aspect('equal')

    def update(self, val):
        self.ax.clear(); self.ax.set_facecolor('black')
        v1 = int(self.s1.val) if self.interactive else 3
        v2 = int(self.s2.val) if self.interactive else 5
        v3 = self.s3.val if self.interactive else 1.0
        v4 = int(self.s4.val) if self.interactive else 100000
        if self.mode == 'Algebraic': self.prime_warn_text.set_text(""); self.draw_algebraic(v1, v2, v3, v4)
        else: self.draw_finite(v1, v2)
        self.ax.tick_params(colors='white')
        if self.interactive: self.fig.canvas.draw_idle()

if __name__ == "__main__":
    import sys
    explorer = UnifiedFieldExplorer(interactive="--static" not in sys.argv)
    if "--static" in sys.argv: plt.savefig('./unified_field_interference.png', dpi=300, facecolor='#050505')
    else: plt.show()
