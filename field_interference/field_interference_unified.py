import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
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

def gf_find_primitive(p, n, g):
    q = p**n; q_m_1 = q - 1; factors = []; temp = q_m_1; d = 2
    while d * d <= temp:
        if temp % d == 0:
            factors.append(d)
            while temp % d == 0: temp //= d
        d += 1
    if temp > 1: factors.append(temp)
    for _ in range(1000):
        alpha = [np.random.randint(0, p) for _ in range(n)]
        if all(x == 0 for x in alpha): continue
        is_p = True
        for f in factors:
            exp = q_m_1 // f; res = [1]; base = list(alpha)
            while exp > 0:
                if exp % 2 == 1: res = gf_multiply(res, base, g, p)
                base = gf_multiply(base, base, g, p); exp //= 2
            if len(res)==1 and res[0]==1:
                is_p = False; break
        if is_p: return alpha
    return [0, 1] + [0]*(n-2) if n > 1 else [1]

class UnifiedFieldExplorer:
    def __init__(self, interactive=True):
        self.interactive = interactive
        self.mode = 'Algebraic'
        self.fig = plt.figure(figsize=(14, 10), facecolor='#050505')
        self.ax = self.fig.add_axes([0.1, 0.25, 0.8, 0.65], facecolor='black')
        if self.interactive: self.setup_widgets()
        self.update(None)

    def setup_widgets(self):
        ax_mode = self.fig.add_axes([0.02, 0.8, 0.1, 0.1], facecolor='#111111')
        self.radio = RadioButtons(ax_mode, ('Algebraic', 'Finite'), activecolor='#00d4ff')
        for label in self.radio.labels: label.set_color('white')
        self.radio.on_clicked(self.change_mode)
        self.ax_s1 = self.fig.add_axes([0.25, 0.12, 0.5, 0.02], facecolor='#222222')
        self.s1 = Slider(self.ax_s1, 'Degree/p ', 1, 31, valinit=3, valstep=1)
        self.ax_s2 = self.fig.add_axes([0.25, 0.08, 0.5, 0.02], facecolor='#222222')
        self.s2 = Slider(self.ax_s2, 'Coeff/n ', 1, 20, valinit=5, valstep=1)
        self.ax_s3 = self.fig.add_axes([0.25, 0.04, 0.5, 0.02], facecolor='#222222')
        self.s3 = Slider(self.ax_s3, 'Blur ', 0.1, 3.0, valinit=1.0)
        for s in [self.s1, self.s2, self.s3]: s.on_changed(self.update)

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
        res = 400; Z = np.zeros((res, res))
        for d in range(1, deg + 1):
            num_samples = 10000 // d
            for _ in range(num_samples):
                poly = np.random.randint(-coeff, coeff + 1, d + 1)
                if poly[-1] == 0: poly[-1] = 1
                roots = np.roots(poly[::-1])
                for r in roots:
                    if abs(r.real) <= 2 and abs(r.imag) <= 2:
                        ix = int((r.real + 2) / 4 * (res - 1))
                        iy = int((r.imag + 2) / 4 * (res - 1))
                        Z[iy, ix] += 1
        Z_blur = gaussian_filter(Z, sigma=sigma)
        self.ax.imshow(Z_blur, extent=[-2, 2, -2, 2], origin='lower', cmap='magma', norm=LogNorm(vmin=0.1))
        self.ax.set_title("Field Interference: Algebraic Number Density in Complex Plane", color='white')

    def draw_finite(self, p, n):
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]; p = min(primes, key=lambda x:abs(x-p))
        g = gf_find_irreducible(p, n); alpha_gen = gf_find_primitive(p, n, g)
        elements = []
        for i in range(min(p**n, 5000)):
            v = 0; temp = i; cs = []
            for j in range(n):
                c = temp % p; temp //= p; cs.append(c)
                v += c * (np.exp(1j * 2 * np.pi * j / n) if n > 2 else (1 if j==0 else 1j))
            elements.append((v, cs))
        re, im = [e[0].real for e in elements], [e[0].imag for e in elements]
        self.ax.scatter(re, im, c=np.abs([e[0] for e in elements]), cmap='cool', s=100, edgecolors='white', zorder=3)
        if p**n < 1500:
            for i in range(len(elements)):
                for j in range(i + 1, len(elements)):
                    if sum(elements[i][1][k] != elements[j][1][k] for k in range(n)) == 1:
                        self.ax.plot([re[i], re[j]], [im[i], im[j]], color='#00d4ff', alpha=0.1, lw=0.5)
        self.ax.set_title(f"Finite Field Extension: $GF({p}^{n})$", color='white'); self.ax.set_aspect('equal')

    def update(self, val):
        self.ax.clear(); self.ax.set_facecolor('black')
        v1 = int(self.s1.val) if self.interactive else 3
        v2 = int(self.s2.val) if self.interactive else 5
        v3 = self.s3.val if self.interactive else 1.0
        if self.mode == 'Algebraic': self.draw_algebraic(v1, v2, v3)
        else: self.draw_finite(v1, v2)
        self.ax.tick_params(colors='white')
        if self.interactive: self.fig.canvas.draw_idle()

if __name__ == "__main__":
    import sys
    explorer = UnifiedFieldExplorer(interactive="--static" not in sys.argv)
    if "--static" in sys.argv: plt.savefig('./unified_field_interference.png', dpi=300, facecolor='#050505')
    else: plt.show()
