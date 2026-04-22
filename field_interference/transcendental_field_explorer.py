import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, TextBox, CheckButtons, Button as MplButton
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LogNorm
from scipy.special import gamma as scipy_gamma, zeta as scipy_zeta
import json

# --- Finite Field Math ---
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
    for i in range(1, n // 2 + 1):
        base = [0, 1]; exp = p**i; rem_x_pow = [1]
        while exp > 0:
            if exp % 2 == 1: rem_x_pow = gf_multiply(rem_x_pow, base, poly, p)
            base = gf_multiply(base, base, poly, p); exp //= 2
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

class TranscendentalFieldExplorer:
    def __init__(self, interactive=True):
        self.interactive = interactive
        self.mode = 'Algebraic'
        self.bases = {
            'pi': np.pi, 'e': np.e, 'phi': (1+5**0.5)/2, 'pi+i': np.pi+1j, 'e^i': np.exp(1j),
            'i': 1j, 'zeta(3)': 1.202056903159594, 'gamma': 0.577215664901532
        }
        self.current_base_val = np.pi; self.current_base_name = 'pi'
        self.secondary_base_val = None; self.secondary_base_name = ''
        self.colorbar = None; self.coord_system = 'Standard'
        self.current_cmap = 'magma'; self.show_overlays = True
        self.show_farey = False
        self.coeff_set = 'Standard'; self.base_rotation = 0.0
        self.show_gradient = False; self.auto_normalize = True
        self.max_degree = 3; self.max_coeff = 5; self.sigma = 1.0; self.view_zoom = 1.0
        self.mobius_params = [1, -1j, 1, 1j] # a, b, c, d
        self.last_results = None; self.cached_vals = None; self.cache_key = None
        self.fig = plt.figure(figsize=(14, 10), facecolor='#050505')
        self.ax = self.fig.add_axes([0.15, 0.25, 0.75, 0.65], facecolor='black')
        self.probe_text = None
        if self.interactive:
            self.setup_widgets()
            self.fig.canvas.mpl_connect('motion_notify_event', self.on_hover)
        self.update(None)

    def on_hover(self, event):
        if event.inaxes != self.ax or self.probe_text is None: return
        self.probe_text.set_text(f"z = {event.xdata:.4f} + {event.ydata:.4f}i")
        self.fig.canvas.draw_idle()

    def setup_widgets(self):
        px = 0.01; pw = 0.12
        self.ax_warn = self.fig.add_axes([px, 0.97, pw, 0.02], facecolor='black')
        self.ax_warn.axis('off')
        self.prime_warn_text = self.ax_warn.text(0.5, 0.5, '', color='red', ha='center', va='center', fontsize=8, fontweight='bold')

        ax_mode = self.fig.add_axes([px, 0.80, pw, 0.17], facecolor='#111111')
        self.radio = RadioButtons(ax_mode, ('Algebraic', 'Finite', 'Transcendental', 'Complexity', 'Dominant', 'Resonance', 'Alignment', 'Phase', 'Radial', 'Parity', 'L1 Norm', 'Sensitivity', 'Entropy'), activecolor='#00d4ff')
        for l in self.radio.labels: l.set_color('white')
        self.radio.on_clicked(self.change_mode)
        self.ax_coord = self.fig.add_axes([px, 0.69, pw, 0.1], facecolor='#111111')
        self.coord_radio = RadioButtons(self.ax_coord, ('Standard', 'Log-Polar', 'Reciprocal', 'Joukowsky', 'Euler Space', 'Mobius'), activecolor='#00ff9d')
        for l in self.coord_radio.labels: l.set_color('white')
        self.coord_radio.on_clicked(self.change_coord)
        self.ax_cmap = self.fig.add_axes([px, 0.56, pw, 0.12], facecolor='#111111')
        self.cmap_radio = RadioButtons(self.ax_cmap, ('magma', 'viridis', 'inferno', 'plasma', 'twilight', 'ocean'), activecolor='#ffcc00')
        for l in self.cmap_radio.labels: l.set_color('white')
        self.cmap_radio.on_clicked(self.change_cmap)
        self.ax_base = self.fig.add_axes([px, 0.40, pw, 0.15], facecolor='#111111')
        self.base_radio = RadioButtons(self.ax_base, list(self.bases.keys()), activecolor='#ff007f')
        for l in self.base_radio.labels: l.set_color('white')
        self.base_radio.on_clicked(self.change_base)
        self.ax_text = self.fig.add_axes([px, 0.35, pw, 0.04], facecolor='#111111')
        self.text_box = TextBox(self.ax_text, 'Base A: ', initial='pi', color='white', hovercolor='#333333')
        self.text_box.on_submit(self.submit_custom_base)
        self.ax_text2 = self.fig.add_axes([px, 0.30, pw, 0.04], facecolor='#111111')
        self.text_box2 = TextBox(self.ax_text2, 'Base B: ', initial='', color='white', hovercolor='#333333')
        self.text_box2.on_submit(self.submit_secondary_base)
        self.ax_coeff = self.fig.add_axes([px, 0.20, pw, 0.09], facecolor='#111111')
        self.coeff_radio = RadioButtons(self.ax_coeff, ('Standard', 'Binary', 'Littlewood'), activecolor='#ff5500')
        for l in self.coeff_radio.labels: l.set_color('white')
        self.coeff_radio.on_clicked(self.change_coeff_set)
        self.ax_check = self.fig.add_axes([px, 0.08, pw, 0.11], facecolor='#111111')
        self.check = CheckButtons(self.ax_check, ['Overlays', 'Gradient', 'Normalize', 'Farey'], [True, False, True, False])
        for l in self.check.labels: l.set_color('white')
        self.check.on_clicked(self.toggle_checks)
        self.ax_export = self.fig.add_axes([px, 0.04, pw, 0.035], facecolor='#111111')
        self.btn_export = MplButton(self.ax_export, 'Export PNG', color='#333333', hovercolor='#555555')
        self.btn_export.label.set_color('white'); self.btn_export.on_clicked(self.export_high_res)
        self.ax_data = self.fig.add_axes([px, 0.005, pw, 0.035], facecolor='#111111')
        self.btn_data = MplButton(self.ax_data, 'Export Data', color='#333333', hovercolor='#555555')
        self.btn_data.label.set_color('white'); self.btn_data.on_clicked(self.export_data)

        self.ax_s1 = self.fig.add_axes([0.25, 0.15, 0.5, 0.015], facecolor='#222222')
        self.s1 = Slider(self.ax_s1, 'Degree/p ', 1, 11, valinit=3, valstep=1)
        self.ax_s2 = self.fig.add_axes([0.25, 0.12, 0.5, 0.015], facecolor='#222222')
        self.s2 = Slider(self.ax_s2, 'Coeff/n ', 1, 20, valinit=5, valstep=1)
        self.ax_s3 = self.fig.add_axes([0.25, 0.09, 0.5, 0.015], facecolor='#222222')
        self.s3 = Slider(self.ax_s3, 'Blur ', 0.1, 3.0, valinit=1.0)
        self.ax_s4 = self.fig.add_axes([0.25, 0.06, 0.5, 0.015], facecolor='#222222')
        self.s4 = Slider(self.ax_s4, 'Zoom ', 0.1, 20.0, valinit=1.0)
        self.ax_s5 = self.fig.add_axes([0.25, 0.03, 0.5, 0.015], facecolor='#222222')
        self.s5 = Slider(self.ax_s5, 'Samples ', 1e4, 2e6, valinit=3e5, valstep=1e4)

        self.ax_br = self.fig.add_axes([0.25, 0.18, 0.23, 0.015], facecolor='#222222')
        self.sr = Slider(self.ax_br, 'Base Re ', -5.0, 5.0, valinit=np.pi)
        self.ax_bi = self.fig.add_axes([0.52, 0.18, 0.23, 0.015], facecolor='#222222')
        self.si = Slider(self.ax_bi, 'Base Im ', -5.0, 5.0, valinit=0.0)

        self.ax_mob = self.fig.add_axes([0.77, 0.25, 0.08, 0.15], facecolor='#111111')
        self.ax_mob.set_title("Mobius", color='white', fontsize=10)
        self.text_mob = TextBox(self.ax_mob, 'a,b,c,d: ', initial='1,-1j,1,1j', color='white', hovercolor='#333333')
        self.text_mob.on_submit(self.submit_mobius)

        for s in [self.s1, self.s2, self.s3, self.s4, self.s5, self.sr, self.si]: s.on_changed(self.update)

    def submit_mobius(self, text):
        try:
            self.mobius_params = [complex(x) for x in text.split(',')]
            if len(self.mobius_params) != 4: self.mobius_params = [1, -1j, 1, 1j]
        except: self.mobius_params = [1, -1j, 1, 1j]
        self.update(None)

    def change_mode(self, label):
        self.mode = label; is_trans = self.mode in ['Transcendental', 'Complexity', 'Dominant', 'Resonance', 'Alignment', 'Phase', 'Radial', 'Parity', 'L1 Norm', 'Sensitivity', 'Entropy']
        self.ax_base.set_visible(is_trans); self.ax_text.set_visible(is_trans); self.ax_text2.set_visible(is_trans); self.ax_coord.set_visible(is_trans); self.ax_cmap.set_visible(is_trans); self.ax_coeff.set_visible(is_trans)
        if self.mode == 'Algebraic': self.s1.label.set_text('Max Degree '); self.s2.label.set_text('Max Coeff '); self.ax_s3.set_visible(True)
        elif self.mode == 'Finite': self.s1.label.set_text('Prime p '); self.s2.label.set_text('Degree n '); self.ax_s3.set_visible(False)
        else: self.s1.label.set_text('Extension Deg '); self.s2.label.set_text('Coeff Range '); self.ax_s3.set_visible(True)
        self.update(None)

    def change_coord(self, label): self.coord_system = label; self.update(None)
    def change_cmap(self, label): self.current_cmap = label; self.update(None)
    def change_coeff_set(self, label): self.coeff_set = label; self.update(None)
    def change_base(self, label):
        self.current_base_name = label; self.current_base_val = self.bases[label]
        if self.interactive:
            self.sr.set_val(self.current_base_val.real); self.si.set_val(self.current_base_val.imag)
        self.update(None)
    def toggle_checks(self, label):
        if label == 'Overlays': self.show_overlays = not self.show_overlays
        elif label == 'Gradient': self.show_gradient = not self.show_gradient
        elif label == 'Normalize': self.auto_normalize = not self.auto_normalize
        elif label == 'Farey': self.show_farey = not self.show_farey
        self.update(None)
    def reset_view(self, event): self.s4.set_val(1.0); self.update(None)
    def export_high_res(self, event):
        fn = f"field_export_{self.current_base_name}_{self.mode}_{self.coord_system}.png"; self.fig.savefig(fn, dpi=600, facecolor='#050505'); print(f"Exported to {fn}")
    def export_data(self, event):
        if self.last_results is None: print("No data"); return
        fn = f"field_data_{self.current_base_name}_{self.mode}.json"
        with open(fn, 'w') as f: json.dump({"real": self.last_results.real.tolist(), "imag": self.last_results.imag.tolist()}, f)
    def submit_custom_base(self, text):
        val = self._safe_eval(text)
        if val is not None:
            self.current_base_name = text; self.current_base_val = complex(val)
            if self.interactive:
                self.sr.set_val(self.current_base_val.real); self.si.set_val(self.current_base_val.imag)
            self.update(None)
    def submit_secondary_base(self, text):
        if not text.strip(): self.secondary_base_val = None; self.secondary_base_name = ''
        else:
            val = self._safe_eval(text);
            if val is not None: self.secondary_base_name = text; self.secondary_base_val = complex(val)
        self.update(None)
    def _safe_eval(self, text):
        try:
            allowed = { 'pi': np.pi, 'e': np.e, 'phi': (1+5**0.5)/2, 'i': 1j, 'j': 1j, 'exp': np.exp, 'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 'sqrt': np.sqrt, 'log': np.log, 'gamma': 0.577215664901532, 'zeta': scipy_zeta, 'gamma_func': scipy_gamma, 'abs': np.abs, 'pow': np.power, 'real': np.real, 'imag': np.imag }
            return eval(text, {"__builtins__": None}, allowed)
        except Exception as e: print(f"Eval error: {e}"); return None

    def get_convergents(self, x, n=10):
        if abs(x.imag) > 1e-9:
            limit = int(np.ceil(np.abs(x)) + 2); res = []
            for re in range(-limit, limit + 1):
                for im in range(-limit, limit + 1):
                    res.append(complex(re, im)); res.append(complex(re + 0.5*im, np.sqrt(3)/2*im))
            return res
        x = x.real; a = int(np.floor(x)); convs = [a]; rem = x - a
        for _ in range(n):
            if abs(rem) < 1e-11: break
            inv = 1.0 / rem; a = int(np.floor(inv)); convs.append(a); rem = inv - a
        res = []
        for i in range(1, len(convs) + 1):
            p, q = 1, 0
            for j in reversed(range(i)): p, q = convs[j] * p + q, p
            res.append(p / q if q != 0 else p)
        return res

    def get_farey_sequence(self, n):
        a, b, c, d = 0, 1, 1, n; res = [a/b]
        while c <= n:
            k = (n + b) // d; a, b, c, d = c, d, k*c - a, k*d - b; res.append(a/b)
        return res

    def get_coeffs(self, ns, d, c_range):
        if self.coeff_set == 'Binary': return np.random.randint(0, 2, (ns, d + 1))
        if self.coeff_set == 'Littlewood': return np.random.choice([-1, 1], size=(ns, d + 1))
        return np.random.randint(-c_range, c_range + 1, (ns, d + 1))

    def map_coord(self, v, b):
        if self.coord_system == 'Log-Polar': r = np.abs(v); t = np.angle(v); r[r < 1e-15] = 1e-15; return np.log(r) + 1j * t
        if self.coord_system == 'Reciprocal': m = np.abs(v) > 1e-15; r = np.zeros_like(v); r[m] = 1.0 / v[m]; return r
        if self.coord_system == 'Joukowsky': m = np.abs(v) > 1e-15; r = np.zeros_like(v); r[m] = v[m] + 1.0 / v[m]; return r
        if self.coord_system == 'Mobius':
            a, b, c, d = self.mobius_params; return (a*v + b) / (c*v + d + 1e-15)
        if self.coord_system == 'Euler Space': s = np.abs(b) if np.abs(b) > 0 else 1.0; return np.exp(1j * np.pi * v / s)
        return v

    def draw_transcendental(self, deg, coeff_range, sigma, mode='Transcendental', zoom=1.0, num_samples=300000, base=np.pi):
        ckey = (deg, coeff_range, num_samples, base, self.coeff_set, self.secondary_base_val)
        if self.cache_key == ckey and self.cached_vals is not None: vals, coeffs, powers = self.cached_vals
        else:
            if self.secondary_base_val is None: coeffs = self.get_coeffs(num_samples, deg, coeff_range); powers = base ** np.arange(deg + 1); vals = coeffs @ powers
            else:
                d_sub = max(1, deg // 2); pa = base ** np.arange(d_sub + 1); pb = self.secondary_base_val ** np.arange(d_sub + 1)
                powers = np.outer(pa, pb).flatten(); coeffs = self.get_coeffs(num_samples, len(powers) - 1, coeff_range); vals = coeffs @ powers
            self.cached_vals = (vals, coeffs, powers); self.cache_key = ckey
        self.last_results = vals; mv = self.map_coord(vals, base); res = int(min(1200, 400 + 200 * np.log10(num_samples/1e4) * np.sqrt(zoom)))
        if self.auto_normalize: limit = np.percentile(np.abs(mv), 95) / (zoom or 1.0); xr = (-limit, limit); yr = (-limit, limit)
        else: xr = (-10/zoom, 10/zoom); yr = (-10/zoom, 10/zoom)
        tb = self.current_base_name.replace('_', '\\_')
        if mode != 'Transcendental':
            ix = ((mv.real - xr[0]) / (xr[1] - xr[0] or 1) * (res - 1)).astype(int); iy = ((mv.imag - yr[0]) / (yr[1] - yr[0] or 1) * (res - 1)).astype(int)
            mask = (ix >= 0) & (ix < res) & (iy >= 0) & (iy < res)
            cmap = self.current_cmap
            if mode == 'Complexity': Zm = np.full((res, res), np.inf); np.minimum.at(Zm, (iy[mask], ix[mask]), np.sum(np.abs(coeffs), axis=1)[mask].astype(float)); Zm[np.isinf(Zm)] = np.nan; label = 'Min L1 Complexity'
            elif mode == 'Dominant': Zm = np.full((res, res), -1.0); np.maximum.at(Zm, (iy[mask], ix[mask]), np.argmax(np.abs(coeffs * powers), axis=1)[mask].astype(float)); Zm[Zm == -1.0] = np.nan; label = 'Dominant Power'
            elif mode == 'Resonance':
                anchors = np.array(self.get_convergents(base, n=12)); metric = np.full(len(vals), 1e9); chunk = 100000
                for i in range(0, len(vals), chunk):
                    end = min(i + chunk, len(vals)); ds = np.abs(vals[i:end, np.newaxis] - anchors); metric[i:end] = np.min(ds, axis=1)
                metric = -np.log10(metric + 1e-15); label = 'Rational/Lattice Resonance'; Zm = np.full((res, res), -15.0); np.maximum.at(Zm, (iy[mask], ix[mask]), metric[mask].astype(float)); Zm[Zm == -15.0] = np.nan
            elif mode == 'Sensitivity':
                eps = 1e-6; base2 = base * np.exp(1j * eps); vals2 = coeffs @ (base2 ** np.arange(deg + 1) if self.secondary_base_val is None else np.outer(base2 ** np.arange(max(1, deg//2) + 1), self.secondary_base_val ** np.arange(max(1, deg//2) + 1)).flatten())
                metric = np.abs(vals2 - vals) / eps; label = 'Rotation Sensitivity'; H_sum, _, _ = np.histogram2d(mv.real[mask], mv.imag[mask], bins=res, range=[[xr[0], xr[1]], [yr[0], yr[1]]], weights=metric[mask]); H_cnt, _, _ = np.histogram2d(mv.real[mask], mv.imag[mask], bins=res, range=[[xr[0], xr[1]], [yr[0], yr[1]]]); Zm = np.divide(H_sum.T, H_cnt.T, out=np.full((res, res), np.nan), where=H_cnt.T > 0)
            elif mode == 'Entropy': H, _, _ = np.histogram2d(mv.real[mask], mv.imag[mask], bins=res, range=[[xr[0], xr[1]], [yr[0], yr[1]]]); Zf = gaussian_filter(H.T, sigma=sigma); Zf[Zf <= 0] = 1e-15; Zm = -Zf * np.log(Zf); label = 'Local Shannon Entropy'
            elif mode in ['Alignment', 'L1 Norm']:
                if mode == 'Alignment': tms = coeffs * powers; phs = np.exp(1j * np.angle(tms)); metric = 1.0 - np.abs(np.mean(phs, axis=1)); label = 'Phase Symmetrization'
                else: metric = np.sum(np.abs(coeffs), axis=1); label = 'Sum of Absolute Coeffs'
                H_sum, _, _ = np.histogram2d(mv.real[mask], mv.imag[mask], bins=res, range=[[xr[0], xr[1]], [yr[0], yr[1]]], weights=metric[mask]); H_cnt, _, _ = np.histogram2d(mv.real[mask], mv.imag[mask], bins=res, range=[[xr[0], xr[1]], [yr[0], yr[1]]]); Zm = np.divide(H_sum.T, H_cnt.T, out=np.full((res, res), np.nan), where=H_cnt.T > 0)
            elif mode == 'Phase': Zm = np.full((res, res), np.nan); np.maximum.at(Zm, (iy[mask], ix[mask]), np.angle(vals)[mask].astype(float)); label = 'Argument (Phase)'; cmap = 'twilight'
            elif mode == 'Radial': Zm = np.full((res, res), -1.0); np.maximum.at(Zm, (iy[mask], ix[mask]), np.abs(vals)[mask].astype(float)); Zm[Zm == -1.0] = np.nan; label = 'Radial Distance'
            elif mode == 'Parity': Zm = np.full((res, res), -1.0); np.maximum.at(Zm, (iy[mask], ix[mask]), (np.sum(coeffs, axis=1) % 2)[mask].astype(float)); Zm[Zm == -1.0] = np.nan; label = 'Coeff Parity'
            im = self.ax.imshow(Zm, extent=[xr[0], xr[1], yr[0], yr[1]], origin='lower', cmap=cmap)
            if self.colorbar:
                try: self.colorbar.remove()
                except: pass
            self.colorbar = self.fig.colorbar(im, ax=self.ax, label=label); self.colorbar.ax.yaxis.label.set_color('white'); self.colorbar.ax.tick_params(colors='white')
            self.ax.set_title(rf"Transcendental Analysis: $\mathbb{{Q}}({tb})$" + "\n" + f"{label} across the extension", color='white', fontsize=14)
        else:
            if self.colorbar:
                try: self.colorbar.remove()
                except: pass
                self.colorbar = None
            H, _, _ = np.histogram2d(mv.real, mv.imag, bins=res, range=[[xr[0], xr[1]], [yr[0], yr[1]]])
            Zf = gaussian_filter(H.T, sigma=sigma)
            if self.show_gradient: gy, gx = np.gradient(Zf); Zf = np.sqrt(gx**2 + gy**2); norm = plt.Normalize(vmin=0, vmax=np.percentile(Zf, 99))
            else: vmin = 1.0; vmax = max(vmin + 1.0, np.max(Zf)); norm = LogNorm(vmin=vmin, vmax=vmax)
            self.ax.imshow(Zf, extent=[xr[0], xr[1], yr[0], yr[1]], origin='lower', cmap=self.current_cmap, norm=norm)
            title = rf"$\mathbb{{Q}}({tb}"
            if self.secondary_base_val is not None: title += f", {self.secondary_base_name.replace('_', '\\_')}"
            self.ax.set_title(f"Transcendental Field Interference: {title})$\nCoupled Extension micro-structures", color='white', fontsize=14)
        if self.show_overlays:
            if not np.iscomplex(base): cvs = self.get_convergents(base); self.ax.scatter(cvs, np.zeros_like(cvs), color='red', s=50, marker='x', label='Rational Convergents', zorder=5); self.ax.legend(facecolor='#111111', labelcolor='white')
            self.draw_reference_overlays(xr, yr, base)

    def draw_reference_overlays(self, xr, yr, b):
        t = np.linspace(0, 2*np.pi, 500); circle = np.exp(1j * t); mv = self.map_coord(circle, b)
        self.ax.plot(mv.real, mv.imag, color='white', ls='--', alpha=0.3, lw=1, label='Unit Circle')
        self.ax.axhline(0, color='white', alpha=0.1, lw=0.5); self.ax.axvline(0, color='white', alpha=0.1, lw=0.5)
        if self.show_farey:
            fseq = self.get_farey_sequence(8)
            for r in fseq:
                mv_r = self.map_coord(r, b); self.ax.plot([mv_r.real], [mv_r.imag], 'ro', markersize=2, alpha=0.5)

    def draw_algebraic(self, deg, coeff, sigma, num_samples=50000):
        res = 600; all_r = []
        for d in range(1, deg + 1):
            ns = num_samples // d; polys = self.get_coeffs(ns, d, coeff).astype(float); polys[polys[:, -1] == 0, -1] = 1
            if d == 1: all_r.extend(-polys[:, 0] / polys[:, 1])
            else:
                monic = polys[:, :-1] / polys[:, -1:]; mats = np.zeros((ns, d, d)); mats[:, 1:, :-1] = np.eye(d - 1); mats[:, :, -1] = -monic
                roots = np.linalg.eigvals(mats); all_r.extend(roots.flatten())
        all_r = np.array(all_r); self.last_results = all_r; mv = self.map_coord(all_r, 1.0)
        if self.auto_normalize: limit = np.percentile(np.abs(mv), 95); xr = (-limit, limit); yr = (-limit, limit)
        else: xr = (-2.5, 2.5); yr = (-2.5, 2.5)
        roots = mv[(np.abs(mv.real) <= 10) & (np.abs(mv.imag) <= 10)]
        H, _, _ = np.histogram2d(roots.real, roots.imag, bins=res, range=[[xr[0], xr[1]], [yr[0], yr[1]]])
        Zb = gaussian_filter(H.T, sigma=sigma); vmin = 0.1; vmax = max(vmin + 1.0, np.max(Zb))
        self.ax.imshow(Zb, extent=[xr[0], xr[1], yr[0], yr[1]], origin='lower', cmap=self.current_cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
        self.ax.set_title(rf"Field Interference: Algebraic Density ({self.coeff_set})", color='white', fontsize=15)

    def draw_finite(self, p, n):
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]; p = min(primes, key=lambda x:abs(x-p)); q = p**n
        g = gf_find_irreducible(p, n); alpha_gen = gf_find_primitive(p, n, g); elements = []
        for i in range(q):
            v = 0; temp = i; cs = []
            for j in range(n):
                c = temp % p; temp //= p; cs.append(c)
                if n == 2: v += c * (1 if j == 0 else 1j)
                else: v += c * np.exp(1j * 2 * np.pi * j / n)
            elements.append((v, cs))
        self.last_results = np.array([e[0] for e in elements]); mv = self.map_coord(self.last_results, 1.0)
        re = [e.real for e in mv]; im = [e.imag for e in mv]
        if q < 2000:
            for i in range(q):
                for j in range(i + 1, q):
                    if sum(elements[i][1][k] != elements[j][1][k] for k in range(n)) == 1: self.ax.plot([re[i], re[j]], [im[i], im[j]], color='#00d4ff', alpha=0.15, lw=0.8, zorder=1)
        if q > 1 and q < 5000:
            curr = [0] * n; curr[0] = 1; orbit = []
            for _ in range(q):
                v = 0
                for j in range(n): v += curr[j] * ((1 if j==0 else 1j) if n==2 else np.exp(1j*2*np.pi*j/n))
                orbit.append(v); curr = gf_multiply(curr, alpha_gen, g, p)
                if all(x == (1 if j==0 else 0) for j, x in enumerate(curr)): orbit.append(orbit[0]); break
            mov = self.map_coord(np.array(orbit), 1.0)
            self.ax.plot([z.real for z in mov], [z.imag for z in mov], color='yellow', alpha=0.6, lw=1.5, zorder=2, label='Multiplicative Orbit')
        self.ax.scatter(re, im, c=np.abs([e[0] for e in elements]), cmap='cool', s=80, edgecolors='white', zorder=3)
        self.ax.set_title(f"Finite Field Extension: $GF({p}^{n})$", color='white', fontsize=15); self.ax.set_aspect('equal'); self.ax.legend(facecolor='#111111', labelcolor='white')

    def is_prime(self, n):
        if n < 2: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0: return False
        return True

    def update(self, val):
        self.ax.clear(); self.ax.set_facecolor('black')
        if self.interactive:
            if self.mode == 'Finite' and not self.is_prime(int(self.s1.val)):
                self.prime_warn_text.set_text("Warning: p not prime")
            else:
                self.prime_warn_text.set_text("")
        v1 = int(self.s1.val) if self.interactive else 3; v2 = int(self.s2.val) if self.interactive else 5
        v3 = self.s3.val if self.interactive else 1.0; v4 = self.s4.val if self.interactive else 1.0; v5 = int(self.s5.val) if self.interactive else 300000
        cur_base = complex(self.sr.val, self.si.val) if self.interactive else self.current_base_val
        self.probe_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes, color='cyan', fontsize=10, verticalalignment='top', bbox=dict(facecolor='black', alpha=0.6))
        if self.mode == 'Algebraic': self.draw_algebraic(v1, v2, v3, v5 // 10)
        elif self.mode == 'Finite': self.draw_finite(v1, v2)
        else: self.draw_transcendental(v1, v2, v3, mode=self.mode, zoom=v4, num_samples=v5, base=cur_base)
        self.ax.tick_params(colors='white'); self.fig.canvas.draw_idle()

if __name__ == "__main__":
    import sys
    if "--static" in sys.argv:
        explorer = TranscendentalFieldExplorer(interactive=False); explorer.mode = 'Transcendental'; explorer.update(None)
        plt.savefig('./transcendental_interference_static.png', dpi=300, facecolor='#050505'); print("Saved static image.")
    else: explorer = TranscendentalFieldExplorer(interactive=True); plt.show()
