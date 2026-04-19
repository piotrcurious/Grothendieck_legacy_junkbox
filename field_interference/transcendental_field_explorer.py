import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, TextBox, CheckButtons, Button as MplButton
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
        self.coord_system = 'Standard'
        self.current_cmap = 'magma'
        self.show_overlays = True
        self.coeff_set = 'Standard'
        self.base_rotation = 0.0
        self.show_gradient = False
        
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
        # Mode Selection
        ax_mode = self.fig.add_axes([0.01, 0.82, 0.13, 0.15], facecolor='#111111')
        self.radio = RadioButtons(ax_mode, ('Algebraic', 'Finite', 'Transcendental', 'Complexity', 'Dominant', 'Mean Coeff', 'Phase', 'Radial'), activecolor='#00d4ff')
        for label in self.radio.labels:
            label.set_color('white')
        self.radio.on_clicked(self.change_mode)

        # Coordinate System Selection
        self.ax_coord = self.fig.add_axes([0.01, 0.71, 0.13, 0.1], facecolor='#111111')
        self.coord_radio = RadioButtons(self.ax_coord, ('Standard', 'Log-Polar', 'Reciprocal'), activecolor='#00ff9d')
        for label in self.coord_radio.labels:
            label.set_color('white')
        self.coord_radio.on_clicked(self.change_coord)
        self.ax_coord.set_visible(False)

        # Colormap Selection
        self.ax_cmap = self.fig.add_axes([0.01, 0.58, 0.13, 0.12], facecolor='#111111')
        self.cmap_radio = RadioButtons(self.ax_cmap, ('magma', 'viridis', 'twilight', 'ocean'), activecolor='#ffcc00')
        for label in self.cmap_radio.labels:
            label.set_color('white')
        self.cmap_radio.on_clicked(self.change_cmap)
        self.ax_cmap.set_visible(False)
        
        # Base Selection (for Transcendental mode)
        self.ax_base = self.fig.add_axes([0.01, 0.42, 0.13, 0.15], facecolor='#111111')
        self.base_radio = RadioButtons(self.ax_base, list(self.bases.keys()), activecolor='#ff007f')
        for label in self.base_radio.labels:
            label.set_color('white')
        self.base_radio.on_clicked(self.change_base)
        self.ax_base.set_visible(False)

        # Custom Base Input
        self.ax_text = self.fig.add_axes([0.01, 0.36, 0.13, 0.05], facecolor='#111111')
        self.text_box = TextBox(self.ax_text, 'Custom: ', initial='pi', color='white', hovercolor='#333333')
        self.text_box.on_submit(self.submit_custom_base)
        self.ax_text.set_visible(False)

        # Coefficient Set Selection
        self.ax_coeff = self.fig.add_axes([0.01, 0.22, 0.13, 0.1], facecolor='#111111')
        self.coeff_radio = RadioButtons(self.ax_coeff, ('Standard', 'Binary', 'Littlewood'), activecolor='#ff5500')
        for label in self.coeff_radio.labels:
            label.set_color('white')
        self.coeff_radio.on_clicked(self.change_coeff_set)
        self.ax_coeff.set_visible(False)

        # Overlays Toggle
        self.ax_check = self.fig.add_axes([0.01, 0.13, 0.13, 0.08], facecolor='#111111')
        self.check = CheckButtons(self.ax_check, ['Overlays', 'Gradient'], [True, False])
        for label in self.check.labels:
            label.set_color('white')
        self.check.on_clicked(self.toggle_checks)

        # Export Button
        self.ax_export = self.fig.add_axes([0.01, 0.05, 0.13, 0.05], facecolor='#111111')
        self.btn_export = MplButton(self.ax_export, 'Export PNG', color='#333333', hovercolor='#555555')
        self.btn_export.label.set_color('white')
        self.btn_export.on_clicked(self.export_high_res)
        
        # Sliders
        self.ax_s1 = self.fig.add_axes([0.25, 0.14, 0.5, 0.015], facecolor='#222222')
        self.s1 = Slider(self.ax_s1, 'Degree/p ', 1, 11, valinit=3, valstep=1)
        
        self.ax_s2 = self.fig.add_axes([0.25, 0.11, 0.5, 0.015], facecolor='#222222')
        self.s2 = Slider(self.ax_s2, 'Coeff/n ', 1, 20, valinit=5, valstep=1)
        
        self.ax_s3 = self.fig.add_axes([0.25, 0.08, 0.5, 0.015], facecolor='#222222')
        self.s3 = Slider(self.ax_s3, 'Blur ', 0.1, 3.0, valinit=1.0)

        self.ax_s4 = self.fig.add_axes([0.25, 0.05, 0.5, 0.015], facecolor='#222222')
        self.s4 = Slider(self.ax_s4, 'Zoom ', 0.1, 20.0, valinit=1.0)

        self.ax_s5 = self.fig.add_axes([0.25, 0.02, 0.5, 0.015], facecolor='#222222')
        self.s5 = Slider(self.ax_s5, 'Samples ', 1e4, 1e6, valinit=3e5, valstep=1e4)

        self.ax_s6 = self.fig.add_axes([0.25, 0.17, 0.5, 0.015], facecolor='#222222')
        self.s6 = Slider(self.ax_s6, 'Base Rot ', 0, 2*np.pi, valinit=0.0)
        
        self.s1.on_changed(self.update)
        self.s2.on_changed(self.update)
        self.s3.on_changed(self.update)
        self.s4.on_changed(self.update)
        self.s5.on_changed(self.update)
        self.s6.on_changed(self.update)

    def change_mode(self, label):
        self.mode = label
        is_trans = self.mode in ['Transcendental', 'Complexity', 'Dominant', 'Mean Coeff', 'Phase', 'Radial']
        self.ax_base.set_visible(is_trans)
        self.ax_text.set_visible(is_trans)
        self.ax_coord.set_visible(is_trans)
        self.ax_cmap.set_visible(is_trans)
        self.ax_coeff.set_visible(is_trans)

    def change_coord(self, label):
        self.coord_system = label
        self.update(None)

    def change_cmap(self, label):
        self.current_cmap = label
        self.update(None)

    def change_coeff_set(self, label):
        self.coeff_set = label
        self.update(None)
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

    def toggle_checks(self, label):
        if label == 'Overlays':
            self.show_overlays = not self.show_overlays
        elif label == 'Gradient':
            self.show_gradient = not self.show_gradient
        self.update(None)

    def export_high_res(self, event):
        filename = f"field_export_{self.current_base_name}_{self.mode}_{self.coord_system}.png"
        self.fig.savefig(filename, dpi=600, facecolor='#050505')
        print(f"High-resolution export saved to {filename}")

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
        if abs(x.imag) > 1e-9:
            return convergents
        x = x.real
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

    def draw_transcendental(self, deg, coeff_range, sigma, mode='Transcendental', zoom=1.0, num_samples=300000, rotation=0.0):
        """
        Visualizes the 'interference' of a transcendental base using vectorized computation.
        We generate elements of the form: sum_{i=0}^{deg} c_i * base^i
        """
        # Dynamic Resolution based on zoom and samples
        res = int(min(1200, 400 + 200 * np.log10(num_samples/1e4) * np.sqrt(zoom)))
        base = self.current_base_val * np.exp(1j * rotation)

        # Vectorized coefficient generation
        if self.coeff_set == 'Binary':
            coeffs = np.random.randint(0, 2, (num_samples, deg + 1))
        elif self.coeff_set == 'Littlewood':
            coeffs = np.random.choice([-1, 1], size=(num_samples, deg + 1))
        else: # Standard
            coeffs = np.random.randint(-coeff_range, coeff_range + 1, (num_samples, deg + 1))

        # Ensure at least some non-zero to avoid 0/0
        coeffs[np.all(coeffs == 0, axis=1), 0] = 1

        powers = base ** np.arange(deg + 1)
        vals = coeffs @ powers

        # Apply Coordinate Mapping
        if self.coord_system == 'Log-Polar':
            # r' = log(r), theta' = theta
            r = np.abs(vals)
            theta = np.angle(vals)
            # Avoid log(0)
            r[r < 1e-9] = 1e-9
            mapped_vals = np.log(r) + 1j * theta
        elif self.coord_system == 'Reciprocal':
            # w = 1/z
            mask = np.abs(vals) > 1e-9
            mapped_vals = np.zeros_like(vals)
            mapped_vals[mask] = 1.0 / vals[mask]
        else:
            mapped_vals = vals

        limit = np.percentile(np.abs(mapped_vals), 95) / zoom
        x_range = (-limit, limit)
        y_range = (-limit, limit)
        
        title_base = self.current_base_name.replace('_', '\\_')
        
        if mode != 'Transcendental':
            # Map vals to pixel coords
            ix = ((mapped_vals.real - x_range[0]) / (x_range[1] - x_range[0]) * (res - 1)).astype(int)
            iy = ((mapped_vals.imag - y_range[0]) / (y_range[1] - y_range[0]) * (res - 1)).astype(int)
            mask = (ix >= 0) & (ix < res) & (iy >= 0) & (iy < res)

            cmap = self.current_cmap
            if mode == 'Complexity':
                metric = np.sum(np.abs(coeffs), axis=1)
                label = 'Min L1 Coefficient Complexity'
                agg_func = np.minimum.at
                init_val = np.inf
            elif mode == 'Dominant':
                terms = np.abs(coeffs * powers)
                metric = np.argmax(terms, axis=1)
                label = 'Dominant Power Index'
                agg_func = np.maximum.at
                init_val = -1.0
            elif mode == 'Mean Coeff':
                metric = np.mean(np.abs(coeffs), axis=1)
                label = 'Mean Coefficient Magnitude'
                agg_func = np.maximum.at
                init_val = 0.0
            elif mode == 'Phase':
                metric = np.angle(vals)
                label = 'Argument (Phase)'
                agg_func = np.maximum.at
                init_val = -np.pi - 1.0
            elif mode == 'Radial':
                metric = np.abs(vals)
                label = 'Radial Distance'
                agg_func = np.maximum.at
                init_val = -1.0

            Z_metric = np.full((res, res), init_val)
            agg_func(Z_metric, (iy[mask], ix[mask]), metric[mask].astype(float))

            if mode == 'Complexity':
                Z_metric[np.isinf(Z_metric)] = np.nan
            else:
                Z_metric[Z_metric == init_val] = np.nan

            im = self.ax.imshow(Z_metric, extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                                origin='lower', cmap=cmap)
            if self.colorbar:
                self.colorbar.remove()
            self.colorbar = self.fig.colorbar(im, ax=self.ax, label=label)
            self.colorbar.ax.yaxis.label.set_color('white')
            self.colorbar.ax.tick_params(colors='white')

            self.ax.set_title(rf"Transcendental Analysis: $\mathbb{{Q}}({title_base})$" + "\n" +
                              f"{label} across the extension",
                              color='white', fontsize=14)
        else:
            if self.colorbar:
                self.colorbar.remove()
                self.colorbar = None
                
            H, xedges, yedges = np.histogram2d(mapped_vals.real, mapped_vals.imag, bins=res,
                                               range=[[x_range[0], x_range[1]], [y_range[0], y_range[1]]])
            Z_final = gaussian_filter(H.T, sigma=sigma)

            if self.show_gradient:
                gy, gx = np.gradient(Z_final)
                Z_final = np.sqrt(gx**2 + gy**2)
                norm = plt.Normalize(vmin=0, vmax=np.percentile(Z_final, 99))
            else:
                vmin = 1.0
                vmax = max(vmin + 1.0, np.max(Z_final))
                norm = LogNorm(vmin=vmin, vmax=vmax)

            self.ax.imshow(Z_final, extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
                            origin='lower', cmap=self.current_cmap, norm=norm)
            self.ax.set_title(rf"Transcendental Field Interference: $\mathbb{{Q}}({title_base})$" + "\n" +
                              rf"Density of elements $\sum_{{i=0}}^{{{deg}}} c_i \cdot {title_base}^i$",
                              color='white', fontsize=14)

        # Plot Rational Convergents if base is real
        if not np.iscomplex(base) and self.show_overlays:
            convs = self.get_convergents(base)
            self.ax.scatter(convs, np.zeros_like(convs), color='red', s=50, marker='x',
                            label='Rational Convergents', zorder=5)
            self.ax.legend(facecolor='#111111', labelcolor='white')

        if self.show_overlays:
            self.draw_reference_overlays(x_range, y_range)

    def draw_reference_overlays(self, x_range, y_range):
        # Unit Circle
        t = np.linspace(0, 2*np.pi, 500)
        circle = np.exp(1j * t)

        # Apply transformation if needed
        if self.coord_system == 'Log-Polar':
            mapped_circle = np.log(np.abs(circle) + 1e-9) + 1j * np.angle(circle)
        elif self.coord_system == 'Reciprocal':
            mapped_circle = 1.0 / circle
        else:
            mapped_circle = circle

        self.ax.plot(mapped_circle.real, mapped_circle.imag, color='white',
                     linestyle='--', alpha=0.3, lw=1, label='Unit Circle')

        # Grid axes
        self.ax.axhline(0, color='white', alpha=0.1, lw=0.5)
        self.ax.axvline(0, color='white', alpha=0.1, lw=0.5)

        # Roots of Unity (e.g., 8-th roots)
        n_roots = 8
        roots = np.exp(1j * 2 * np.pi * np.arange(n_roots) / n_roots)
        if self.coord_system == 'Log-Polar':
            mapped_roots = np.log(np.abs(roots) + 1e-9) + 1j * np.angle(roots)
        elif self.coord_system == 'Reciprocal':
            mapped_roots = 1.0 / roots
        else:
            mapped_roots = roots

        self.ax.scatter(mapped_roots.real, mapped_roots.imag, color='cyan', s=10, alpha=0.5, label=f'{n_roots}-th Roots')

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
        v6 = self.s6.val if self.interactive else 0.0
        
        if self.mode == 'Algebraic':
            self.draw_algebraic(v1, v2, v3)
        elif self.mode == 'Finite':
            self.draw_finite(v1, v2)
        else:
            self.draw_transcendental(v1, v2, v3, mode=self.mode, zoom=v4, num_samples=v5, rotation=v6)
            
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
