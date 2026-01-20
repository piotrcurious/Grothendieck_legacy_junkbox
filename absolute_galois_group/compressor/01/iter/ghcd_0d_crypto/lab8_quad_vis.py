import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import subprocess
import os
from PIL import Image, ImageTk

class GDHCVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("GDHC Tunable Visualizer")
        self.root.geometry("1400x850")
        self.root.configure(bg="#2c3e50")
        
        # --- State ---
        self.input_path = ""
        self.temp_pgm = "input_temp.pgm"
        self.output_bin = "output.gdhc"
        self.decoded_pgm = "decoded.pgm"
        self.exe = "./gdhc_tunable"
        
        self.scale = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        self.img_orig = None
        self.img_reco = None

        self.setup_ui()

    def setup_ui(self):
        # Sidebar with fixed width
        sidebar = tk.Frame(self.root, width=320, padx=10, pady=10, bg="#f8f9fa", relief=tk.FLAT)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="GDHC TUNING", font=('Arial', 11, 'bold'), bg="#f8f9fa").pack(pady=(0, 10))
        
        tk.Button(sidebar, text="📁 Open Image", command=self.load_image, 
                  relief=tk.GROOVE, cursor="hand2").pack(fill=tk.X, pady=2)
        
        # Parameter Container
        scroll_frame = tk.Frame(sidebar, bg="#f8f9fa")
        scroll_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.params = {}
        
        # Sections for logical grouping
        sections = [
            ("Spatial Block", [
                ("Size", "spatial_block", 4, 64, 16, 1),
                ("Stride", "spatial_stride", 1, 64, 8, 1),
            ]),
            ("Spectral Block", [
                ("Size", "spectral_block", 4, 64, 8, 1),
                ("Coeffs", "spectral_coeffs", 1, 64, 4, 1),
            ]),
            ("Quantization", [
                ("Step", "quant_step", 0.1, 25.0, 2.0, 0.1),
                ("Lambda", "rdo_lambda", 0.0, 20.0, 0.1, 0.05),
            ]),
            ("Dictionary", [
                ("Spatial Ent", "spatial_entries", 8, 1024, 256, 8),
                ("Spectral Ent", "spectral_entries", 8, 1024, 128, 8),
                ("Min Var", "min_variance", 0.0, 5.0, 0.5, 0.01),
            ])
        ]

        for sec_name, items in sections:
            # Section Header
            lbl = tk.Label(scroll_frame, text=sec_name.upper(), font=('Arial', 7, 'bold'), 
                           bg="#f8f9fa", fg="#7f8c8d")
            lbl.pack(anchor=tk.W, pady=(8, 2))
            
            for label, key, v_min, v_max, v_def, res in items:
                row = tk.Frame(scroll_frame, bg="#f8f9fa")
                row.pack(fill=tk.X, pady=1)
                
                # Small label on the left
                tk.Label(row, text=label, font=('Arial', 8), bg="#f8f9fa", width=10, anchor=tk.W).pack(side=tk.LEFT)
                
                # Compact Scale
                s = tk.Scale(row, from_=v_min, to=v_max, resolution=res, orient=tk.HORIZONTAL,
                             showvalue=True, font=('Arial', 7), bg="#f8f9fa", highlightthickness=0,
                             length=150, sliderlength=15, bd=0)
                s.set(v_def)
                s.pack(side=tk.RIGHT, fill=tk.X, expand=True)
                self.params[key] = s

        # Boolean Row
        bool_row = tk.Frame(scroll_frame, bg="#f8f9fa")
        bool_row.pack(fill=tk.X, pady=10)
        self.use_quadtree_var = tk.IntVar(value=1)
        tk.Checkbutton(bool_row, text="Use Quadtree Search", variable=self.use_quadtree_var, 
                       font=('Arial', 8, 'bold'), bg="#f8f9fa").pack(side=tk.LEFT)

        # Action Button
        tk.Button(sidebar, text="▶ RUN COMPRESSION", bg="#2ecc71", fg="white", 
                  font=('Arial', 9, 'bold'), relief=tk.FLAT, command=self.process,
                  pady=8, cursor="hand2").pack(fill=tk.X, pady=10)
        
        # Stats
        self.stats_frame = tk.Frame(sidebar, bg="#ecf0f1", padx=5, pady=5)
        self.stats_frame.pack(fill=tk.X)
        self.stats = tk.Label(self.stats_frame, text="Ratio: -- | Size: --", 
                              bg="#ecf0f1", font=('Courier', 9))
        self.stats.pack()

        # Viewport Area
        self.view_container = tk.Frame(self.root, bg="#121212")
        self.view_container.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # Left/Right Viewports
        self.can_orig = self.create_viewport(self.view_container, "ORIGINAL", 0, "#2980b9")
        self.can_reco = self.create_viewport(self.view_container, "RECONSTRUCTED", 0.5, "#d35400")

    def create_viewport(self, parent, title, x, color):
        frame = tk.Frame(parent, bg="#121212")
        frame.place(relx=x, rely=0, relwidth=0.5, relheight=1)
        tk.Label(frame, text=title, fg="white", bg=color, font=('Arial', 8, 'bold')).pack(side=tk.TOP, fill=tk.X)
        canvas = tk.Canvas(frame, bg="#121212", highlightthickness=0)
        canvas.pack(expand=True, fill=tk.BOTH)
        canvas.bind("<ButtonPress-1>", self.on_pan_start)
        canvas.bind("<B1-Motion>", self.on_pan_move)
        canvas.bind("<MouseWheel>", self.on_zoom)
        return canvas

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.pgm *.bmp")])
        if path:
            self.input_path = path
            img = Image.open(path).convert('L')
            img.save(self.temp_pgm)
            self.img_orig = img
            self.scale = 1.0
            self.pan_x = 0; self.pan_y = 0
            self.process()

    def process(self):
        if not self.input_path: return
        
        # Match argc 5 - 14 logic
        cmd = [
            self.exe, "c", self.temp_pgm, self.output_bin,
            str(int(self.params["spatial_block"].get())),
            str(int(self.params["spatial_stride"].get())),
            str(int(self.params["spectral_block"].get())),
            str(int(self.params["spectral_coeffs"].get())),
            f"{self.params['quant_step'].get():.4f}",
            str(int(self.params["spatial_entries"].get())),
            str(int(self.params["spectral_entries"].get())),
            f"{self.params['min_variance'].get():.4f}",
            str(self.use_quadtree_var.get()),
            f"{self.params['rdo_lambda'].get():.4f}"
        ]
        
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                messagebox.showerror("Error", res.stderr or res.stdout)
                return

            dec = subprocess.run([self.exe, "d", self.output_bin, self.decoded_pgm], capture_output=True)
            if dec.returncode == 0 and os.path.exists(self.decoded_pgm):
                self.img_reco = Image.open(self.decoded_pgm)
                isz = os.path.getsize(self.temp_pgm) / 1024
                osz = os.path.getsize(self.output_bin) / 1024
                self.stats.config(text=f"Ratio: {isz/osz:.2f}x | Size: {osz:.1f} KB")
                self.redraw()
        except Exception as e:
            messagebox.showerror("System Error", str(e))

    def on_pan_start(self, event):
        self.last_mouse_x, self.last_mouse_y = event.x, event.y

    def on_pan_move(self, event):
        self.pan_x += (event.x - self.last_mouse_x)
        self.pan_y += (event.y - self.last_mouse_y)
        self.last_mouse_x, self.last_mouse_y = event.x, event.y
        self.redraw()

    def on_zoom(self, event):
        factor = 1.1 if (event.delta > 0 or event.num == 4) else 0.9
        self.scale = max(0.1, min(self.scale * factor, 20.0))
        self.redraw()

    def redraw(self):
        if not self.img_orig: return
        w, h = self.img_orig.size
        nw, nh = int(w * self.scale), int(h * self.scale)
        if nw < 2 or nh < 2: return

        # Original
        io = self.img_orig.resize((nw, nh), Image.NEAREST)
        self.tk_o = ImageTk.PhotoImage(io)
        self.can_orig.delete("all")
        self.can_orig.create_image(self.pan_x, self.pan_y, anchor=tk.NW, image=self.tk_o)

        # Reconstructed
        if self.img_reco:
            ir = self.img_reco.resize((nw, nh), Image.NEAREST)
            self.tk_r = ImageTk.PhotoImage(ir)
            self.can_reco.delete("all")
            self.can_reco.create_image(self.pan_x, self.pan_y, anchor=tk.NW, image=self.tk_r)

if __name__ == "__main__":
    root = tk.Tk()
    app = GDHCVisualizer(root)
    root.mainloop()
