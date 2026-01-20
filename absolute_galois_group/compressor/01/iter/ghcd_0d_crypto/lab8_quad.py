import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os
from PIL import Image, ImageTk

class GDHCVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("GDHC Tunable Visualizer")
        self.root.geometry("1400x950") 
        
        # --- State ---
        self.input_path = ""
        self.temp_pgm = "input_temp.pgm"
        self.output_bin = "output.gdhc"
        self.decoded_pgm = "decoded.pgm"
        self.exe = "./gdhc_tunable"
        
        # Navigation State
        self.scale = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        self.img_orig = None
        self.img_reco = None

        self.setup_ui()

    def setup_ui(self):
        # Sidebar
        sidebar = tk.Frame(self.root, width=300, padx=20, pady=20, relief=tk.RAISED, borderwidth=1)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(sidebar, text="GDHC CONTROLS", font=('Arial', 12, 'bold')).pack(pady=(0, 15))
        
        tk.Button(sidebar, text="📁 Open Image", font=('Arial', 10), command=self.load_image).pack(fill=tk.X, pady=5)
        
        # Parameters mapped to the C++ interface (argc 5 through 14)
        self.params = {}
        
        # Numeric Configs
        configs = [
            ("Spatial Block Size", "spatial_block", 4, 64, 16),
            ("Spatial Stride",     "spatial_stride", 1, 64, 8),
            ("Spectral Block Size","spectral_block", 4, 64, 8),
            ("Spectral Coeffs",    "spectral_coeffs", 1, 64, 4),
            ("Quantization Step",  "quant_step", 0.1, 25.0, 2.0),
            ("Spatial Entries",    "spatial_entries", 8, 1024, 256),
            ("Spectral Entries",   "spectral_entries", 8, 1024, 128),
            ("Min Variance",       "min_variance", 0.0, 10.0, 0.5),
            ("RDO Lambda",         "rdo_lambda", 0.0, 50.0, 0.1)
        ]

        for label, key, v_min, v_max, v_def in configs:
            tk.Label(sidebar, text=label, font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(8, 0))
            res = 0.01 if key in ["min_variance", "rdo_lambda"] else (0.1 if isinstance(v_def, float) else 1)
            s = tk.Scale(sidebar, from_=v_min, to=v_max, resolution=res, orient=tk.HORIZONTAL)
            s.set(v_def)
            s.pack(fill=tk.X)
            self.params[key] = s

        # Boolean Config: Use Quadtree
        self.use_quadtree_var = tk.IntVar(value=1)
        tk.Checkbutton(sidebar, text="Use Quadtree", variable=self.use_quadtree_var, font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=10)

        tk.Button(sidebar, text="▶ Run Compression", bg="#27ae60", fg="white", 
                  font=('Arial', 10, 'bold'), command=self.process).pack(fill=tk.X, pady=15)
        
        # Stats Display
        self.stats_frame = tk.LabelFrame(sidebar, text="Statistics", padx=10, pady=10)
        self.stats_frame.pack(fill=tk.X)
        self.stats = tk.Label(self.stats_frame, text="Ratio: --\nSize: --", justify=tk.LEFT, font=('Courier', 10))
        self.stats.pack(anchor=tk.W)
        
        help_text = "\n[View Controls]\n• Scroll: Zoom\n• Left Click Drag: Pan"
        tk.Label(sidebar, text=help_text, fg="#7f8c8d", justify=tk.LEFT, font=('Arial', 8)).pack(side=tk.BOTTOM)

        # Viewport Area
        self.view_container = tk.Frame(self.root, bg="#121212")
        self.view_container.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        self.f_left = tk.Frame(self.view_container, bg="#121212")
        self.f_left.place(relx=0, rely=0, relwidth=0.5, relheight=1)
        tk.Label(self.f_left, text="ORIGINAL", fg="white", bg="#2980b9").pack(side=tk.TOP, fill=tk.X)
        self.can_orig = tk.Canvas(self.f_left, bg="#121212", highlightthickness=0)
        self.can_orig.pack(expand=True, fill=tk.BOTH)

        self.f_right = tk.Frame(self.view_container, bg="#121212")
        self.f_right.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)
        tk.Label(self.f_right, text="RECONSTRUCTED", fg="white", bg="#d35400").pack(side=tk.TOP, fill=tk.X)
        self.can_reco = tk.Canvas(self.f_right, bg="#121212", highlightthickness=0)
        self.can_reco.pack(expand=True, fill=tk.BOTH)

        for c in (self.can_orig, self.can_reco):
            c.bind("<ButtonPress-1>", self.on_pan_start)
            c.bind("<B1-Motion>", self.on_pan_move)
            c.bind("<MouseWheel>", self.on_zoom)
            c.bind("<Button-4>", self.on_zoom) 
            c.bind("<Button-5>", self.on_zoom)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.input_path = path
            img = Image.open(self.input_path).convert('L')
            img.save(self.temp_pgm)
            self.img_orig = img
            self.scale = 1.0
            self.pan_x = 0
            self.pan_y = 0
            self.process()

    def process(self):
        if not self.input_path: return
        
        # Updated Command line arguments mapping (argc 1 to 14):
        # 1: mode, 2: input, 3: output, 4: spatial_block, 5: spatial_stride,
        # 6: spectral_block, 7: spectral_coeffs, 8: quant_step, 9: spatial_entries,
        # 10: spectral_entries, 11: min_variance, 12: use_quadtree, 13: rdo_lambda
        
        cmd = [
            self.exe, 
            "c", 
            self.temp_pgm, 
            self.output_bin,
            str(int(self.params["spatial_block"].get())),
            str(int(self.params["spatial_stride"].get())),
            str(int(self.params["spectral_block"].get())),
            str(int(self.params["spectral_coeffs"].get())),
            str(self.params["quant_step"].get()),
            str(int(self.params["spatial_entries"].get())),
            str(int(self.params["spectral_entries"].get())),
            str(self.params["min_variance"].get()),
            str(self.use_quadtree_var.get()),      # 0 or 1
            str(self.params["rdo_lambda"].get())
        ]
        
        try:
            proc_c = subprocess.run(cmd, capture_output=True, text=True)
            if proc_c.returncode != 0:
                messagebox.showerror("Compression Error", proc_c.stderr or proc_c.stdout)
                return

            proc_d = subprocess.run([self.exe, "d", self.output_bin, self.decoded_pgm], 
                                     capture_output=True, text=True)
            
            if proc_d.returncode == 0 and os.path.exists(self.decoded_pgm):
                self.img_reco = Image.open(self.decoded_pgm)
                orig_sz = os.path.getsize(self.temp_pgm) / 1024
                comp_sz = os.path.getsize(self.output_bin) / 1024
                ratio = orig_sz / comp_sz if comp_sz > 0 else 1.0
                self.stats.config(text=f"Ratio: {ratio:.2f}x\nSize: {comp_sz:.1f} KB")
                self.redraw()
            else:
                messagebox.showerror("Decompression Error", proc_d.stderr or proc_d.stdout)

        except Exception as e:
            messagebox.showerror("System Error", f"Execution failed: {e}")

    # Navigation Logic remains unchanged
    def on_pan_start(self, event):
        self.last_mouse_x, self.last_mouse_y = event.x, event.y

    def on_pan_move(self, event):
        self.pan_x += (event.x - self.last_mouse_x)
        self.pan_y += (event.y - self.last_mouse_y)
        self.last_mouse_x, self.last_mouse_y = event.x, event.y
        self.redraw()

    def on_zoom(self, event):
        if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0): factor = 1.2
        else: factor = 0.8
        self.scale *= factor
        self.scale = max(self.scale, 0.05)
        self.redraw()

    def redraw(self):
        if self.img_orig is None: return
        w, h = self.img_orig.size
        nw, nh = int(w * self.scale), int(h * self.scale)
        if nw < 1 or nh < 1: return

        res_orig = self.img_orig.resize((nw, nh), Image.NEAREST)
        self.tk_orig = ImageTk.PhotoImage(res_orig)
        self.can_orig.delete("all")
        self.can_orig.create_image(self.pan_x, self.pan_y, anchor=tk.NW, image=self.tk_orig)

        if self.img_reco:
            res_reco = self.img_reco.resize((nw, nh), Image.NEAREST)
            self.tk_reco = ImageTk.PhotoImage(res_reco)
            self.can_reco.delete("all")
            self.can_reco.create_image(self.pan_x, self.pan_y, anchor=tk.NW, image=self.tk_reco)

if __name__ == "__main__":
    root = tk.Tk()
    app = GDHCVisualizer(root)
    root.mainloop()
