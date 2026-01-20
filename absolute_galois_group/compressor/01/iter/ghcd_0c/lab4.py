import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os
from PIL import Image, ImageTk

class GDHCVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("GDHC Tunable - Sync Zoom/Pan")
        self.root.geometry("1400x800")
        
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
        sidebar = tk.Frame(self.root, width=250, padx=15, pady=15, relief=tk.RAISED, borderwidth=1)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)

        tk.Button(sidebar, text="Load Image", command=self.load_image).pack(fill=tk.X)
        
        self.params = {}
        configs = [
            ("Dict Block Size", "dict_blk", 4, 64, 16),
            ("Dict Entries", "dict_ent", 8, 512, 64),
            ("DCT Block Size", "dct_blk", 4, 32, 8),
            ("DCT Quant Step", "dct_step", 0.5, 20.0, 4.0),
            ("DCT Coeffs (NxN)", "dct_coeffs", 1, 32, 2)
        ]

        for label, key, v_min, v_max, v_def in configs:
            tk.Label(sidebar, text=label, font=('Arial', 8, 'bold')).pack(anchor=tk.W, pady=(5,0))
            s = tk.Scale(sidebar, from_=v_min, to=v_max, resolution=0.5 if isinstance(v_def, float) else 1, orient=tk.HORIZONTAL)
            s.set(v_def); s.pack(fill=tk.X); self.params[key] = s

        tk.Button(sidebar, text="Process", bg="#2ecc71", command=self.process).pack(fill=tk.X, pady=10)
        self.stats = tk.Label(sidebar, text="Ratio: --\nSize: --", justify=tk.LEFT, font=('Courier', 9))
        self.stats.pack(anchor=tk.W)
        tk.Label(sidebar, text="\n[Controls]\nWheel: Zoom\nLeft-Click: Pan", fg="gray").pack()

        # Dual Viewport
        self.view_container = tk.Frame(self.root, bg="#1a1a1a")
        self.view_container.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # Left: Source
        self.f_left = tk.Frame(self.view_container, bg="#1a1a1a")
        self.f_left.place(relx=0, rely=0, relwidth=0.5, relheight=1)
        tk.Label(self.f_left, text="SOURCE", fg="white", bg="black").pack(side=tk.TOP, fill=tk.X)
        self.can_orig = tk.Canvas(self.f_left, bg="#1a1a1a", highlightthickness=0)
        self.can_orig.pack(expand=True, fill=tk.BOTH)

        # Right: Recompressed
        self.f_right = tk.Frame(self.view_container, bg="#1a1a1a")
        self.f_right.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)
        tk.Label(self.f_right, text="RECOMPRESSED", fg="white", bg="black").pack(side=tk.TOP, fill=tk.X)
        self.can_reco = tk.Canvas(self.f_right, bg="#1a1a1a", highlightthickness=0)
        self.can_reco.pack(expand=True, fill=tk.BOTH)

        # Bindings (Bind to both canvases for convenience)
        for c in (self.can_orig, self.can_reco):
            c.bind("<ButtonPress-1>", self.on_pan_start)
            c.bind("<B1-Motion>", self.on_pan_move)
            c.bind("<MouseWheel>", self.on_zoom)  # Windows/MacOS
            c.bind("<Button-4>", self.on_zoom)    # Linux scroll up
            c.bind("<Button-5>", self.on_zoom)    # Linux scroll down

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.input_path = path
            self.scale = 1.0 # Reset view
            self.pan_x = 0
            self.pan_y = 0
            self.process()

    def process(self):
        if not self.input_path: return
        
        d_blk = self.params["dict_blk"].get()
        # Crop to multiple of dict block size
        img = Image.open(self.input_path).convert('L')
        w, h = img.size
        img = img.crop((0, 0, w - (w % d_blk), h - (h % d_blk)))
        img.save(self.temp_pgm)
        self.img_orig = img

        # Run C++
        cmd = [self.exe, "c", self.temp_pgm, self.output_bin, str(d_blk), 
               str(self.params["dict_ent"].get()), str(self.params["dct_blk"].get()), 
               str(self.params["dct_step"].get()), str(self.params["dct_coeffs"].get())]
        
        if subprocess.run(cmd).returncode == 0:
            subprocess.run([self.exe, "d", self.output_bin, self.decoded_pgm])
            self.img_reco = Image.open(self.decoded_pgm)
            
            # Stats
            orig_sz = os.path.getsize(self.temp_pgm) / 1024
            comp_sz = os.path.getsize(self.output_bin) / 1024
            self.stats.config(text=f"Ratio: {orig_sz/comp_sz:.2f}x\nSize: {comp_sz:.1f} KB")
            
            self.redraw()

    # --- Navigation Logic ---
    def on_pan_start(self, event):
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

    def on_pan_move(self, event):
        self.pan_x += (event.x - self.last_mouse_x)
        self.pan_y += (event.y - self.last_mouse_y)
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y
        self.redraw()

    def on_zoom(self, event):
        # Determine zoom direction
        if event.num == 4 or event.delta > 0: factor = 1.1
        else: factor = 0.9
        
        self.scale *= factor
        self.redraw()

    def redraw(self):
        if self.img_orig is None: return
        
        # Calculate scaled dimensions
        w, h = self.img_orig.size
        nw, nh = int(w * self.scale), int(h * self.scale)
        
        # Redraw Source
        res_orig = self.img_orig.resize((nw, nh), Image.NEAREST)
        self.tk_orig = ImageTk.PhotoImage(res_orig)
        self.can_orig.delete("all")
        self.can_orig.create_image(self.pan_x, self.pan_y, anchor=tk.NW, image=self.tk_orig)

        # Redraw Recompressed
        if self.img_reco:
            res_reco = self.img_reco.resize((nw, nh), Image.NEAREST)
            self.tk_reco = ImageTk.PhotoImage(res_reco)
            self.can_reco.delete("all")
            self.can_reco.create_image(self.pan_x, self.pan_y, anchor=tk.NW, image=self.tk_reco)

if __name__ == "__main__":
    root = tk.Tk()
    app = GDHCVisualizer(root)
    root.mainloop()
