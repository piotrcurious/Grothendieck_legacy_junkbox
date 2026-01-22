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

        tk.Button(sidebar, text="Load Image", command=self.load_image, font=('Arial', 10, 'bold')).pack(fill=tk.X, pady=(0, 10))
        
        self.params = {}
        # Updated configurations to match the new C++ struct and main() argv
        configs = [
            # Label, Key, Min, Max, Default
            ("Spatial Block Size", "spatial_blk", 4, 64, 16),
            ("Spatial Stride", "spatial_stride", 1, 32, 8),
            ("Spectral Block Size", "spectral_blk", 4, 32, 8),
            ("Spectral Coeffs", "spectral_coeffs", 1, 32, 4),
            ("Quantization Step", "quant", 0.1, 20.0, 2.0),
        ]

        for label, key, v_min, v_max, v_def in configs:
            tk.Label(sidebar, text=label, font=('Arial', 8, 'bold')).pack(anchor=tk.W, pady=(5,0))
            res = 0.1 if isinstance(v_def, float) else 1
            s = tk.Scale(sidebar, from_=v_min, to=v_max, resolution=res, orient=tk.HORIZONTAL)
            s.set(v_def)
            s.pack(fill=tk.X)
            self.params[key] = s

        tk.Button(sidebar, text="Process", bg="#2ecc71", fg="white", font=('Arial', 10, 'bold'), command=self.process).pack(fill=tk.X, pady=20)
        
        self.stats = tk.Label(sidebar, text="Ratio: --\nSize: --", justify=tk.LEFT, font=('Courier', 10))
        self.stats.pack(anchor=tk.W)
        
        tk.Label(sidebar, text="\n[Controls]\nWheel: Zoom\nLeft-Click: Pan", fg="gray", justify=tk.LEFT).pack(side=tk.BOTTOM)

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

        for c in (self.can_orig, self.can_reco):
            c.bind("<ButtonPress-1>", self.on_pan_start)
            c.bind("<B1-Motion>", self.on_pan_move)
            c.bind("<MouseWheel>", self.on_zoom)
            c.bind("<Button-4>", self.on_zoom)
            c.bind("<Button-5>", self.on_zoom)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.pgm")])
        if path:
            self.input_path = path
            self.scale = 1.0
            self.pan_x = 0
            self.pan_y = 0
            self.process()

    def process(self):
        if not self.input_path: return
        
        # Prepare image (convert to grayscale and save as PGM)
        img = Image.open(self.input_path).convert('L')
        img.save(self.temp_pgm)
        self.img_orig = img

        # Construct command based on: <c/d> <in> <out> [spatial_block stride spectral_block coeffs quant]
        cmd = [
            self.exe, 
            "c", 
            self.temp_pgm, 
            self.output_bin, 
            str(int(self.params["spatial_blk"].get())),
            str(int(self.params["spatial_stride"].get())),
            str(int(self.params["spectral_blk"].get())),
            str(int(self.params["spectral_coeffs"].get())),
            str(self.params["quant"].get())
        ]
        
        try:
            # Run Compression
            if subprocess.run(cmd).returncode == 0:
                # Run Decompression: <mode> <in> <out>
                subprocess.run([self.exe, "d", self.output_bin, self.decoded_pgm])
                
                if os.path.exists(self.decoded_pgm):
                    self.img_reco = Image.open(self.decoded_pgm)
                    
                    # Calculate Stats
                    orig_sz = os.path.getsize(self.temp_pgm) / 1024
                    comp_sz = os.path.getsize(self.output_bin) / 1024
                    ratio = orig_sz / comp_sz if comp_sz > 0 else 0
                    self.stats.config(text=f"Ratio: {ratio:.2f}x\nSize: {comp_sz:.1f} KB")
                    
                    self.redraw()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run executable:\n{e}")

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
        if event.num == 4 or event.delta > 0: factor = 1.2
        else: factor = 0.8
        self.scale *= factor
        self.redraw()

    def redraw(self):
        if self.img_orig is None: return
        
        w, h = self.img_orig.size
        nw, nh = max(1, int(w * self.scale)), max(1, int(h * self.scale))
        
        # Source
        res_orig = self.img_orig.resize((nw, nh), Image.NEAREST)
        self.tk_orig = ImageTk.PhotoImage(res_orig)
        self.can_orig.delete("all")
        self.can_orig.create_image(self.pan_x, self.pan_y, anchor=tk.NW, image=self.tk_orig)

        # Recompressed
        if self.img_reco:
            res_reco = self.img_reco.resize((nw, nh), Image.NEAREST)
            self.tk_reco = ImageTk.PhotoImage(res_reco)
            self.can_reco.delete("all")
            self.can_reco.create_image(self.pan_x, self.pan_y, anchor=tk.NW, image=self.tk_reco)

if __name__ == "__main__":
    root = tk.Tk()
    app = GDHCVisualizer(root)
    root.mainloop()
