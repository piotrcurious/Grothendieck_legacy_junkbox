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
        sidebar = tk.Frame(self.root, width=260, padx=15, pady=15, relief=tk.RAISED, borderwidth=1)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)

        tk.Button(sidebar, text="📂 Load Image", font=('Arial', 10, 'bold'), command=self.load_image).pack(fill=tk.X, pady=(0, 15))
        
        self.params = {}
        # Matches your CompressionConfig defaults and main() order
        configs = [
            # Label, Key, Min, Max, Default
            ("Spatial Block", "spatial_blk", 4, 64, 16),
            ("Stride (Overlap)", "stride", 1, 32, 8),
            ("Spectral Block", "spectral_blk", 4, 64, 8),
            ("Coeffs (Spectral)", "coeffs", 1, 64, 4),
            ("Quant Step", "quant", 0.5, 20.0, 2.0)
        ]

        for label, key, v_min, v_max, v_def in configs:
            tk.Label(sidebar, text=label, font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(5,0))
            res = 0.5 if isinstance(v_def, float) else 1
            s = tk.Scale(sidebar, from_=v_min, to=v_max, resolution=res, orient=tk.HORIZONTAL)
            s.set(v_def)
            s.pack(fill=tk.X)
            self.params[key] = s

        tk.Button(sidebar, text="⚡ Process & Compare", bg="#27ae60", fg="white", 
                  font=('Arial', 10, 'bold'), command=self.process).pack(fill=tk.X, pady=20)
        
        self.stats = tk.Label(sidebar, text="Ratio: --\nSize: --", justify=tk.LEFT, font=('Courier', 10))
        self.stats.pack(anchor=tk.W)
        
        tk.Label(sidebar, text="\n[Controls]\nWheel: Zoom\nDrag: Pan", fg="gray", justify=tk.LEFT).pack(side=tk.BOTTOM)

        # Dual Viewport
        self.view_container = tk.Frame(self.root, bg="#1a1a1a")
        self.view_container.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # Left View
        self.f_left = tk.Frame(self.view_container, bg="#1a1a1a")
        self.f_left.place(relx=0, rely=0, relwidth=0.5, relheight=1)
        tk.Label(self.f_left, text="SOURCE", fg="#3498db", bg="black", font=('Arial', 9, 'bold')).pack(side=tk.TOP, fill=tk.X)
        self.can_orig = tk.Canvas(self.f_left, bg="#1a1a1a", highlightthickness=0)
        self.can_orig.pack(expand=True, fill=tk.BOTH)

        # Right View
        self.f_right = tk.Frame(self.view_container, bg="#1a1a1a")
        self.f_right.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)
        tk.Label(self.f_right, text="GDHC RECONSTRUCTED", fg="#e67e22", bg="black", font=('Arial', 9, 'bold')).pack(side=tk.TOP, fill=tk.X)
        self.can_reco = tk.Canvas(self.f_right, bg="#1a1a1a", highlightthickness=0)
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
            # Convert to PGM immediately for standard processing
            img = Image.open(self.input_path).convert('L')
            img.save(self.temp_pgm)
            self.img_orig = img
            self.process()

    def process(self):
        if not self.input_path: return
        
        # Command mapped to your updated main():
        # <mode> <in> <out> [spatial_block] [stride] [spectral_block] [coeffs] [quant]
        cmd = [
            self.exe, "c", self.temp_pgm, self.output_bin,
            str(int(self.params["spatial_blk"].get())),
            str(int(self.params["stride"].get())),
            str(int(self.params["spectral_blk"].get())),
            str(int(self.params["coeffs"].get())),
            str(self.params["quant"].get())
        ]
        
        try:
            # 1. Compress
            ret_c = subprocess.run(cmd, capture_output=True, text=True)
            if ret_c.returncode != 0:
                messagebox.showerror("Compression Error", ret_c.stderr)
                return

            # 2. Decompress
            ret_d = subprocess.run([self.exe, "d", self.output_bin, self.decoded_pgm], capture_output=True, text=True)
            if ret_d.returncode == 0 and os.path.exists(self.decoded_pgm):
                self.img_reco = Image.open(self.decoded_pgm)
                
                # Stats
                orig_sz = os.path.getsize(self.temp_pgm) / 1024
                comp_sz = os.path.getsize(self.output_bin) / 1024
                ratio = orig_sz / comp_sz if comp_sz > 0 else 1
                self.stats.config(text=f"Ratio: {ratio:.2f}x\nSize: {comp_sz:.1f} KB")
                self.redraw()
            else:
                messagebox.showerror("Decompression Error", ret_d.stderr)

        except Exception as e:
            messagebox.showerror("System Error", str(e))

    # --- Navigation Logic ---
    def on_pan_start(self, event):
        self.last_mouse_x, self.last_mouse_y = event.x, event.y

    def on_pan_move(self, event):
        self.pan_x += (event.x - self.last_mouse_x)
        self.pan_y += (event.y - self.last_mouse_y)
        self.last_mouse_x, self.last_mouse_y = event.x, event.y
        self.redraw()

    def on_zoom(self, event):
        factor = 1.2 if (event.num == 4 or event.delta > 0) else 0.8
        self.scale *= factor
        self.redraw()

    def redraw(self):
        if self.img_orig is None: return
        
        w, h = self.img_orig.size
        nw, nh = max(1, int(w * self.scale)), max(1, int(h * self.scale))
        
        # Source Redraw
        res_orig = self.img_orig.resize((nw, nh), Image.NEAREST)
        self.tk_orig = ImageTk.PhotoImage(res_orig)
        self.can_orig.delete("all")
        self.can_orig.create_image(self.pan_x, self.pan_y, anchor=tk.NW, image=self.tk_orig)

        # Result Redraw
        if self.img_reco:
            res_reco = self.img_reco.resize((nw, nh), Image.NEAREST)
            self.tk_reco = ImageTk.PhotoImage(res_reco)
            self.can_reco.delete("all")
            self.can_reco.create_image(self.pan_x, self.pan_y, anchor=tk.NW, image=self.tk_reco)

if __name__ == "__main__":
    root = tk.Tk()
    app = GDHCVisualizer(root)
    root.mainloop()
