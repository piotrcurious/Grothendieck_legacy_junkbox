import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os
from PIL import Image, ImageTk

class GDHCVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("GDHC Tunable Compressor Interface")
        self.root.geometry("1100x700")
        
        # Paths
        self.input_path = ""
        self.temp_pgm = "input_temp.pgm"
        self.output_bin = "output.gdhc"
        self.decoded_pgm = "decoded.pgm"
        self.exe = "./shared_comp"

        self.setup_ui()

    def setup_ui(self):
        # Sidebar for controls
        sidebar = tk.Frame(self.root, width=280, padx=15, pady=15, relief=tk.RAISED, borderwidth=1)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)

        tk.Button(sidebar, text="1. Load Image", command=self.load_image, bg="#e1e1e1").pack(fill=tk.X, pady=5)

        # Parameter Sliders
        self.params = {}
        
        configs = [
            ("Dict Block Size", "dict_blk", 4, 64, 16),
            ("Dict Entries", "dict_ent", 8, 512, 64),
            ("DCT Block Size", "dct_blk", 4, 32, 8),
            ("DCT Quant Step", "dct_step", 0.5, 20.0, 4.0),
            ("DCT Coeffs (NxN)", "dct_coeffs", 1, 32, 2)
        ]

        for label, key, v_min, v_max, v_def in configs:
            tk.Label(sidebar, text=label, font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(10, 0))
            res = 0.5 if isinstance(v_def, float) else 1
            s = tk.Scale(sidebar, from_=v_min, to=v_max, resolution=res, orient=tk.HORIZONTAL)
            s.set(v_def)
            s.pack(fill=tk.X)
            self.params[key] = s

        tk.Button(sidebar, text="2. Compress & Preview", bg="#2ecc71", fg="white", 
                  command=self.process, font=('Arial', 10, 'bold'), pady=10).pack(fill=tk.X, pady=20)

        self.stats = tk.Label(sidebar, text="Stats:\nRatio: --\nSize: --", justify=tk.LEFT, font=('Courier', 9))
        self.stats.pack(anchor=tk.W)

        # Preview Area
        self.canvas = tk.Canvas(self.root, bg="#2c3e50")
        self.canvas.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

    def load_image(self):
        path = filedialog.askopenfilename()
        if path:
            self.input_path = path
            self.process()

    def process(self):
        if not self.input_path:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        # Fetch values
        d_blk = self.params["dict_blk"].get()
        d_ent = self.params["dict_ent"].get()
        c_blk = self.params["dct_blk"].get()
        c_step = self.params["dct_step"].get()
        c_coeffs = self.params["dct_coeffs"].get()

        # Validation per C++ logic
        if c_coeffs > c_blk:
            messagebox.showerror("Error", "DCT Coeffs cannot exceed DCT Block Size")
            return

        # Prepare Image (Crop to multiple of Dictionary Block Size)
        img = Image.open(self.input_path).convert('L')
        w, h = img.size
        img = img.crop((0, 0, w - (w % d_blk), h - (h % d_blk)))
        img.save(self.temp_pgm)

        # Run C++ Compressor
        # Order: mode in out dict_blk dict_ent dct_blk dct_step dct_coeffs
        cmd_c = [
            self.exe, "c", self.temp_pgm, self.output_bin,
            str(d_blk), str(d_ent), str(c_blk), str(c_step), str(c_coeffs)
        ]
        
        result = subprocess.run(cmd_c, capture_output=True, text=True)
        if result.returncode != 0:
            messagebox.showerror("C++ Error", result.stderr)
            return

        # Run C++ Decompressor
        subprocess.run([self.exe, "d", self.output_bin, self.decoded_pgm], capture_output=True)

        # Update Metrics and UI
        self.update_display()

    def update_display(self):
        orig_sz = os.path.getsize(self.temp_pgm) / 1024
        comp_sz = os.path.getsize(self.output_bin) / 1024
        ratio = orig_sz / comp_sz if comp_sz > 0 else 0
        
        self.stats.config(text=f"Original: {orig_sz:.1f} KB\nCompressed: {comp_sz:.1f} KB\nRatio: {ratio:.2f}x")

        # Visuals
        orig = Image.open(self.temp_pgm)
        reco = Image.open(self.decoded_pgm)
        
        # Create side-by-side
        combined = Image.new('L', (orig.width + reco.width + 20, max(orig.height, reco.height)), color=40)
        combined.paste(orig, (0, 0))
        combined.paste(reco, (orig.width + 20, 0))

        # Scale for window
        win_w, win_h = 800, 600
        combined.thumbnail((win_w, win_h))

        self.tk_img = ImageTk.PhotoImage(combined)
        self.canvas.delete("all")
        self.canvas.create_image(self.canvas.winfo_width()//2, self.canvas.winfo_height()//2, 
                                 anchor=tk.CENTER, image=self.tk_img)

if __name__ == "__main__":
    root = tk.Tk()
    # Make sure binary exists
    if not os.path.exists("./shared_comp"):
        print("Error: ./shared_comp binary not found. Please compile the C++ code first.")
    else:
        app = GDHCVisualizer(root)
        root.mainloop()
