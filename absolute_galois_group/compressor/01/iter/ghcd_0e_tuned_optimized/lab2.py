import tkinter as tk
from tkinter import filedialog, ttk
import subprocess
import os
from PIL import Image, ImageTk

class GDHCVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("GDHC Hybrid Compressor Tuner")
        
        # --- State ---
        self.input_path = ""
        self.temp_pgm = "input_temp.pgm"
        self.output_bin = "output.gdhc"
        self.decoded_pgm = "decoded.pgm"
        
        # --- UI Layout ---
        self.setup_sidebar()
        self.setup_preview()

    def setup_sidebar(self):
        sidebar = tk.Frame(self.root, width=250, padx=10, pady=10, relief=tk.RIDGE, borderwidth=2)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)

        tk.Button(sidebar, text="Load Image", command=self.load_image).pack(fill=tk.X, pady=5)

        # Dictionary Entries Slider
        tk.Label(sidebar, text="Dict Entries (L1)").pack(anchor=tk.W)
        self.dict_entries = tk.Scale(sidebar, from_=8, to=256, orient=tk.HORIZONTAL)
        self.dict_entries.set(64)
        self.dict_entries.pack(fill=tk.X)

        # DCT Step Slider
        tk.Label(sidebar, text="DCT Quant Step (L2)").pack(anchor=tk.W, pady=(10, 0))
        self.dct_step = tk.Scale(sidebar, from_=0.5, to=20.0, resolution=0.5, orient=tk.HORIZONTAL)
        self.dct_step.set(4.0)
        self.dct_step.pack(fill=tk.X)

        # DCT Coeffs Slider
        tk.Label(sidebar, text="DCT Coeffs (NxN)").pack(anchor=tk.W, pady=(10, 0))
        self.dct_coeffs = tk.Scale(sidebar, from_=1, to=8, orient=tk.HORIZONTAL)
        self.dct_coeffs.set(2)
        self.dct_coeffs.pack(fill=tk.X)

        tk.Button(sidebar, text="Compress & Preview", bg="#4CAF50", fg="white", 
                  command=self.process, font=('Arial', 10, 'bold')).pack(fill=tk.X, pady=20)

        self.stats_label = tk.Label(sidebar, text="Ratio: 0x\nSize: 0 KB", justify=tk.LEFT)
        self.stats_label.pack(anchor=tk.W)

    def setup_preview(self):
        self.preview_frame = tk.Frame(self.root, bg="gray")
        self.preview_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        
        self.canvas = tk.Canvas(self.preview_frame)
        self.canvas.pack(expand=True, fill=tk.BOTH)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.pgm *.bmp")])
        if path:
            self.input_path = path
            # Pre-convert to PGM for the C++ tool
            img = Image.open(path).convert('L')
            w, h = img.size
            img = img.crop((0, 0, w - (w % 16), h - (h % 16)))
            img.save(self.temp_pgm)
            self.process()

    def process(self):
        if not self.input_path: return

        # 1. Run C++ Compression
        cmd_c = [
            "./shared_comp", "c", self.temp_pgm, self.output_bin,
            str(self.dict_entries.get()), str(self.dct_step.get()), str(self.dct_coeffs.get())
        ]
        subprocess.run(cmd_c, capture_output=True)

        # 2. Run C++ Decompression
        cmd_d = ["./shared_comp", "d", self.output_bin, self.decoded_pgm]
        subprocess.run(cmd_d, capture_output=True)

        # 3. Update Stats
        orig_sz = os.path.getsize(self.temp_pgm) / 1024
        comp_sz = os.path.getsize(self.output_bin) / 1024
        ratio = orig_sz / comp_sz if comp_sz > 0 else 0
        self.stats_label.config(text=f"Original: {orig_sz:.1f} KB\nCompressed: {comp_sz:.1f} KB\nRatio: {ratio:.2f}x")

        # 4. Update Display
        self.display_results()

    def display_results(self):
        # Load images for display
        orig = Image.open(self.temp_pgm)
        reco = Image.open(self.decoded_pgm)
        
        # Simple side-by-side stitch
        total_w = orig.width + reco.width + 10
        max_h = max(orig.height, reco.height)
        combined = Image.new('L', (total_w, max_h), color=128)
        combined.paste(orig, (0, 0))
        combined.paste(reco, (orig.width + 10, 0))

        # Resize to fit screen if necessary
        screen_h = self.root.winfo_screenheight() - 200
        if combined.height > screen_h:
            factor = screen_h / combined.height
            combined = combined.resize((int(combined.width * factor), int(combined.height * factor)))

        self.tk_img = ImageTk.PhotoImage(combined)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

if __name__ == "__main__":
    root = tk.Tk()
    app = GDHCVisualizer(root)
    root.mainloop()
