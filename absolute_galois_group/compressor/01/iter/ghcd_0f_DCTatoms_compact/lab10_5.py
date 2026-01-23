import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os
from PIL import Image, ImageTk

class GDHCVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("GDHC v10.5 - Galois-DCT Hybrid Compressor Visualizer")
        self.root.geometry("1400x950")
        
        # --- State ---
        self.input_path = ""
        self.temp_ppm = "input_temp.ppm"
        self.output_bin = "output.gdhc"  # Updated extension
        self.decoded_ppm = "decoded.ppm"
        self.exe = "./shared_comp"
        
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
        sidebar = tk.Frame(self.root, width=380, padx=15, pady=15, relief=tk.RAISED, borderwidth=1)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(sidebar, text="GDHC v10.5 CONTROLS", font=('Arial', 12, 'bold')).pack(pady=(0, 10))
        
        tk.Button(sidebar, text="📁 Open Image", font=('Arial', 10), command=self.load_image).pack(fill=tk.X, pady=5)
        
        slider_container = tk.LabelFrame(sidebar, text="Compression Parameters", padx=5, pady=5)
        slider_container.pack(fill=tk.X, pady=10)

        self.params = {}
        # Updated to match GDHC v10.5 specifications
        # Format: (Label, Flag, Min, Max, Default, Row, Col)
        configs = [
            ("Base Block", "--base-block", 4, 64, 16, 0, 0),
            ("Resid Block", "--resid-block", 2, 32, 8, 0, 1),
            ("Base Dict", "--base-entries", 16, 4096, 512, 2, 0),
            ("Resid Dict", "--resid-entries", 16, 4096, 1024, 2, 1),
            ("Lambda Base", "--lambda-base", 0.0, 1000.0, 150.0, 4, 0),
            ("Lambda Resid", "--lambda-resid", 0.0, 500.0, 50.0, 4, 1),
            ("Min Var Base", "--base-var", 0.0, 500.0, 30.0, 6, 0),
            ("Min Var Resid", "--resid-var", 0.0, 200.0, 5.0, 6, 1),
            ("Max Atoms", "--max-atoms", 1, 3, 3, 8, 0),
            ("Overlap Stride", "--overlap", 0.0, 1.0, 0.5, 8, 1),
            ("Crypto Trials", "--crypto-trials", 1, 512, 32, 10, 0)
        ]

        for label, flag, v_min, v_max, v_def, r, c in configs:
            lbl = tk.Label(slider_container, text=label, font=('Arial', 8, 'bold'))
            lbl.grid(row=r, column=c, sticky=tk.W, padx=5, pady=(5, 0))
            
            is_float = isinstance(v_def, float)
            res = 0.01 if flag == "--overlap" else (0.1 if is_float else 1)
            
            s = tk.Scale(slider_container, from_=v_min, to=v_max, resolution=res, 
                         orient=tk.HORIZONTAL, font=('Arial', 8), length=150)
            s.set(v_def)
            s.grid(row=r+1, column=c, padx=5, pady=(0, 5))
            self.params[flag] = s

        tk.Button(sidebar, text="▶ Run Compression", bg="#27ae60", fg="white", 
                  font=('Arial', 10, 'bold'), command=self.process).pack(fill=tk.X, pady=10)
        
        self.stats_frame = tk.LabelFrame(sidebar, text="Statistics", padx=10, pady=10)
        self.stats_frame.pack(fill=tk.X)
        self.stats_lbl = tk.Label(self.stats_frame, text="Ratio: --\nSize: --\nPSNR: --", 
                                  justify=tk.LEFT, font=('Courier', 9))
        self.stats_lbl.pack(anchor=tk.W)
        
        help_text = "[Controls]\n• Scroll: Zoom\n• Left Click Drag: Pan\n• Resize window to fit"
        tk.Label(sidebar, text=help_text, fg="#7f8c8d", justify=tk.LEFT, font=('Arial', 8)).pack(side=tk.BOTTOM)

        # Viewport
        self.view_container = tk.Frame(self.root, bg="#121212")
        self.view_container.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        self.f_left = tk.Frame(self.view_container, bg="#121212")
        self.f_left.place(relx=0, rely=0, relwidth=0.5, relheight=1)
        tk.Label(self.f_left, text="ORIGINAL", fg="white", bg="#2980b9", font=('Arial', 9, 'bold')).pack(side=tk.TOP, fill=tk.X)
        self.can_orig = tk.Canvas(self.f_left, bg="#121212", highlightthickness=0)
        self.can_orig.pack(expand=True, fill=tk.BOTH)

        self.f_right = tk.Frame(self.view_container, bg="#121212")
        self.f_right.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)
        tk.Label(self.f_right, text="RECONSTRUCTED (GDHC)", fg="white", bg="#d35400", font=('Arial', 9, 'bold')).pack(side=tk.TOP, fill=tk.X)
        self.can_reco = tk.Canvas(self.f_right, bg="#121212", highlightthickness=0)
        self.can_reco.pack(expand=True, fill=tk.BOTH)

        for c in (self.can_orig, self.can_reco):
            c.bind("<ButtonPress-1>", self.on_pan_start)
            c.bind("<B1-Motion>", self.on_pan_move)
            c.bind("<MouseWheel>", self.on_zoom)
            c.bind("<Button-4>", self.on_zoom) 
            c.bind("<Button-5>", self.on_zoom)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.ppm *.jpg *.png *.bmp")])
        if path:
            self.input_path = path
            img = Image.open(self.input_path).convert('RGB')
            img.save(self.temp_ppm)
            self.img_orig = img
            self.scale = 1.0
            self.pan_x = 0
            self.pan_y = 0
            self.process()

    def update_stats(self):
        if not os.path.exists(self.temp_ppm) or not os.path.exists(self.output_bin):
            return
            
        try:
            orig_size = os.path.getsize(self.temp_ppm)
            comp_size = os.path.getsize(self.output_bin)
            
            if comp_size > 0:
                ratio = orig_size / comp_size
                size_kb = comp_size / 1024
                
                stats_text = f"Ratio: {ratio:.2f}x\n"
                stats_text += f"Size:  {size_kb:.1f} KB\n"
                stats_text += f"Orig:  {orig_size/1024:.1f} KB"
                self.stats_lbl.config(text=stats_text)
        except Exception as e:
            print(f"Stats error: {e}")

    def process(self):
        if not self.input_path: return
        
        # Build Compression Command
        cmd_comp = [self.exe, "c", self.temp_ppm, self.output_bin]
        for flag, slider in self.params.items():
            cmd_comp.append(flag)
            val = slider.get()
            # Ensure proper formatting for integers vs floats
            if flag in ["--base-block", "--resid-block", "--base-entries", "--resid-entries", "--max-atoms", "--crypto-trials"]:
                cmd_comp.append(str(int(val)))
            else:
                cmd_comp.append(f"{val:.2f}")
        
        try:
            # Execute Compression
            proc_c = subprocess.run(cmd_comp, capture_output=True, text=True)
            if proc_c.returncode != 0:
                messagebox.showerror("GDHC Error", proc_c.stderr or proc_c.stdout)
                return

            # Execute Decompression
            cmd_decomp = [self.exe, "d", self.output_bin, self.decoded_ppm]
            proc_d = subprocess.run(cmd_decomp, capture_output=True, text=True)
            
            if proc_d.returncode == 0 and os.path.exists(self.decoded_ppm):
                self.img_reco = Image.open(self.decoded_ppm)
                self.update_stats()
                self.redraw()
            else:
                messagebox.showerror("Decompression Error", proc_d.stderr or proc_d.stdout)

        except Exception as e:
            messagebox.showerror("System Error", f"Process failed: {e}")

    # --- Interaction Handlers ---
    def on_pan_start(self, event):
        self.last_mouse_x, self.last_mouse_y = event.x, event.y

    def on_pan_move(self, event):
        self.pan_x += (event.x - self.last_mouse_x)
        self.pan_y += (event.y - self.last_mouse_y)
        self.last_mouse_x, self.last_mouse_y = event.x, event.y
        self.redraw()

    def on_zoom(self, event):
        if event.num == 4 or event.delta > 0: factor = 1.1
        else: factor = 0.9
        self.scale *= factor
        self.scale = max(0.05, min(self.scale, 50.0))
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
