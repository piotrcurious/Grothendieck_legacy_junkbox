import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os
from PIL import Image, ImageTk

class GDHCVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Shared Compression Tunable Visualizer")
        self.root.geometry("1400x950")
        
        # --- State ---
        self.input_path = ""
        self.temp_ppm = "input_temp.ppm"
        self.output_bin = "output.bin"
        self.decoded_ppm = "decoded.ppm"
        self.exe = "./shared_comp"
        
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
        sidebar = tk.Frame(self.root, width=320, padx=20, pady=20, relief=tk.RAISED, borderwidth=1)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(sidebar, text="COMPRESSION CONTROLS", font=('Arial', 12, 'bold')).pack(pady=(0, 15))
        
        tk.Button(sidebar, text="📁 Open Image (PPM/JPG/PNG)", font=('Arial', 10), command=self.load_image).pack(fill=tk.X, pady=5)
        
        # Parameters mapped to the new shared_comp interface
        self.params = {}
        configs = [
            # Label, Flag, Min, Max, Default
            ("Base Block Size", "--base-block", 4, 64, 16),
            ("Residual Block Size", "--resid-block", 2, 16, 4),
            ("Base Dictionary Size", "--base-entries", 8, 1024, 128),
            ("Residual Dictionary Size", "--resid-entries", 8, 1024, 256),
            ("Lambda Base (RDO)", "--lambda-base", 0.0, 1000.0, 150.0),
            ("Lambda Residual (RDO)", "--lambda-resid", 0.0, 500.0, 20.0),
            ("Min Var Base", "--base-var", 0.0, 500.0, 100.0),
            ("Min Var Resid", "--resid-var", 0.0, 200.0, 10.0),
            ("Crypto Trials", "--crypto-trials", 1, 256, 64)
        ]

        # Container for sliders to allow scrolling if needed
        slider_container = tk.Frame(sidebar)
        slider_container.pack(fill=tk.BOTH, expand=True)

        for label, flag, v_min, v_max, v_def in configs:
            tk.Label(slider_container, text=label, font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=(8, 0))
            
            # Use float resolution for lambda and variance
            is_float = isinstance(v_def, float)
            res = 0.1 if is_float else 1
            
            s = tk.Scale(slider_container, from_=v_min, to=v_max, resolution=res, orient=tk.HORIZONTAL)
            s.set(v_def)
            s.pack(fill=tk.X)
            self.params[flag] = s

        tk.Button(sidebar, text="▶ Run Compression", bg="#27ae60", fg="white", 
                  font=('Arial', 10, 'bold'), command=self.process).pack(fill=tk.X, pady=20)
        
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

        # Left: Original
        self.f_left = tk.Frame(self.view_container, bg="#121212")
        self.f_left.place(relx=0, rely=0, relwidth=0.5, relheight=1)
        tk.Label(self.f_left, text="ORIGINAL", fg="white", bg="#2980b9").pack(side=tk.TOP, fill=tk.X)
        self.can_orig = tk.Canvas(self.f_left, bg="#121212", highlightthickness=0)
        self.can_orig.pack(expand=True, fill=tk.BOTH)

        # Right: Reconstructed
        self.f_right = tk.Frame(self.view_container, bg="#121212")
        self.f_right.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)
        tk.Label(self.f_right, text="RECONSTRUCTED", fg="white", bg="#d35400").pack(side=tk.TOP, fill=tk.X)
        self.can_reco = tk.Canvas(self.f_right, bg="#121212", highlightthickness=0)
        self.can_reco.pack(expand=True, fill=tk.BOTH)

        # Input Bindings
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
            # Convert to RGB PPM (P6) as the tool expects PPM P6
            img = Image.open(self.input_path).convert('RGB')
            img.save(self.temp_ppm)
            self.img_orig = img
            self.scale = 1.0
            self.pan_x = 0
            self.pan_y = 0
            self.process()

    def process(self):
        if not self.input_path: 
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        
        # Build command: ./shared_comp c input.ppm output.bin [options]
        cmd_comp = [self.exe, "c", self.temp_ppm, self.output_bin]
        
        # Append all parameters from sliders
        for flag, slider in self.params.items():
            cmd_comp.append(flag)
            val = slider.get()
            # Pass as int if it has no fractional part, else float
            cmd_comp.append(str(int(val)) if val == int(val) else f"{val:.2f}")
        
        try:
            # Run Compression
            proc_c = subprocess.run(cmd_comp, capture_output=True, text=True)
            if proc_c.returncode != 0:
                messagebox.showerror("Compression Error", proc_c.stderr or proc_c.stdout)
                return

            # Run Decompression: ./shared_comp d output.bin decoded.ppm
            cmd_decomp = [self.exe, "d", self.output_bin, self.decoded_ppm]
            proc_d = subprocess.run(cmd_decomp, capture_output=True, text=True)
            
            if proc_d.returncode == 0 and os.path.exists(self.decoded_ppm):
                self.img_reco = Image.open(self.decoded_ppm)
                
                # Update Stats
                orig_sz = os.path.getsize(self.temp_ppm) / 1024
                comp_sz = os.path.getsize(self.output_bin) / 1024
                ratio = orig_sz / comp_sz if comp_sz > 0 else 1.0
                self.stats.config(text=f"Ratio: {ratio:.2f}x\nSize: {comp_sz:.1f} KB")
                
                self.redraw()
            else:
                messagebox.showerror("Decompression Error", proc_d.stderr or proc_d.stdout)

        except Exception as e:
            messagebox.showerror("System Error", f"Execution failed: {e}")

    # --- Navigation Logic ---
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
        self.scale = max(0.01, min(self.scale, 50.0))
        self.redraw()

    def redraw(self):
        if self.img_orig is None: return
        
        w, h = self.img_orig.size
        nw, nh = int(w * self.scale), int(h * self.scale)
        if nw < 1 or nh < 1: return

        # Original View
        res_orig = self.img_orig.resize((nw, nh), Image.NEAREST)
        self.tk_orig = ImageTk.PhotoImage(res_orig)
        self.can_orig.delete("all")
        self.can_orig.create_image(self.pan_x, self.pan_y, anchor=tk.NW, image=self.tk_orig)

        # Reconstructed View
        if self.img_reco:
            res_reco = self.img_reco.resize((nw, nh), Image.NEAREST)
            self.tk_reco = ImageTk.PhotoImage(res_reco)
            self.can_reco.delete("all")
            self.can_reco.create_image(self.pan_x, self.pan_y, anchor=tk.NW, image=self.tk_reco)

if __name__ == "__main__":
    root = tk.Tk()
    app = GDHCVisualizer(root)
    root.mainloop()
