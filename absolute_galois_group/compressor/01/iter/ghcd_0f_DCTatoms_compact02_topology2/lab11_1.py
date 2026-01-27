import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os
from PIL import Image, ImageTk

class GDHCVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("GDHC v11.1 - VLC Compact Spectral Hybrid")
        self.root.geometry("1500x950")
        
        # --- State ---
        self.input_path = ""
        self.temp_ppm = "input_temp.ppm"
        self.output_bin = "output.gdhc"
        self.decoded_ppm = "decoded.ppm"
        self.exe = "./shared_comp" # Ensure this matches your compiled binary name
        
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
        sidebar = tk.Frame(self.root, width=400, padx=15, pady=15, relief=tk.RAISED, borderwidth=1)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(sidebar, text="GDHC v11.1 CONTROLS", font=('Arial', 14, 'bold')).pack(pady=(0, 10))
        
        tk.Button(sidebar, text="📁 Open Image", font=('Arial', 10, 'bold'), command=self.load_image).pack(fill=tk.X, pady=5)
        
        # Scrollable container for many sliders
        container = tk.LabelFrame(sidebar, text="Compression Parameters (v11.1)", padx=10, pady=10)
        container.pack(fill=tk.BOTH, expand=True, pady=10)

        self.params = {}
        
        # Configuration Groups: (Label, Flag, Min, Max, Default, Row_Offset, Column)
        # Type logic: Flags ending in 's', 'e', or '2' (except lr2/rv2) are often ints. 
        # Based on your C++: stoi for bs/rs/rs2/be/re/re2, stof for lb/lr/lr2/bv/rv/rv2
        
        groups = [
            ("--- BASE LAYER ---", [
                ("Block Size", "--bs", 4, 64, 16, 0),
                ("Entries", "--be", 16, 2048, 256, 2),
                ("Lambda", "--lb", 0.0, 2000.0, 250.0, 4),
                ("Min Var", "--bv", 0.0, 500.0, 40.0, 6),
            ], 0),
            ("--- RESID LAYER 1 ---", [
                ("Block Size", "--rs", 2, 32, 8, 0),
                ("Entries", "--re", 16, 2048, 512, 2),
                ("Lambda", "--lr", 0.0, 1000.0, 120.0, 4),
                ("Min Var", "--rv", 0.0, 200.0, 10.0, 6),
            ], 1),
            ("--- RESID LAYER 2 ---", [
                ("Block Size", "--rs2", 2, 32, 8, 0),
                ("Entries", "--re2", 16, 2048, 512, 2),
                ("Lambda", "--lr2", 0.0, 1000.0, 120.0, 4),
                ("Min Var", "--rv2", 0.0, 200.0, 10.0, 6),
            ], 2)
        ]

        for title, configs, col in groups:
            tk.Label(container, text=title, font=('Arial', 8, 'bold'), fg="#34495e").grid(row=0, column=col, pady=(10, 5), sticky=tk.W)
            for label, flag, v_min, v_max, v_def, r_off in configs:
                tk.Label(container, text=label, font=('Arial', 7)).grid(row=r_off+1, column=col, sticky=tk.W, padx=5)
                
                is_float = isinstance(v_def, float)
                res = 0.1 if is_float else 1
                
                s = tk.Scale(container, from_=v_min, to=v_max, resolution=res, 
                             orient=tk.HORIZONTAL, font=('Arial', 8), length=110)
                s.set(v_def)
                s.grid(row=r_off+2, column=col, padx=5, pady=(0, 5))
                self.params[flag] = s

        tk.Button(sidebar, text="▶ Run Spectral Hybrid", bg="#27ae60", fg="white", 
                  font=('Arial', 11, 'bold'), command=self.process).pack(fill=tk.X, pady=10)
        
        self.stats_frame = tk.LabelFrame(sidebar, text="Statistics", padx=10, pady=10)
        self.stats_frame.pack(fill=tk.X)
        self.stats_lbl = tk.Label(self.stats_frame, text="Ratio: --\nSize: --\nInput: --", 
                                  justify=tk.LEFT, font=('Courier', 10))
        self.stats_lbl.pack(anchor=tk.W)
        
        help_text = "[Controls]\n• Scroll: Zoom\n• Drag: Pan\n• V11.1: 3-layer Hybrid"
        tk.Label(sidebar, text=help_text, fg="#7f8c8d", justify=tk.LEFT, font=('Arial', 8)).pack(side=tk.BOTTOM)

        # Viewport
        self.view_container = tk.Frame(self.root, bg="#121212")
        self.view_container.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # Panels for comparison
        self.f_left = tk.Frame(self.view_container, bg="#121212")
        self.f_left.place(relx=0, rely=0, relwidth=0.5, relheight=1)
        tk.Label(self.f_left, text="SOURCE", fg="white", bg="#2980b9", font=('Arial', 9, 'bold')).pack(side=tk.TOP, fill=tk.X)
        self.can_orig = tk.Canvas(self.f_left, bg="#121212", highlightthickness=0)
        self.can_orig.pack(expand=True, fill=tk.BOTH)

        self.f_right = tk.Frame(self.view_container, bg="#121212")
        self.f_right.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)
        tk.Label(self.f_right, text="GDHC v11.1 HYBRID", fg="white", bg="#c0392b", font=('Arial', 9, 'bold')).pack(side=tk.TOP, fill=tk.X)
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
                stats_text = f"Ratio: {ratio:.2f}x\n"
                stats_text += f"Size:  {comp_size/1024:.2f} KB\n"
                stats_text += f"Input: {orig_size/1024:.2f} KB"
                self.stats_lbl.config(text=stats_text)
        except Exception as e:
            print(f"Stats calculation error: {e}")

    def process(self):
        if not self.input_path: return
        
        # Build Command for v11.1
        # Usage: <exe> c <in> <out> [opts]
        cmd_comp = [self.exe, "c", self.temp_ppm, self.output_bin]
        
        for flag, slider in self.params.items():
            val = slider.get()
            cmd_comp.append(flag)
            # Match C++ stoi (int) vs stof (float)
            if any(x in flag for x in ["bs", "rs", "be", "re"]):
                cmd_comp.append(str(int(val)))
            else:
                cmd_comp.append(f"{val:.4f}")
        
        try:
            # Execute Compression
            print(f"Executing: {' '.join(cmd_comp)}")
            proc_c = subprocess.run(cmd_comp, capture_output=True, text=True)
            if proc_c.returncode != 0:
                messagebox.showerror("GDHC Engine Error", f"Compression failed:\n{proc_c.stderr}")
                return

            # Execute Decompression
            # Usage: <exe> d <in> <out>
            cmd_decomp = [self.exe, "d", self.output_bin, self.decoded_ppm]
            proc_d = subprocess.run(cmd_decomp, capture_output=True, text=True)
            
            if proc_d.returncode == 0 and os.path.exists(self.decoded_ppm):
                self.img_reco = Image.open(self.decoded_ppm)
                self.update_stats()
                self.redraw()
            else:
                messagebox.showerror("GDHC Engine Error", f"Decompression failed:\n{proc_d.stderr}")

        except Exception as e:
            messagebox.showerror("System Error", f"Failed to run process: {e}")

    def on_pan_start(self, event):
        self.last_mouse_x, self.last_mouse_y = event.x, event.y

    def on_pan_move(self, event):
        self.pan_x += (event.x - self.last_mouse_x)
        self.pan_y += (event.y - self.last_mouse_y)
        self.last_mouse_x, self.last_mouse_y = event.x, event.y
        self.redraw()

    def on_zoom(self, event):
        if event.num == 4 or event.delta > 0: factor = 1.2
        else: factor = 0.8
        self.scale *= factor
        self.scale = max(0.01, min(self.scale, 100.0))
        self.redraw()

    def redraw(self):
        if self.img_orig is None: return
        w, h = self.img_orig.size
        nw, nh = int(w * self.scale), int(h * self.scale)
        if nw < 1 or nh < 1: return

        # Synchronized view redraw
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
