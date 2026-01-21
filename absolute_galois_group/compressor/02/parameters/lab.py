import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import subprocess
import os
from PIL import Image, ImageTk

class ScrollableMessageDialog(tk.Toplevel):
    """Custom dialog to display scrollable, selectable text (e.g., error logs)."""
    def __init__(self, parent, title, message):
        super().__init__(parent)
        self.title(title)
        self.geometry("600x400")
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.padx = 10
        self.pady = 10

        label = tk.Label(self, text=title, font=('Arial', 10, 'bold'))
        label.pack(pady=(10, 5))

        # Container for text and scrollbar
        frame = tk.Frame(self)
        frame.pack(expand=True, fill=tk.BOTH, padx=self.padx, pady=self.pady)

        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.text_area = tk.Text(frame, wrap=tk.WORD, yscrollcommand=scrollbar.set, font=('Courier', 9))
        self.text_area.insert(tk.END, message)
        self.text_area.config(state=tk.DISABLED)  # Read-only but selectable
        self.text_area.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        scrollbar.config(command=self.text_area.yview)

        btn_close = tk.Button(self, text="Close", command=self.destroy, width=10)
        btn_close.pack(pady=(0, 10))
        
        # Focus the window
        self.transient(parent)
        self.grab_set()
        parent.wait_window(self)

class GDHCVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("GDHC v11.4 - Advanced Hybrid Compressor Visualizer")
        self.root.geometry("1450x980")
        
        # --- State ---
        self.input_path = ""
        self.temp_ppm = "input_temp.ppm"
        self.output_gdhc = "output.gdhc"
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

    def show_log(self, title, message):
        """Helper to show the selectable message dialog."""
        ScrollableMessageDialog(self.root, title, message)

    def setup_ui(self):
        # Sidebar
        sidebar = tk.Frame(self.root, width=380, padx=10, pady=10, relief=tk.RAISED, borderwidth=1)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(sidebar, text="GDHC v11.4 ENGINE", font=('Arial', 12, 'bold')).pack(pady=(0, 10))
        
        btn_open = tk.Button(sidebar, text="📁 Load Source Image", font=('Arial', 10), command=self.load_image)
        btn_open.pack(fill=tk.X, pady=5)
        
        # Scrollable area for many parameters
        param_canvas = tk.Canvas(sidebar, highlightthickness=0)
        scrollbar = ttk.Scrollbar(sidebar, orient="vertical", command=param_canvas.yview)
        self.scroll_frame = tk.Frame(param_canvas)

        self.scroll_frame.bind(
            "<Configure>",
            lambda e: param_canvas.configure(scrollregion=param_canvas.bbox("all"))
        )

        param_canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        param_canvas.configure(yscrollcommand=scrollbar.set)

        param_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.params = {}
        self.flags_bool = {}

        # --- Section: Structure ---
        self.add_section_label("Block Structure")
        self.add_slider("Base Size", "--base-size", 4, 64, 16)
        self.add_slider("Resid Size", "--resid-size", 2, 16, 4)
        
        # --- Section: Dictionary ---
        self.add_section_label("Dictionary Management")
        self.add_slider("Base Entries", "--base-entries", 8, 2048, 128)
        self.add_slider("Resid Entries", "--resid-entries", 8, 2048, 256)
        self.add_slider("K-SVD Iters", "--ksvd-iters", 1, 32, 8)
        self.add_slider("OMP Sparsity", "--omp-sparsity", 1, 8, 2)

        # --- Section: RDO & Variance ---
        self.add_section_label("Rate-Distortion Optimization")
        self.add_slider("Lambda Base", "--lambda-base", 0.0, 1000.0, 100.0)
        self.add_slider("Lambda Resid", "--lambda-resid", 0.0, 500.0, 12.0)
        self.add_slider("Var Base", "--base-var", 0.0, 500.0, 40.0)
        self.add_slider("Var Resid", "--resid-var", 0.0, 200.0, 4.0)

        # --- Section: Geometric & Refinement ---
        self.add_section_label("Geometry & Refinement")
        self.add_slider("Crypto Trials", "--crypto-trials", 1, 2048, 512)
        self.add_slider("Atom Trials", "--atom-trials", 1, 20, 6)
        self.add_slider("Deform Lambda", "--deformation-lambda", 0.0, 50.0, 8.0)
        self.add_slider("Gradient Weight", "--gradient-weight", 0.0, 1.0, 0.7)
        
        # Boolean refinement
        self.var_2atom = tk.BooleanVar(value=True)
        chk_2atom = tk.Checkbutton(self.scroll_frame, text="Enable 2-Atom Refinement", variable=self.var_2atom)
        chk_2atom.pack(anchor=tk.W, padx=10, pady=5)

        # Action Buttons
        tk.Button(sidebar, text="▶ Run Optimization Cycle", bg="#27ae60", fg="white", 
                  font=('Arial', 10, 'bold'), height=2, command=self.process).pack(fill=tk.X, pady=10)
        
        self.stats_lbl = tk.Label(sidebar, text="Ratio: -- | Size: --", relief=tk.SUNKEN, bd=1, font=('Courier', 10), pady=5)
        self.stats_lbl.pack(fill=tk.X)

        # Viewport Area
        self.view_container = tk.Frame(self.root, bg="#121212")
        self.view_container.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # Views
        self.f_left = tk.Frame(self.view_container, bg="#121212")
        self.f_left.place(relx=0, rely=0, relwidth=0.5, relheight=1)
        tk.Label(self.f_left, text="SOURCE (PPM/BMP)", fg="white", bg="#2980b9", font=('Arial', 9, 'bold')).pack(side=tk.TOP, fill=tk.X)
        self.can_orig = tk.Canvas(self.f_left, bg="#121212", highlightthickness=0)
        self.can_orig.pack(expand=True, fill=tk.BOTH)

        self.f_right = tk.Frame(self.view_container, bg="#121212")
        self.f_right.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)
        tk.Label(self.f_right, text="RECONSTRUCTED (GDHC v11.4)", fg="white", bg="#d35400", font=('Arial', 9, 'bold')).pack(side=tk.TOP, fill=tk.X)
        self.can_reco = tk.Canvas(self.f_right, bg="#121212", highlightthickness=0)
        self.can_reco.pack(expand=True, fill=tk.BOTH)

        for c in (self.can_orig, self.can_reco):
            c.bind("<ButtonPress-1>", self.on_pan_start)
            c.bind("<B1-Motion>", self.on_pan_move)
            c.bind("<MouseWheel>", self.on_zoom)

    def add_section_label(self, text):
        lbl = tk.Label(self.scroll_frame, text=text.upper(), font=('Arial', 8, 'bold'), fg="#34495e")
        lbl.pack(anchor=tk.W, padx=5, pady=(15, 2))

    def add_slider(self, label, flag, v_min, v_max, v_def):
        container = tk.Frame(self.scroll_frame)
        container.pack(fill=tk.X, padx=10)
        
        lbl = tk.Label(container, text=label, font=('Arial', 8))
        lbl.pack(side=tk.TOP, anchor=tk.W)
        
        res = 0.01 if isinstance(v_def, float) else 1
        s = tk.Scale(container, from_=v_min, to=v_max, resolution=res, 
                     orient=tk.HORIZONTAL, font=('Arial', 8), length=300)
        s.set(v_def)
        s.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))
        self.params[flag] = s

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.ppm *.jpg *.png *.bmp")])
        if path:
            self.input_path = path
            try:
                img = Image.open(self.input_path).convert('RGB')
                img.save(self.temp_ppm)
                self.img_orig = img
                self.redraw()
            except Exception as e:
                self.show_log("Loading Error", str(e))

    def process(self):
        if not self.img_orig: 
            messagebox.showwarning("Warning", "Load an image first.")
            return
        
        # Build Command for v11.4
        cmd_comp = [self.exe, "-i", self.temp_ppm, "-o", self.output_gdhc, "-m", "compress"]
        
        # Add numeric params
        for flag, slider in self.params.items():
            val = slider.get()
            cmd_comp.append(flag)
            cmd_comp.append(str(int(val)) if val == int(val) else f"{val:.3f}")
        
        # Add toggle params
        cmd_comp.append("--enable-2atom" if self.var_2atom.get() else "--disable-2atom")
        
        try:
            # Compression
            res_c = subprocess.run(cmd_comp, capture_output=True, text=True)
            if res_c.returncode != 0:
                # Capture and show output so it can be copied
                self.show_log("GDHC Compression Failed", f"Return Code: {res_c.returncode}\n\nSTDOUT:\n{res_c.stdout}\n\nSTDERR:\n{res_c.stderr}")
                return

            # Decompression
            cmd_decomp = [self.exe, "-i", self.output_gdhc, "-o", self.decoded_ppm, "-m", "decompress"]
            res_d = subprocess.run(cmd_decomp, capture_output=True, text=True)
            
            if res_d.returncode == 0 and os.path.exists(self.decoded_ppm):
                self.img_reco = Image.open(self.decoded_ppm)
                self.update_stats()
                self.redraw()
            else:
                self.show_log("GDHC Decompression Failed", f"Return Code: {res_d.returncode}\n\nSTDOUT:\n{res_d.stdout}\n\nSTDERR:\n{res_d.stderr}")

        except Exception as e:
            self.show_log("System Error", f"Execution failed: {str(e)}\n\nCommand attempted:\n{' '.join(cmd_comp)}")

    def update_stats(self):
        try:
            in_s = os.path.getsize(self.temp_ppm)
            out_s = os.path.getsize(self.output_gdhc)
            ratio = in_s / out_s if out_s > 0 else 0
            self.stats_lbl.config(text=f"Ratio: {ratio:.2f}x | GDHC: {out_s/1024:.1f}KB | Src: {in_s/1024:.1f}KB")
        except: pass

    def on_pan_start(self, event):
        self.last_mouse_x, self.last_mouse_y = event.x, event.y

    def on_pan_move(self, event):
        self.pan_x += (event.x - self.last_mouse_x)
        self.pan_y += (event.y - self.last_mouse_y)
        self.last_mouse_x, self.last_mouse_y = event.x, event.y
        self.redraw()

    def on_zoom(self, event):
        factor = 1.1 if (event.delta > 0 or event.num == 4) else 0.9
        self.scale = max(0.1, min(self.scale * factor, 40.0))
        self.redraw()

    def redraw(self):
        if not self.img_orig: return
        w, h = self.img_orig.size
        nw, nh = int(w * self.scale), int(h * self.scale)
        
        # Use NEAREST to observe quantization/block artifacts clearly
        r_orig = self.img_orig.resize((nw, nh), Image.NEAREST)
        self.tk_orig = ImageTk.PhotoImage(r_orig)
        self.can_orig.delete("all")
        self.can_orig.create_image(self.pan_x, self.pan_y, anchor=tk.NW, image=self.tk_orig)

        if self.img_reco:
            r_reco = self.img_reco.resize((nw, nh), Image.NEAREST)
            self.tk_reco = ImageTk.PhotoImage(r_reco)
            self.can_reco.delete("all")
            self.can_reco.create_image(self.pan_x, self.pan_y, anchor=tk.NW, image=self.tk_reco)

if __name__ == "__main__":
    root = tk.Tk()
    app = GDHCVisualizer(root)
    root.mainloop()
