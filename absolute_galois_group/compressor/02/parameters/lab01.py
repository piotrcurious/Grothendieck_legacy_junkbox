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
        
        label = tk.Label(self, text=title, font=('Arial', 10, 'bold'))
        label.pack(pady=(10, 5))

        frame = tk.Frame(self)
        frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.text_area = tk.Text(frame, wrap=tk.WORD, yscrollcommand=scrollbar.set, font=('Courier', 9))
        self.text_area.insert(tk.END, message)
        self.text_area.config(state=tk.DISABLED)
        self.text_area.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        scrollbar.config(command=self.text_area.yview)

        btn_close = tk.Button(self, text="Close", command=self.destroy, width=10)
        btn_close.pack(pady=(0, 10))
        
        self.transient(parent)
        self.grab_set()
        parent.wait_window(self)

class TACVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("TAC v1.0 - Topological Algebraic Compressor")
        self.root.geometry("1450x900")
        
        # --- State ---
        self.input_path = ""
        self.temp_input = "input_temp.pgm"
        self.output_tac = "output.tac"
        self.decoded_pgm = "decoded.pgm"
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
        ScrollableMessageDialog(self.root, title, message)

    def setup_ui(self):
        # Sidebar
        sidebar = tk.Frame(self.root, width=380, padx=10, pady=10, relief=tk.RAISED, borderwidth=1)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(sidebar, text="TAC ENGINE (ALGEBRAIC)", font=('Arial', 12, 'bold')).pack(pady=(0, 10))
        
        btn_open = tk.Button(sidebar, text="📁 Load Source Image", font=('Arial', 10), command=self.load_image)
        btn_open.pack(fill=tk.X, pady=5)
        
        # Scrollable area for parameters
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

        # --- Mapping to TAC Parameters ---
        self.add_section_label("Topology Settings")
        self.add_slider("Manifold Size", "--size", 2, 32, 8)
        self.add_slider("Dictionary Atoms", "--atoms", 16, 2048, 256)
        
        self.add_section_label("Compression Logic")
        self.add_slider("Splitting Threshold", "--thresh", 0.1, 100.0, 15.0)
        self.add_slider("K-Means Iterations", "--iters", 1, 100, 10)

        self.add_section_label("Block Constraints")
        self.add_slider("Min Block Size", "--min", 2, 16, 4)
        self.add_slider("Max Block Size", "--max", 16, 128, 32)

        # Action Buttons
        tk.Button(sidebar, text="▶ Compress & Decompress", bg="#2c3e50", fg="white", 
                  font=('Arial', 10, 'bold'), height=2, command=self.process).pack(fill=tk.X, pady=10)
        
        self.stats_lbl = tk.Label(sidebar, text="Ratio: -- | Size: --", relief=tk.SUNKEN, bd=1, font=('Courier', 10), pady=5)
        self.stats_lbl.pack(fill=tk.X)

        # Viewport Area
        self.view_container = tk.Frame(self.root, bg="#1e1e1e")
        self.view_container.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # Views
        self.f_left = tk.Frame(self.view_container, bg="#1e1e1e")
        self.f_left.place(relx=0, rely=0, relwidth=0.5, relheight=1)
        tk.Label(self.f_left, text="SOURCE", fg="white", bg="#34495e", font=('Arial', 9, 'bold')).pack(side=tk.TOP, fill=tk.X)
        self.can_orig = tk.Canvas(self.f_left, bg="#1e1e1e", highlightthickness=0)
        self.can_orig.pack(expand=True, fill=tk.BOTH)

        self.f_right = tk.Frame(self.view_container, bg="#1e1e1e")
        self.f_right.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)
        tk.Label(self.f_right, text="TAC RECONSTRUCTION", fg="white", bg="#c0392b", font=('Arial', 9, 'bold')).pack(side=tk.TOP, fill=tk.X)
        self.can_reco = tk.Canvas(self.f_right, bg="#1e1e1e", highlightthickness=0)
        self.can_reco.pack(expand=True, fill=tk.BOTH)

        for c in (self.can_orig, self.can_reco):
            c.bind("<ButtonPress-1>", self.on_pan_start)
            c.bind("<B1-Motion>", self.on_pan_move)
            c.bind("<MouseWheel>", self.on_zoom)

    def add_section_label(self, text):
        lbl = tk.Label(self.scroll_frame, text=text.upper(), font=('Arial', 8, 'bold'), fg="#7f8c8d")
        lbl.pack(anchor=tk.W, padx=5, pady=(15, 2))

    def add_slider(self, label, flag, v_min, v_max, v_def):
        container = tk.Frame(self.scroll_frame)
        container.pack(fill=tk.X, padx=10)
        lbl = tk.Label(container, text=label, font=('Arial', 8))
        lbl.pack(side=tk.TOP, anchor=tk.W)
        
        res = 0.1 if isinstance(v_def, float) else 1
        s = tk.Scale(container, from_=v_min, to=v_max, resolution=res, 
                      orient=tk.HORIZONTAL, font=('Arial', 8), length=300)
        s.set(v_def)
        s.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))
        self.params[flag] = s

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.pgm *.ppm *.jpg *.png *.bmp")])
        if path:
            self.input_path = path
            try:
                img = Image.open(self.input_path).convert('L') # TAC typically works on Grayscale/PGM
                img.save(self.temp_input)
                self.img_orig = img
                self.redraw()
            except Exception as e:
                self.show_log("Loading Error", str(e))

    def process(self):
        if not self.img_orig: 
            messagebox.showwarning("Warning", "Load an image first.")
            return
        
        # Build Command: ./shared_comp c <in> <out> --key=val
        cmd_comp = [self.exe, "c", self.temp_input, self.output_tac]
        
        for flag, slider in self.params.items():
            val = slider.get()
            formatted_val = str(int(val)) if val == int(val) else f"{val:.2f}"
            cmd_comp.append(f"{flag}={formatted_val}")
        
        try:
            # Compression Phase
            res_c = subprocess.run(cmd_comp, capture_output=True, text=True)
            if res_c.returncode != 0:
                self.show_log("TAC Compression Failed", f"STDOUT:\n{res_c.stdout}\n\nSTDERR:\n{res_c.stderr}")
                return

            # Decompression Phase: ./shared_comp d <in> <out>
            cmd_decomp = [self.exe, "d", self.output_tac, self.decoded_pgm]
            res_d = subprocess.run(cmd_decomp, capture_output=True, text=True)
            
            if res_d.returncode == 0 and os.path.exists(self.decoded_pgm):
                self.img_reco = Image.open(self.decoded_pgm)
                self.update_stats()
                self.redraw()
            else:
                self.show_log("TAC Decompression Failed", f"STDERR:\n{res_d.stderr}")

        except Exception as e:
            self.show_log("System Error", f"Execution failed: {str(e)}")

    def update_stats(self):
        try:
            in_s = os.path.getsize(self.temp_input)
            out_s = os.path.getsize(self.output_tac)
            ratio = in_s / out_s if out_s > 0 else 0
            self.stats_lbl.config(text=f"Ratio: {ratio:.2f}x | TAC: {out_s/1024:.1f}KB | Src: {in_s/1024:.1f}KB")
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
        self.scale = max(0.1, min(self.scale * factor, 50.0))
        self.redraw()

    def redraw(self):
        if not self.img_orig: return
        w, h = self.img_orig.size
        nw, nh = int(w * self.scale), int(h * self.scale)
        
        # Original
        r_orig = self.img_orig.resize((nw, nh), Image.NEAREST)
        self.tk_orig = ImageTk.PhotoImage(r_orig)
        self.can_orig.delete("all")
        self.can_orig.create_image(self.pan_x, self.pan_y, anchor=tk.NW, image=self.tk_orig)

        # Reconstructed
        if self.img_reco:
            r_reco = self.img_reco.resize((nw, nh), Image.NEAREST)
            self.tk_reco = ImageTk.PhotoImage(r_reco)
            self.can_reco.delete("all")
            self.can_reco.create_image(self.pan_x, self.pan_y, anchor=tk.NW, image=self.tk_reco)

if __name__ == "__main__":
    root = tk.Tk()
    app = TACVisualizer(root)
    root.mainloop()
