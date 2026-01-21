import os
import subprocess
import PIL.Image
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, FloatSlider, Layout
import numpy as np

# --- 1. Setup and Compilation ---
CPP_SOURCE = "shared_comp.cpp"
EXE_NAME = "./shared_comp"

def compile_cpp():
    print("Compiling C++ GDHC Tunable...")
    cmd = f"g++ -O3 -I /usr/include/eigen3 {CPP_SOURCE} -o {EXE_NAME} -lboost_iostreams -lz"
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if process.returncode != 0:
        print("Compilation Error:", process.stderr)
    else:
        print("Compilation Successful.")

# --- 2. Helper Functions ---
def prepare_image(img_path):
    """Converts input image to Grayscale PGM for the C++ tool."""
    img = PIL.Image.open(img_path).convert('L')
    # Ensure dimensions are multiples of 16 (dict block size) to avoid padding issues
    w, h = img.size
    img = img.crop((0, 0, w - (w % 16), h - (h % 16)))
    img.save("input_temp.pgm")
    return "input_temp.pgm"

def get_file_size(filepath):
    return os.path.getsize(filepath) / 1024  # KB

# --- 3. Interactive Tuning Logic ---
def run_compression(dict_entries, dct_step, dct_coeffs):
    input_pgm = "input_temp.pgm"
    compressed_file = "output.gdhc"
    decompressed_pgm = "decoded.pgm"
    
    # Run Compression
    cmd_c = f"{EXE_NAME} c {input_pgm} {compressed_file} {dict_entries} {dct_step} {dct_coeffs}"
    subprocess.run(cmd_c, shell=True, capture_output=True)
    
    # Run Decompression
    cmd_d = f"{EXE_NAME} d {compressed_file} {decompressed_pgm}"
    subprocess.run(cmd_d, shell=True, capture_output=True)
    
    # Metrics
    original_size = get_file_size(input_pgm)
    compressed_size = get_file_size(compressed_file)
    ratio = original_size / compressed_size if compressed_size > 0 else 0
    
    # Load for Preview
    orig_img = PIL.Image.open(input_pgm)
    reco_img = PIL.Image.open(decompressed_pgm)
    
    # Visualization
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    ax[0].imshow(orig_img, cmap='gray')
    ax[0].set_title(f"Original\n({original_size:.2f} KB)")
    ax[0].axis('off')
    
    ax[1].imshow(reco_img, cmap='gray')
    ax[1].set_title(f"Compressed (Dict: {dict_entries}, Step: {dct_step})\n({compressed_size:.2f} KB) - Ratio: {ratio:.1f}x")
    ax[1].axis('off')
    plt.show()

# --- 4. Execution ---
# compile_cpp() # Uncomment if you need to compile first
prepare_image("test.jpg") # Path to your test image

# If using Jupyter/Colab:
# interact(run_compression, 
#          dict_entries=IntSlider(min=8, max=256, step=8, value=64),
#          dct_step=FloatSlider(min=0.5, max=20.0, step=0.5, value=4.0),
#          dct_coeffs=IntSlider(min=1, max=8, step=1, value=2))
