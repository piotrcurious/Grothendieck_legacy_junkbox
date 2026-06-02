import os
import sys
import subprocess
import time
from PIL import Image

def run_benchmark(tool_path, input_list, qualities):
    print("# RNN Absolute Benchmark Report")
    print("| Image | Quality | Original Size | Compressed Size | Ratio | Time (s) | Status |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    for img_path in input_list:
        if not os.path.exists(img_path): continue
        orig_size = os.path.getsize(img_path)
        base = os.path.basename(img_path)

        for q in qualities:
            out_rnn = f"tmp_{base}.rnn"
            recon_pgm = f"recon_{base}_q{q}.pgm"

            start = time.time()
            res = subprocess.run([tool_path, "compress", img_path, out_rnn, str(q)], capture_output=True)
            elapsed = time.time() - start

            if res.returncode == 0:
                comp_size = os.path.getsize(out_rnn)
                ratio = orig_size / comp_size

                # Verify decompression
                res_dec = subprocess.run([tool_path, "decompress", out_rnn, recon_pgm], capture_output=True)
                status = "OK" if res_dec.returncode == 0 else "DEC_FAIL"

                # Check bit-perfection if quality is 0
                if q == 0 and status == "OK":
                    try:
                        recon_img = Image.open(recon_pgm).convert("L")
                        orig_img = Image.open(img_path).convert("L")
                        # Center crop to match test logic if needed
                        w, h = recon_img.size
                        if orig_img.size != (w, h):
                           ox, oy = (orig_img.width - w)//2, (orig_img.height - h)//2
                           orig_img = orig_img.crop((ox, oy, ox+w, oy+h))

                        import numpy as np
                        if (np.array(orig_img) != np.array(recon_img)).any():
                            status = "NOT_BIT_PERFECT"
                    except Exception as e:
                        print(f"Verification error: {e}")

                # Convert to PNG for user inspection as requested
                if status == "OK" or status == "NOT_BIT_PERFECT":
                    try:
                        im = Image.open(recon_pgm)
                        im.save(recon_pgm.replace(".pgm", ".png"))
                        os.remove(recon_pgm)
                    except: pass

                print(f"| {base} | {q} | {orig_size} | {comp_size} | {ratio:.4f}:1 | {elapsed:.2f} | {status} |")
                if os.path.exists(out_rnn): os.remove(out_rnn)
            else:
                print(f"| {base} | {q} | {orig_size} | - | - | {elapsed:.2f} | COMP_FAIL |")

if __name__ == "__main__":
    tool = "./rnn_tool"

    jpg_inputs = [
        "../absolute_galois_group/compressor/01/iter/ghcd_08/ghcd_0c_fix/test.jpg",
        "../absolute_galois_group/compressor/01/iter/ghcd_08/test.jpg",
        "../absolute_galois_group/compressor/01/iter/ghcd_05/test.jpg"
    ]

    pgm_inputs = []
    for jpg in jpg_inputs:
        if os.path.exists(jpg):
            pgm = jpg.replace(".jpg", "_conv.pgm")
            try:
                img = Image.open(jpg).convert("L")
                img.crop((0, 0, 128, 128)).save(pgm)
                pgm_inputs.append(pgm)
            except: pass

    run_benchmark(tool, pgm_inputs, [5, 25, 50])

    for pgm in pgm_inputs:
        if os.path.exists(pgm): os.remove(pgm)
