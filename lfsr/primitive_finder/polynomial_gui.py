import subprocess
import tkinter as tk
from tkinter import messagebox, simpledialog
import os

# Path to the compiled C++ finder program and the Python plotter script
FINDER_PATH = './primitive_poly_finder'
PLOTTER_PATH = 'plot_polynomials.py'

def run_finder(degree, x, y):
    try:
        # Run the C++ primitive polynomial finder and capture its output
        output = subprocess.check_output([FINDER_PATH, str(degree), str(x), str(y)], text=True)
        return output.splitlines()
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Failed to run polynomial finder: {e}")
        return []

def plot_polynomial(x, y, seed, polynomial, delay):
    try:
        subprocess.run(['python3', PLOTTER_PATH, '--x', str(x), '--y', str(y), '--seed', seed, '--poly', polynomial, '--delay', str(delay)], check=True)
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Failed to plot polynomial: {e}")

def on_find_and_plot():
    # Get input values
    degree = int(degree_entry.get())
    x = int(x_entry.get())
    y = int(y_entry.get())
    delay = float(delay_entry.get())

    # Find primitive polynomials
    polynomials = run_finder(degree, x, y)
    if not polynomials:
        return

    # Let the user choose a polynomial to plot
    poly_choice = simpledialog.askstring("Choose Polynomial", f"Found {len(polynomials)} primitive polynomials.\nEnter the index (starting from 0) of the polynomial to plot:\n\n" + "\n".join([f"{i}: {p}" for i, p in enumerate(polynomials)]))

    if poly_choice is not None:
        try:
            index = int(poly_choice)
            if 0 <= index < len(polynomials):
                seed = simpledialog.askstring("Enter Seed", "Enter the initial seed as a binary string (e.g., 1000):")
                if seed:
                    plot_polynomial(x, y, seed, polynomials[index], delay)
            else:
                messagebox.showerror("Error", "Invalid polynomial index selected.")
        except ValueError:
            messagebox.showerror("Error", "Invalid input. Please enter a valid index.")

# Set up the GUI
root = tk.Tk()
root.title("Primitive Polynomial Finder & Plotter")

tk.Label(root, text="Degree:").grid(row=0, column=0)
degree_entry = tk.Entry(root)
degree_entry.grid(row=0, column=1)

tk.Label(root, text="X-Dimension:").grid(row=1, column=0)
x_entry = tk.Entry(root)
x_entry.grid(row=1, column=1)

tk.Label(root, text="Y-Dimension:").grid(row=2, column=0)
y_entry = tk.Entry(root)
y_entry.grid(row=2, column=1)

tk.Label(root, text="Delay (seconds):").grid(row=3, column=0)
delay_entry = tk.Entry(root)
delay_entry.grid(row=3, column=1)

find_plot_button = tk.Button(root, text="Find & Plot", command=on_find_and_plot)
find_plot_button.grid(row=4, columnspan=2)

root.mainloop()
