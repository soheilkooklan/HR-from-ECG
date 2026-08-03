import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tkinter import *
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Function to load ECG data from a CSV file
def load_ecg_data():
    filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not filepath:
        return None
    try:
        data = pd.read_csv(filepath)
        return data['ECG'].values  # Assuming 'ECG' is the column containing ECG data
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load ECG data: {str(e)}")
        return None

# Function to calculate heart rate from ECG data
def calculate_heart_rate(ecg_data, sampling_rate=500):
    # Find peaks in the ECG signal (R-peaks)
    peaks, _ = find_peaks(ecg_data, distance=sampling_rate/2.5)  # Assuming 60 bpm as a minimum HR
    rr_intervals = np.diff(peaks) / sampling_rate  # Time difference between R-peaks in seconds
    heart_rates = 60 / rr_intervals  # Convert RR intervals to heart rate
    return heart_rates, peaks

# Function to plot ECG data and heart rate
def plot_ecg_and_heart_rate(ecg_data, heart_rate, peaks):
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    ax.plot(ecg_data, label='ECG Signal')
    ax.plot(peaks, ecg_data[peaks], "x", label='R-peaks')
    ax.set_title("ECG Signal with R-peaks")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    ax.legend(loc='upper right')
    
    # Embed the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=frame_plot)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

# Function to display heart rate in the text box
def display_heart_rate(heart_rate):
    heart_rate_box.delete(1.0, END)
    avg_hr = np.mean(heart_rate)
    heart_rate_box.insert(END, f"Average Heart Rate: {avg_hr:.2f} bpm\n")
    heart_rate_box.insert(END, f"Heart Rates: {heart_rate.round(2)}\n")

# Function to show help information
def show_help():
    help_text = (
        "Instructions:\n"
        "\n"
        "1. Click 'Import ECG Data' to load a CSV file containing ECG data.\n"
        ".................................................................................................\n"
        "2. If you receive the error 'ECG' Your CSV file should have the word 'ECG' in the first column.\n"
        ".................................................................................................\n"
        "3. Select ECG File to calculate heart rate and plot the ECG data with R-peaks.\n"
        ".................................................................................................\n"
        "4. The heart rate is displayed in the text box, along with the average heart rate.\n"
        ".................................................................................................\n"
        "5. For more details, visit: https://github.com/soheilkooklan/HR-from-ECG"
    )
    messagebox.showinfo("Help", help_text)

# Main GUI setup
root = Tk()
root.title("ECG Heart Rate Analyzer")
root.configure(bg='#0A2F5A')  

# Frame for plot
frame_plot = Frame(root, bg='#0A2F5A')
frame_plot.pack(pady=20)

# Text box for displaying heart rate
heart_rate_box = Text(root, height=10, width=50)
heart_rate_box.pack(pady=10)

# Buttons
btn_load = Button(root, text="Import ECG Data", command=lambda: load_and_analyze_ecg(), bg='white', fg='black')
btn_load.pack(pady=5)

btn_help = Button(root, text="Help", command=show_help, bg='white', fg='black')
btn_help.pack(pady=5)

# Function to load and analyze ECG data
def load_and_analyze_ecg():
    ecg_data = load_ecg_data()
    if ecg_data is not None:
        heart_rate, peaks = calculate_heart_rate(ecg_data)
        display_heart_rate(heart_rate)
        plot_ecg_and_heart_rate(ecg_data, heart_rate, peaks)

# Copyright label
copyright_label = Label(root, text="© 2024 | github.com/soheilkooklan", bg='#0A2F5A', fg='white')
copyright_label.pack(side=BOTTOM)

# Run the Tkinter main loop
root.mainloop()