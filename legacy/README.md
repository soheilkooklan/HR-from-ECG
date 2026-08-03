# HR-from-ECG
🫀ECG Heart Rate Analyzer

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

 A user-friendly Python application has been developed to analyze Electrocardiogram (ECG/EKG) data and calculate heart rate. This Python code is designed for amateur users and features an easy-to-use graphical interface (GUI) for loading ECG data, calculating heart rate, and visualizing ECG signals with R-peaks. It is ideal for individuals who are learning to work with ECG data.

## Features

- **Import ECG Data**: Import ECG data from a CSV file for analysis. The noisy signal is accepted, and simple noise reduction is performed in the code.
- **Heart Rate Calculation**: Automatically detect R-peaks in the ECG signal and calculate heart rate in beats per minute (bpm).
- **ECG Visualization**: View the ECG signal and see R-peaks plotted on the graph.
- **Graphical User Interface (GUI)**: An easy-to-use interface built with `tkinter` that allows users to interact with the tool without needing to code.
- **Help Feature**: A built-in help button explains how to load data, analyze it, and use the interface.

## Screenshots

![ECG Heart Rate Analyzer GUI](HR%20from%20ECG%20GUI.jpg)
![ECG Heart Rate Analyzer With Sample ECG](HR%20from%20Sample%20ECG%20.jpg)

## License
- This project is licensed under the MIT License. See the LICENSE file for details.
- This project was inspired by tutorials on working with ECG data in Python. Thanks to the Python community and scientific libraries like `numpy`, `scipy`, `tkinter`, and `matplotlib` for making such projects possible.

## Getting Started

### ECG Data Format
The application expects the ECG data to be in a CSV file with a single column named ECG. Here's an example of how the data should look:
| ECG  |
| ------------- | 
| 0.1 | 
| 0.2 |
| 0.3 | 
| ... | 

### Prerequisites

To run this application, you need to have the following libraries installed:

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `tkinter` (pre-installed with most Python version)

You can install the required libraries using:

```bash
pip install numpy pandas scipy matplotlib
py -m pip install numpy pandas scipy matplotlib
