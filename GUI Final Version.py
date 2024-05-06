import pickle
import tkinter as tk
from tkinter import messagebox
import unicodeit
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             mean_absolute_percentage_error, r2_score,
                             median_absolute_error, mean_squared_log_error, root_mean_squared_error)
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from ngboost import NGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.impute import SimpleImputer

# Load your dataset
data = pd.read_excel("Dataset.xlsx")

# Assuming the last column is the target and the others are features
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
# tkinter GUI
root = tk.Tk()
root.title(f"Prediction of load carrying capacity of column")

canvas1 = tk.Canvas(root, width=600, height=600)
canvas1.configure(background='#e9ecef')
canvas1.pack()


# Labels and entry boxes
labels = ['Eccentricity (mm)',
          'Column Height  (mm)',
          'Concrete Strength of Standard Cylinder  (MPa)',
          'Area of Concrete Core  (mm\u00b2)',
          'Area of Steel Tube  (mm\u00b2)',
          'Yield strength of steel tube (MPa)',
          'Total Thickness of FRP Warps   (mm)',
          'Width of FRP Wraps × Clear Spacing of FRP (mm)',
          'Elastic Modulus of FRP (MPa)'
          ]

entry_boxes = []
for i, label_text in enumerate(labels):
    label = tk.Label(root, text=unicodeit.replace(label_text), font=('Times New Roman', 15, 'italic'), bg='#e9ecef',
                     pady=5)
    canvas1.create_window(20, 120 + i * 30, anchor="w", window=label)

    entry = tk.Entry(root)
    canvas1.create_window(530, 120 + i * 30, window=entry)
    entry_boxes.append(entry)

# label_output = tk.Label(root, text='Flow of Concrete', font=('Times New Roman', 12, 'bold'),
# bg='#e9ecef')
# canvas1.create_window(50, 420, anchor="w", window=label_output)

label_output1 = tk.Label(root, text='Load carrying capacity:', font=('Times New Roman', 18, 'bold'),
                         bg='#e9ecef')
canvas1.create_window(20, 560, anchor="w", window=label_output1)

def reset_entries():
    for entry in entry_boxes:
        entry.delete(0, tk.END)
def values():
    # Validate and get the values from the entry boxes
    input_values = []
    for entry_box in entry_boxes:
        value = entry_box.get().strip()
        if value:
            try:
                input_values.append(float(value))
            except ValueError:
                messagebox.showerror("Error", "Invalid input. Please enter valid numeric values.")
                return
        else:
            messagebox.showerror("Error", "Please fill in all the input fields.")
            return


    print(len(input_values))
    input_data = pd.DataFrame([input_values ],
                        columns=X.columns)

    # Load the trained MultiOutputRegressor model
    # Assuming input_values are collected correctly
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    with open('DNN_Load_Capacity.pkl', 'rb') as model_file:
        trained_model = pickle.load(model_file)

    input_data = pd.DataFrame([input_values], columns=X.columns)
    input_data_scaled = scaler.transform(input_data)  # Scale the data
    prediction_result = trained_model.predict(input_data_scaled)
    prediction_result1 = round(prediction_result[0], 2)

    # Display the prediction on the GUI
    label_prediction = tk.Label(root, text=f'{str(prediction_result1)} kN', font=('Times New Roman', 20, 'bold'),
                                bg='white')
    canvas1.create_window(280, 560, anchor="w", window=label_prediction)


button1 = tk.Button(root, text='Predict', command=values, bg='#4285f4', fg='white',
                    font=('Times New Roman', 20, 'bold'),
                    bd=3, relief='ridge')
canvas1.create_window(440, 560, anchor="w", window=button1)

# Reset Button
button_reset = tk.Button(root, text="Reset", command=reset_entries, bg="red", fg="white", font=("Times New Roman", 20, "bold"), bd=3, relief="ridge")
canvas1.create_window(500, 500, window=button_reset)

root.mainloop()
