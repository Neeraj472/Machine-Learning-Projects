#                           DIABETES PREDICTION USING LOGISTIC REGRESSION - TKINTER APP

import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# Load & Train the Model

df = pd.read_csv("C:\\Users\\LENOVO\\Desktop\\Javascript\\diabetes2.csv")

# Using only 4 features
X = df[["Glucose", "BloodPressure", "BMI", "Age"]]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Function for Prediction

def predict_diabetes():

    try:
        data = [
            float(entry_glucose.get()),
            float(entry_bp.get()),
            float(entry_bmi.get()),
            float(entry_age.get())
        ]
    except:
        messagebox.showerror("Error", "Please enter valid numbers!")
        return

    final_data = np.array([data])
    final_scaled = scaler.transform(final_data)

    result = model.predict(final_scaled)

    if result[0] == 1:
        messagebox.showinfo("Result", "Patient is LIKELY to have Diabetes.")
    else:
        messagebox.showinfo("Result", "Patient is NOT likely to have Diabetes.")


# TKINTER UI DESIGN

root = tk.Tk()
root.title("Diabetes Prediction App")
root.geometry("450x550")
root.config(bg="#f0f5f5")

title = tk.Label(root, text="Diabetes Prediction System",
                 font=("Arial", 18, "bold"), bg="#f0f5f5", fg="blue")
title.pack(pady=10)

# Labels & Entry fields
labels = ["Glucose", "Blood Pressure", "BMI", "Age"]
entries = []

for label in labels:
    lbl = tk.Label(root, text=label + ":", font=("Arial", 12), bg="#f0f5f5")
    lbl.pack()
    ent = tk.Entry(root, font=("Arial", 12), width=25)
    ent.pack(pady=5)
    entries.append(ent)

# Assign correct entries
entry_glucose, entry_bp, entry_bmi, entry_age = entries

# Predict Button
btn = tk.Button(root, text="Predict Diabetes", font=("Arial", 14, "bold"),
                bg="green", fg="white", padx=10, pady=5, command=predict_diabetes)
btn.pack(pady=20)

root.mainloop()
