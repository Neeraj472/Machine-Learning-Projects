from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
from PIL import Image, ImageTk

r = tk.Tk()
r.title("Car Price Prediction App")
r.geometry("400x400")
r.resizable(False,False)
#BACKGROUND IMAGE
bg_image = Image.open("C:\\Users\\LENOVO\\Downloads\\pexels-eslames1-31078589.jpg")
bg_image = bg_image.resize((400, 400))
bg_photo = ImageTk.PhotoImage(bg_image)

bg_label = tk.Label(r, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)


df = pd.read_csv("C:\\Users\\LENOVO\\Desktop\\Javascript\\car_cleaned.csv")
X = df[["km_driven", "mileage(Ltr)"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

def predict():

    #  CHECK EMPTY FIELDS
    if (carname_entry.get() == "" or 
        year_entry.get() == "" or 
        km_driven_entry.get() == "" or 
        fuel_entry.get() == "" or 
        mileage_entry.get() == ""):

        messagebox.showerror("Error", "Please fill all fields before predicting!")
        return


    try:
        car_name = carname_entry.get()
        value1 = int(year_entry.get())
        value2 = int(km_driven_entry.get())
        value3 = fuel_entry.get()
        value4 = float(mileage_entry.get())

        pred = model.predict([[value2, value4]])

        messagebox.showinfo(
            "Prediction",
            f"Car Name: {car_name}\nPredicted Price: {abs(pred)}INR"
        )
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid numeric values!")
        return

    # Clear fields
    carname_entry.delete(0, tk.END)
    year_entry.delete(0, tk.END)
    km_driven_entry.delete(0, tk.END)
    fuel_entry.delete(0, tk.END)
    mileage_entry.delete(0, tk.END)



# UI WIDGETS 

title = ctk.CTkLabel(r, text="Car Price Prediction", font=("Arial", 22), fg_color="#2B7EF3")
title.place(x=90, y=20)

# Car Name
carname_label = ctk.CTkLabel(r, text="Car Name:", font=("Arial", 15),
                              text_color="black", fg_color="transparent", bg_color="transparent")
carname_label.place(x=60, y=60)
carname_entry = ctk.CTkEntry(r, width=180)
carname_entry.place(x=150, y=60)

# Year
year_label = ctk.CTkLabel(r, text="Year:", font=("Arial", 15),
                           text_color="black", fg_color="transparent", bg_color="transparent")
year_label.place(x=60, y=100)
year_entry = ctk.CTkEntry(r, width=180)
year_entry.place(x=150, y=100)

# Km Driven
km_driven_label = ctk.CTkLabel(r, text="Km Driven:", font=("Arial", 15),
                                text_color="black", fg_color="transparent", bg_color="transparent")
km_driven_label.place(x=60, y=150)
km_driven_entry = ctk.CTkEntry(r, width=180)
km_driven_entry.place(x=150, y=150)

# Fuel
fuel_label = ctk.CTkLabel(r, text="Fuel:", font=("Arial", 15),
                           text_color="black", fg_color="transparent", bg_color="transparent")
fuel_label.place(x=60, y=200)
fuel_entry = ctk.CTkEntry(r, width=180)
fuel_entry.place(x=150, y=200)

# Mileage
mileage_label = ctk.CTkLabel(r, text="Mileage (Ltr):", font=("Arial", 15),
                              text_color="black", fg_color="transparent", bg_color="transparent")
mileage_label.place(x=60, y=250)
mileage_entry = ctk.CTkEntry(r, width=180)
mileage_entry.place(x=150, y=250)

# Predict Button
predict_button = ctk.CTkButton(r, text="Predict Price", command=predict,
                                width=150, fg_color="red", bg_color="red")
predict_button.place(x=130, y=300)

r.mainloop()
