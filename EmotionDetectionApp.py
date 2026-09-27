import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Files
DATASET_FILE = "C:\\Users\\LENOVO\\Desktop\\Javascript\\dataset\\emotion_dataset.csv"
BACKGROUND_FILE = "C:\\Users\\LENOVO\\Downloads\\emojis.jpg"

# Load the dataset
try:
    data = pd.read_csv(DATASET_FILE)
except FileNotFoundError:
    print("emotion_dataset.csv was not found.")
    exit()

# Remove rows with missing values
data = data.dropna(subset=["text", "emotion"])
texts = data["text"].astype(str)
emotions = data["emotion"].astype(str)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(texts,emotions,test_size=0.20,random_state=42,stratify=emotions)
# Convert text into TF-IDF features
vectorizer = TfidfVectorizer(lowercase=True,stop_words="english",ngram_range=(1, 2),max_features=10000)

x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

# Train the emotion classifier
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# Check model performance
predictions = model.predict(x_test)

accuracy = accuracy_score(y_test, predictions)

print("Model accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification report:")
print(classification_report(y_test, predictions))

def predict_emotion():
    text = input_box.get("1.0", tk.END).strip()

    if not text:
        messagebox.showwarning("Missing text","Please enter some text first.")
        return

    text_features = vectorizer.transform([text])
    emotion = model.predict(text_features)[0]
    probabilities = model.predict_proba(text_features)[0]
    confidence = max(probabilities) * 100

    result_label.config(text=f"Emotion: {emotion.title()}")

    confidence_label.config(text=f"Confidence: {confidence:.2f}%")


def clear_text():
    input_box.delete("1.0", tk.END)
    result_label.config(text="Emotion: -")
    confidence_label.config(text="Confidence: -")

# Create the application window
root = tk.Tk()
root.title("Emotion Detection App")
root.geometry("900x600")
root.resizable(False, False)

# Add the background image
try:
    image = Image.open(BACKGROUND_FILE)
    image = image.resize((900, 600))
    background = ImageTk.PhotoImage(image)
    background_label = tk.Label(root,image=background)
    background_label.place(x=0,y=0,relwidth=1,relheight=1)

except FileNotFoundError:
    root.configure(bg="#111827")

# Main application panel
panel = tk.Frame(root,bg="#111111",width=760,height=500)
panel.place(relx=0.5,rely=0.5,anchor="center")
panel.pack_propagate(False)

# App title
title = tk.Label(panel,text="Emotion Detection App",font=("Arial", 28, "bold"),bg="#111111",fg="white")
title.pack(pady=(30, 5))

subtitle = tk.Label(panel,text="Enter a sentence and find its emotion",font=("Arial", 13),bg="#111111",fg="#cccccc")
subtitle.pack(pady=(0, 20))


# Text input
input_label = tk.Label(panel,text="Enter Text",font=("Arial", 14, "bold"),bg="#111111",fg="white")
input_label.pack(anchor="w",padx=60)

input_box = tk.Text(
    panel,
    height=7,
    width=70,font=("Arial", 13),
    wrap=tk.WORD,bg="#222222",fg="white",insertbackground="white",
    relief="flat",padx=12,pady=12)

input_box.pack(padx=60,pady=10)

# Buttons
buttons = tk.Frame(panel,bg="#111111")
buttons.pack(pady=10)

predict_button = tk.Button(
    buttons,
    text="Predict Emotion",
    font=("Arial", 12, "bold"),
    bg="#2563eb",
    fg="white",
    activebackground="#1d4ed8",
    activeforeground="white",
    relief="flat",
    cursor="hand2",
    padx=25,
    pady=10,
    command=predict_emotion
)
predict_button.pack(side="left",padx=8)

clear_button = tk.Button(
    buttons,
    text="Clear",
    font=("Arial", 12),
    bg="#374151",
    fg="white",
    activebackground="#4b5563",
    activeforeground="white",
    relief="flat",
    cursor="hand2",
    padx=30,
    pady=10,
    command=clear_text
)

clear_button.pack(side="left",padx=8)

# Prediction result
result_label = tk.Label(panel,text="Emotion: -",font=("Arial", 22, "bold"),bg="#111111",fg="#60a5fa")
result_label.pack(pady=(20, 5))

confidence_label = tk.Label(panel,text="Confidence: -",font=("Arial", 13),bg="#111111",fg="#d1d5db")
confidence_label.pack(pady=5)

root.mainloop()