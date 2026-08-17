import pandas as pd
import matplotlib.pyplot as mlt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
#import tkinter as tk
import customtkinter as ctk
import re
from tkinter import messagebox
from PIL import Image,ImageTk


r =ctk.CTk()
r.geometry("500x200")
r.config(bg="black")
r.title("Emotion Detection APP")
r.resizable(False,False)

bg_image = Image.open("C:\\Users\\LENOVO\\Downloads\\emojis.jpg")
bg_image = bg_image.resize((600,300))
bg_photo = ImageTk.PhotoImage(bg_image)

bg_label = ctk.CTkLabel(r, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

df=pd.read_csv("C:\\Users\\LENOVO\\Desktop\\Javascript\\dataset\\emotion_dataset.csv")
X=df["text"]
df["emotion"]=df["emotion"].map({'surprise':0, 'joy':1, 'disgust':2, 'fear':3, 'sadness':4, 'anger':5})
y=df["emotion"]
vector =TfidfVectorizer()
X_vectorized =vector.fit_transform(X)
#X_train,X_test,y_train,y_test=train_test_split(X_vectorized,y,random_state=42,test_size=0.2)
model =LogisticRegression()
model.fit(X_vectorized,y)

App_name =ctk.CTkLabel(r,text="Emotion Detection App",text_color="red",font=("Arial",20,"bold"),bg_color="transparent",fg_color="transparent")
App_name.place(x=150,y=20)

#f=tk.Frame(bg="white")



label =ctk.CTkLabel(r,text="Enter Text :",text_color="blue",font=("Arial",15), fg_color="transparent", bg_color="transparent")
label.place(x=80,y=105)

text_entry =ctk.CTkEntry(r,width=300,placeholder_text="Enter...",placeholder_text_color="#B2D7FD")
text_entry.place(x=200,y=105)



def Predict():
    message =text_entry.get()
    if(message==""):
        messagebox.showerror(title="ERROR",message="Please Enter message!")
        return
    message = message.lower()
    message = re.sub(r"[^a-z0-9\s']", " ", message)
    cleaned =re.sub(r"\s+", " ", message).strip()
    message_vectorized=vector.transform([cleaned])
    pred = model.predict(message_vectorized[0])
    
    if(pred==0):
        messagebox.showinfo(title="Emotion",message="Surprise😲")
    elif(pred==1):
        messagebox.showinfo(title="Emotion",message="Joy😊")
    elif(pred==2):
        messagebox.showinfo(title="Emotion",message="Disgust🤢")
    elif(pred==3):
        messagebox.showinfo(title="Emotion",message="Fear😨")
    elif(pred==4):
        messagebox.showinfo(title="Emotion",message="Sadness😢")
    elif(pred==5):
        messagebox.showinfo(title="Emotion",message="Anger😡")
    else:
        messagebox.showerror(title="Emotion",message="Something is wrong")

    
    text_entry.delete(0,ctk.END)
    
button =ctk.CTkButton(r,text="check",command=Predict,fg_color="green",font=("Arial",15),bg_color="black")
button.place(x=200,y=150)

#f.place(x=150,y=150)
r.mainloop()