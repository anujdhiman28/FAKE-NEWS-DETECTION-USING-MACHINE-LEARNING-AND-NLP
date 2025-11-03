# # ==========================================
# # 📰 Fake News Detection with Tkinter + WordCloud
# # ==========================================
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import PassiveAggressiveClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# import tkinter as tk
# from tkinter import *
# from tkinter import messagebox

# # -------------------------------
# # STEP 1: Load Dataset
# # -------------------------------
# path = r"C:\Users\anujn\Downloads\archive (17)\fake_or_real_news.csv"
# data = pd.read_csv(path)

# # rename columns if needed
# data.columns = ['id', 'title', 'text', 'label']

# # combine title and text for better accuracy
# data['clean_text'] = data['title'] + " " + data['text']

# # -------------------------------
# # STEP 2: Split Data
# # -------------------------------
# x_train, x_test, y_train, y_test = train_test_split(
#     data['clean_text'], data['label'], test_size=0.2, random_state=42
# )

# # -------------------------------
# # STEP 3: TF-IDF Vectorization
# # -------------------------------
# vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
# tfidf_train = vectorizer.fit_transform(x_train)
# tfidf_test = vectorizer.transform(x_test)

# # -------------------------------
# # STEP 4: Model Training
# # -------------------------------
# pac = PassiveAggressiveClassifier(max_iter=50)
# pac.fit(tfidf_train, y_train)

# # -------------------------------
# # STEP 5: Accuracy & Confusion Matrix
# # -------------------------------
# y_pred = pac.predict(tfidf_test)
# score = accuracy_score(y_test, y_pred)
# print(f"Model Accuracy: {round(score * 100, 2)}%")

# cm = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(5, 4))
# plt.imshow(cm, cmap='Greens')
# plt.title('Confusion Matrix')
# plt.colorbar()
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()

# # -------------------------------
# # STEP 6: WordCloud Visualization
# # -------------------------------
# # Fake News WordCloud
# fake_text = " ".join(data[data['label'] == "FAKE"]['clean_text'])
# real_text = " ".join(data[data['label'] == "REAL"]['clean_text'])

# fake_wc = WordCloud(width=600, height=400, background_color='white', colormap='Reds').generate(fake_text)
# real_wc = WordCloud(width=600, height=400, background_color='white', colormap='Greens').generate(real_text)

# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.imshow(fake_wc, interpolation='bilinear')
# plt.axis('off')
# plt.title("Fake News WordCloud", fontsize=14, color='red')

# plt.subplot(1, 2, 2)
# plt.imshow(real_wc, interpolation='bilinear')
# plt.axis('off')
# plt.title("Real News WordCloud", fontsize=14, color='green')
# plt.show()

# # -------------------------------
# # STEP 7: Tkinter UI
# # -------------------------------
# root = Tk()
# root.title("📰 Fake News Detection System")
# root.geometry("650x550")
# root.configure(bg='lightyellow')

# # Title
# Label(root, text="📰 Fake News Detector", font=("Arial", 20, "bold"), bg="skyblue", fg="white").pack(fill=X, pady=10)

# # Input Box
# Label(root, text="Enter News Content:", font=("Arial", 14), bg='lightyellow', fg='black').pack(pady=5)
# text_box = Text(root, height=10, width=70, wrap=WORD, font=("Arial", 12))
# text_box.pack(pady=10)

# # Function to Predict
# def predict_news():
#     user_text = text_box.get("1.0", END).strip()
#     if user_text == "":
#         messagebox.showwarning("Input Error", "Please enter some news text!")
#         return
    
#     input_data = [user_text]
#     vectorized_input = vectorizer.transform(input_data)
#     prediction = pac.predict(vectorized_input)[0]
    
#     if prediction.lower() == "fake":
#         result_label.config(text="🚫 FAKE NEWS", fg="red")
#     else:
#         result_label.config(text="✅ REAL NEWS", fg="green")

# # Function to Clear
# def clear_text():
#     text_box.delete("1.0", END)
#     result_label.config(text="")

# # Buttons
# btn_frame = Frame(root, bg='lightyellow')
# btn_frame.pack(pady=15)

# Button(btn_frame, text="🔍 Detect", command=predict_news, bg="lightgreen", fg="black",
#        font=("Arial", 12, "bold"), padx=20, pady=5).grid(row=0, column=0, padx=10)

# Button(btn_frame, text="🧹 Clear", command=clear_text, bg="orange", fg="white",
#        font=("Arial", 12, "bold"), padx=20, pady=5).grid(row=0, column=1, padx=10)

# # Result Label
# result_label = Label(root, text="", font=("Arial", 18, "bold"), bg='lightyellow')
# result_label.pack(pady=20)

# # Footer
# Label(root, text="Developed by Anuj 💡", bg="lightyellow", fg="blue", font=("Arial", 10, "italic")).pack(side=BOTTOM, pady=5)

# root.mainloop()






# ==========================================
# 📰 Fake News Detection with Tkinter + WordCloud Button
# ==========================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import *
from tkinter import messagebox

# -------------------------------
# STEP 1: Load Dataset
# -------------------------------
path = r"C:\Users\anujn\Downloads\archive (17)\fake_or_real_news.csv"
data = pd.read_csv(path)

# rename columns if needed
data.columns = ['id', 'title', 'text', 'label']

# combine title and text for better accuracy
data['clean_text'] = data['title'] + " " + data['text']

# -------------------------------
# STEP 2: Split Data
# -------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    data['clean_text'], data['label'], test_size=0.2, random_state=42
)

# -------------------------------
# STEP 3: TF-IDF Vectorization
# -------------------------------
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = vectorizer.fit_transform(x_train)
tfidf_test = vectorizer.transform(x_test)

# -------------------------------
# STEP 4: Model Training
# -------------------------------
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# -------------------------------
# STEP 5: Accuracy & Confusion Matrix
# -------------------------------
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {round(score * 100, 2)}%")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap='Greens')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# -------------------------------
# STEP 6: Prepare WordCloud Data
# -------------------------------
fake_text = " ".join(data[data['label'] == "FAKE"]['clean_text'])
real_text = " ".join(data[data['label'] == "REAL"]['clean_text'])

# -------------------------------
# STEP 7: Tkinter UI
# -------------------------------
root = Tk()
root.title("📰 Fake News Detection System")
root.geometry("700x600")
root.configure(bg='lightyellow')

# Title
Label(root, text="📰 Fake News Detector", font=("Arial", 22, "bold"), bg="skyblue", fg="white").pack(fill=X, pady=10)

# Input Box
Label(root, text="Enter News Content:", font=("Arial", 14), bg='lightyellow', fg='black').pack(pady=5)
text_box = Text(root, height=10, width=70, wrap=WORD, font=("Arial", 12))
text_box.pack(pady=10)

# Function to Predict
def predict_news():
    user_text = text_box.get("1.0", END).strip()
    if user_text == "":
        messagebox.showwarning("Input Error", "Please enter some news text!")
        return
    
    input_data = [user_text]
    vectorized_input = vectorizer.transform(input_data)
    prediction = pac.predict(vectorized_input)[0]
    
    if prediction.lower() == "fake":
        result_label.config(text="🚫 FAKE NEWS", fg="red")
    else:
        result_label.config(text="✅ REAL NEWS", fg="green")

# Function to Clear
def clear_text():
    text_box.delete("1.0", END)
    result_label.config(text="")

# Function to Show WordClouds
def show_wordclouds():
    fake_wc = WordCloud(width=600, height=400, background_color='white', colormap='Reds').generate(fake_text)
    real_wc = WordCloud(width=600, height=400, background_color='white', colormap='Greens').generate(real_text)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(fake_wc, interpolation='bilinear')
    plt.axis('off')
    plt.title("Fake News WordCloud", fontsize=14, color='red')

    plt.subplot(1, 2, 2)
    plt.imshow(real_wc, interpolation='bilinear')
    plt.axis('off')
    plt.title("Real News WordCloud", fontsize=14, color='green')
    plt.show()

# Buttons
btn_frame = Frame(root, bg='lightyellow')
btn_frame.pack(pady=15)

Button(btn_frame, text="🔍 Detect", command=predict_news, bg="lightgreen", fg="black",
       font=("Arial", 12, "bold"), padx=20, pady=5).grid(row=0, column=0, padx=10)

Button(btn_frame, text="🧹 Clear", command=clear_text, bg="orange", fg="white",
       font=("Arial", 12, "bold"), padx=20, pady=5).grid(row=0, column=1, padx=10)

Button(btn_frame, text="🌈 Show WordCloud", command=show_wordclouds, bg="plum", fg="black",
       font=("Arial", 12, "bold"), padx=20, pady=5).grid(row=0, column=2, padx=10)

# Result Label
result_label = Label(root, text="", font=("Arial", 18, "bold"), bg='lightyellow')
result_label.pack(pady=20)

# Footer
Label(root, text="Developed by Anuj 💡", bg="lightyellow", fg="blue", font=("Arial", 10, "italic")).pack(side=BOTTOM, pady=5)

root.mainloop()
