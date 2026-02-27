import streamlit as st
import pandas as pd
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data.csv")

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

df["clean_text"] = df["text"].apply(preprocess)

# TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

# Train Model
model = LogisticRegression()
model.fit(X, y)

# Streamlit UI
st.title("📰 Fake News Detection App")

user_input = st.text_area("Enter News Text:")

if st.button("Predict"):
    clean_input = preprocess(user_input)
    vector_input = vectorizer.transform([clean_input])
    prediction = model.predict(vector_input)[0]
    st.success(f"Prediction: {prediction.upper()}")