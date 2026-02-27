import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load saved objects
with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Text preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# App UI
st.title("📰 Fake News Detection App")
st.write("Enter a news article to check if it is Fake or Real.")

user_input = st.text_area("News Text")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter text.")
    else:
        clean_text = preprocess_text(user_input)
        vector = tfidf.transform([clean_text])
        prediction = model.predict(vector)

        if prediction[0] == 1:
            st.error("🟥 Fake News")
        else:
            st.success("🟩 Real News")
