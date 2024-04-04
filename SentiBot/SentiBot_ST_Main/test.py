import streamlit as st
import pandas as pd
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# MultiNomial Naive Bayes
df = pd.read_csv('SentiBot_ST_Main\sentiment.csv' , quotechar='"')
X = df['sentence']
y = df['mood']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Model Accuracy: {accuracy}')

st.title("Mood Companion")

if st.button("Click here to Start Recording"):
    # Speech-to-text 
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        st.write("Listening... Click the button again to stop recording.")
        try:
            audio = recognizer.listen(source, timeout=None)

            st.write("Processing...")

            user_input_text = recognizer.recognize_google(audio)
            st.write(f"Speech-to-Text Result: {user_input_text}")

            # Sentiment analysis prediction
            input_data = vectorizer.transform([user_input_text])
            prediction = model.predict(input_data)[0]
            st.write(f"Predicted Mood: {prediction}")

        except sr.UnknownValueError:
            st.write("Sorry, I could not understand the audio.")
