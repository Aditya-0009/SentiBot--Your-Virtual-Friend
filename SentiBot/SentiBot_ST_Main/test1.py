import streamlit as st
import pandas as pd
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Ensemble Learning ( Multinomail Naive Bayes , svm , decision tree)
df = pd.read_csv('sentiment.csv',quotechar='"')
X = df['sentence']
y = df['mood']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the models
vectorizer = TfidfVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)

model1 = MultinomialNB()
model2 = SVC(probability=True, kernel='linear')
model3 = DecisionTreeClassifier()

# Create a Voting Classifier
ensemble_model = VotingClassifier(estimators=[
    ('nb', model1),
    ('svm', model2),
    ('dt', model3)
], voting='hard')  # 'hard' for hard voting (majority voting)

# Train the ensemble model
ensemble_model.fit(X_train_transformed, y_train)

# Evaluate the ensemble model
y_pred_ensemble = ensemble_model.predict(X_test_transformed)
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
st.write(f'Ensemble Model Accuracy: {accuracy_ensemble}')

st.title("Mood Companion")

# Display the clickable button to start recording
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

            # Sentiment analysis prediction using the ensemble model
            input_data = vectorizer.transform([user_input_text])
            prediction = ensemble_model.predict(input_data)[0]
            st.write(f"Predicted Mood: {prediction}")

        except sr.UnknownValueError:
            st.write("Sorry, I could not understand the audio.")
