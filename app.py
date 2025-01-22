import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pyttsx3
import pickle
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from flask import session
import threading

app = Flask(__name__)

# Load the GPT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", padding_side='left')
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return redirect(url_for('home'))
    else:
        return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/home', methods=['GET', 'POST'])
def home():
    username = session.get('username')
    return render_template('home.html', username=username)

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return render_template('video.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/chat')
def chat_page():
    return render_template('chat.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route("/get", methods=["POST"])
def chat():
    if request.method == "POST":
        user_message = request.form["msg"]
        response_text = get_chat_response(user_message)
        speak_response(response_text)
        return jsonify({'user_message': user_message, 'chatbot_response': response_text})

def get_chat_response(text):
    new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')
    chat_history_ids = model.generate(new_user_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response_text.strip()

# Function to speak the response text
def speak_response(text):
    def speak_in_thread():
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()

    threading.Thread(target=speak_in_thread).start()


stopwords_set = set(stopwords.words('english'))
emoticon_pattern = re.compile('(?::|;|=)(?:-)?(?:\)|\(|D|P)')


# Load your sentiment analysis model and TF-IDF vectorizer
# Load your sentiment analysis model and TF-IDF vectorizer
with open(os.path.join(os.path.dirname(__file__), 'models', 'clf.pkl'), 'rb') as f:
    clf = pickle.load(f)
with open(os.path.join(os.path.dirname(__file__), 'models', 'tfidf.pkl'), 'rb') as f:
    tfidf = pickle.load(f)

def preprocessing(text):
    text = re.sub('<[^>]*>', '', text)
    emojis = emoticon_pattern.findall(text)
    text = re.sub('[\W+]', ' ', text.lower()) + ' '.join(emojis).replace('-', '')
    prter = PorterStemmer()
    text = [prter.stem(word) for word in text.split() if word not in stopwords_set]

    return " ".join(text)

def predict_sentiment(text):
    preprocessed_text = preprocessing(text)
    text_vector = tfidf.transform([preprocessed_text])
    sentiment = clf.predict(text_vector)[0]
    return "Positive" if sentiment == 1 else "Negative"



@app.route('/process_audio_video', methods=['POST'])
def process_audio_video():
    if request.method == 'POST':
        data = request.json
        text_data = data.get('text_data')
        if text_data:
            predicted_mood = predict_sentiment(text_data)
            return jsonify({'status': 'success', 'predicted_mood': predicted_mood})
        else:
            return jsonify({'status': 'error', 'message': 'No text data provided'})

if __name__ == '__main__':
    app.run(debug=True)
