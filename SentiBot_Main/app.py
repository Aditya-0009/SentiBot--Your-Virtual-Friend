from flask import Flask, render_template, request, jsonify, redirect, url_for
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pyttsx3

import requests
from flask import redirect


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", padding_side='left')
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return redirect(url_for('home'))
    else:
        return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

from flask import session  # Import session

@app.route('/home', methods=['GET', 'POST'])
def home():
    # Retrieve username from session
    username = session.get('username')
    return render_template('home.html', username=username)  # Pass username to the template


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

@app.route("/get", methods=["POST"])
def chat():
    if request.method == "POST":
        user_message = request.form["msg"]      
        response_text = get_Chat_response(user_message)
        speak_response(response_text)
        return jsonify({'user_message': user_message, 'chatbot_response': response_text})

def get_Chat_response(text):
    new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')
    chat_history_ids = model.generate(new_user_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response_text.strip()

engine = pyttsx3.init()
import threading
# Function to speak the response text
def speak_response(text):
    # Create a new thread to run the speech synthesis
    def speak_in_thread():
        # Speak the text
        engine.say(text)
        engine.runAndWait()

    # Start the speech synthesis in a new thread
    threading.Thread(target=speak_in_thread).start()


@app.route('/process_audio_video', methods=['POST'])
def process_audio_video():
    if request.method == 'POST':
        data = request.json
        text_data = data.get('text_data')
        if text_data:
            predicted_mood = predict_sentiment_vader(text_data)
            return jsonify({'status': 'success', 'predicted_mood': predicted_mood})
        else:
            return jsonify({'status': 'error', 'message': 'No text data provided'})

def predict_sentiment_vader(text):
    sa = SentimentIntensityAnalyzer()
    dd = sa.polarity_scores(text=text)
    compound = dd['compound']
    if compound >= 0.05:
        return 'Positive'
    elif compound <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'


if __name__ == '__main__':
    app.run(debug=True)
