from flask import Flask, render_template, request, jsonify, redirect, url_for
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Handle the login form submission here
        # You can process the form data and perform any necessary actions
        return redirect(url_for('home'))
    else:
        return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/home', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return render_template('video.html')

@app.route('/chat')
def chat_page():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    global chat_history_ids
    
    if request.method == "POST":
        # Get user message from the request form
        user_message = request.form["msg"]
        
        # Generate response from chatbot model
        response = get_Chat_response(user_message)
        
        return jsonify({'user_message': user_message, 'chatbot_response': response})
    
def get_Chat_response(text):

    # Let's chat for 5 lines
    for step in range(5):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        # generated a response while limiting the total chat history to 1000 tokens, 
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        # pretty print last ouput tokens from bot
        return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)


@app.route('/process_audio_video', methods=['POST'])
def process_audio_video():
    if request.method == 'POST':
        data = request.json
        
        # Extract text data from the request
        text_data = data.get('text_data')

        # Perform sentiment analysis prediction
        if text_data:
            # Perform sentiment analysis using Vader
            predicted_mood = predict_sentiment_vader(text_data)

            # Return response with predicted mood
            return jsonify({'status': 'success', 'predicted_mood': predicted_mood})
        else:
            return jsonify({'status': 'error', 'message': 'No text data provided'})

def predict_sentiment_vader(text):
    # Initialize Vader sentiment analyzer
    sa = SentimentIntensityAnalyzer()
    dd = sa.polarity_scores(text=text)
    compound = dd['compound']
    
    # Classify sentiment based on compound score
    if compound >= 0.05:
        return 'Positive'
    elif compound <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

if __name__ == '__main__':
    app.run(debug=True)
