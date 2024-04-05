from flask import Flask, render_template, request, jsonify, redirect, url_for
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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
def chat():
    return render_template('chat.html')


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
