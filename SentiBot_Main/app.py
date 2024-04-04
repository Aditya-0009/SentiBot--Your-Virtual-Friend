from flask import Flask, render_template, request, jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

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
