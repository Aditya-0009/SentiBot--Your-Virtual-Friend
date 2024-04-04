from flask import Flask, render_template, request, jsonify
import os
import joblib
import numpy as np

app = Flask(__name__)

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Specify the path to the sentiment model file
model_path = os.path.join(current_dir, 'sentiment_model.pkl')

# Check if the sentiment model file exists
if os.path.exists(model_path):
    # Load the trained sentiment analysis model
    model = joblib.load(model_path)
else:
    print(f"Error: sentiment model file '{model_path}' not found.")

# Route to render the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle audio/video processing and text processing
# Route to handle audio/video processing and text processing
@app.route('/process_audio_video', methods=['POST'])
def process_audio_video():
    if request.method == 'POST':
        # Receive data from the client
        data = request.json
        
        # Extract text data from the request
        text_data = data.get('text_data')

        # Placeholder for the predicted mood
        predicted_mood = ""

        # Perform sentiment analysis prediction
        if text_data:
            # Perform sentiment analysis prediction using the loaded model
            # Replace this with your actual sentiment analysis prediction code
            # For example:
            # predicted_mood = model.predict([text_data])[0]
            predicted_mood = np.random.choice(['Positive', 'Negative', 'Neutral'])  # Placeholder prediction

        # Return response with predicted mood
        return jsonify({'status': 'success', 'predicted_mood': predicted_mood})

if __name__ == '__main__':
    app.run(debug=True)
