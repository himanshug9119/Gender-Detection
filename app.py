# Import necessary libraries
from flask import Flask, render_template, request
import numpy as np
from scipy.io import wavfile
from gender_voice import gender_voice

# Initialize Flask app
app = Flask(__name__)

# Define a route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Define a route for processing voice input
@app.route('/process_voice', methods=['POST'])
def process_voice():
    try:
        # Save the voice file
        voice_data = request.files['voiceData'].read()
        with open('input_voice.wav', 'wb') as f:
            f.write(voice_data)

        # Preprocess the voice file
        sample_rate, _ = wavfile.read('input_voice.wav')
        gender_prediction = gender_voice.predict_gender('input_voice.wav')

        # Display result on the webpage
        result_message = f"The predicted gender is {gender_prediction.capitalize()}."
        return render_template('index.html', message=result_message)
    except Exception as e:
        return render_template('index.html', message=f"Error processing voice: {str(e)}")

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
