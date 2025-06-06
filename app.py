from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

app = Flask(__name__)

# ======== Load Model dan Tokenizer ========
model = tf.keras.models.load_model('model/model.h5', compile=False)

with open('tokenizer/tokenizer1.json', 'r') as f:
    tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)

# Panjang input saat training
MAXLEN = 30

# Label hasil training (harus sesuai urutan `LabelEncoder.classes_`)
label_map = ['Negative', 'Positive']  # Index 0 = Negative, Index 1 = Positive

# ======== Route untuk Form ========
@app.route('/')
def index():
    return render_template('index.html')

# ======== Route untuk Prediksi ========
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']

    # Tokenisasi dan padding
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAXLEN, padding='post')

    # Prediksi
    prediction = model.predict(padded)[0]
    predicted_index = np.argmax(prediction)
    confidence = prediction[predicted_index]

    label = label_map[predicted_index]

    return render_template('index.html', prediction=label, score=confidence, input_text=text)

if __name__ == '__main__':
    app.run(debug=True)
