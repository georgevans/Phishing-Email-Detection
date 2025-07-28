from flask import Flask, render_template, request
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("sys.path:", sys.path)
from models.DataCleaner import clean_and_vectorise, vectorizer
from models.MNBmodel import load_model, load_vectorizer


app = Flask(__name__)

model = load_model()
vectorizer = load_vectorizer()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    raw_body = request.form.get('body')
    raw_subject = request.form.get('subject')
    print(f'raw {raw_body}')
    X = clean_and_vectorise(raw_body, vectorizer)
    print(X)
    prediction = model.predict(X)

    prediction_text = prediction[0]
    print(prediction_text)
    return render_template('index.html', prediction=prediction_text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
