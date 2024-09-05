from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Đọc mô hình và độ chính xác đã lưu
model = joblib.load('iris_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = [
        data['sepal_length'],
        data['sepal_width'],
        data['petal_length'],
        data['petal_width']
    ]
    prediction = model.predict([features])[0]

    # Tính toán độ chính xác
    predicted_probs = model.predict_proba([features])[0]
    accuracy = max(predicted_probs) * 100

    return jsonify({
        'prediction': prediction,
        'accuracy': accuracy
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
