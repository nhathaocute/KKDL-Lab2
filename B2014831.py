from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

app = Flask(__name__)

# Đọc mô hình đã lưu
model = joblib.load('iris_model.pkl')

# Đọc dữ liệu để tính nhãn thực tế
data = pd.read_csv('iris.csv')
X = data.drop('species', axis=1)
y = data['species']

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
    # Lấy nhãn thực tế cho dự đoán
    predicted_probs = model.predict_proba([features])[0]

    # Tính toán độ chính xác dựa trên xác suất dự đoán
    accuracy = max(predicted_probs) * 100  # Chọn xác suất cao nhất

    return jsonify({
        'prediction': prediction,
        'accuracy': accuracy
    })

if __name__ == '__main__':
    app.run(debug=True)