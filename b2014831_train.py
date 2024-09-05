import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Đọc dữ liệu từ file iris.csv
data = pd.read_csv('iris.csv')

# Chia dữ liệu thành đặc trưng và nhãn
X = data.drop('species', axis=1)  # Đặc trưng
y = data['species']                # Nhãn

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Huấn luyện mô hình
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Tính toán độ chính xác
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Lưu mô hình
joblib.dump(model, 'iris_model.pkl')