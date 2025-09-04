import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import gradio as gr


# Bước 1: Đọc dữ liệu
df = pd.read_csv('insurance.csv')

# Bước 2: Tiền xử lý
# Mã hóa các cột phân loại
le_sex = LabelEncoder()
df['sex'] = le_sex.fit_transform(df['sex'])

le_smoker = LabelEncoder()
df['smoker'] = le_smoker.fit_transform(df['smoker'])
df = pd.get_dummies(df, columns=['region'], drop_first=True)

# Bước 3: Chia X và y
X = df.drop('charges', axis=1)
y = df['charges']

# Bước 4: Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bước 5: Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Bước 6: Xây dựng và compile mô hình
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Bước 7: Huấn luyện mô hình
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

# Bước 8: Lưu mô hình và scaler
model.save('insurance_model.h5')
# Lưu scaler và encoder để sử dụng cho dự đoán
import joblib
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_sex, 'le_sex.pkl')
joblib.dump(le_smoker, 'le_smoker.pkl')


from sklearn.metrics import mean_absolute_error, r2_score

# Đánh giá mô hình trên tập kiểm tra
y_predict_test = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_predict_test)
r2 = r2_score(y_test, y_predict_test)
# In ra terminal để kiểm tra
print(f"Test MAE: {mae:.2f}")
print(f"Test R-squared: {r2:.2f}")

# Tải lại các file đã lưu
from tensorflow.keras.models import load_model
import joblib
import numpy as np

# Load the saved model and preprocessors
model = load_model('insurance_model.h5', compile=False)
scaler = joblib.load('scaler.pkl')
le_sex = joblib.load('le_sex.pkl')
le_smoker = joblib.load('le_smoker.pkl')

def predict_insurance_cost(age, sex, bmi, children, smoker, region):
    # 1. Mã hóa các đặc trưng phân loại
    sex_encoded = le_sex.transform([sex])[0]
    smoker_encoded = le_smoker.transform([smoker])[0]
    
    region_northwest = 1 if region == 'northwest' else 0
    region_southeast = 1 if region == 'southeast' else 0
    region_southwest = 1 if region == 'southwest' else 0
    
    # 2. Sắp xếp dữ liệu đầu vào theo đúng thứ tự
    input_data = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_northwest, region_southeast, region_southwest]])
    
    # 3. Chuẩn hóa dữ liệu đầu vào
    input_data_scaled = scaler.transform(input_data)
    
    # 4. Dự đoán
    prediction = model.predict(input_data_scaled)[0][0]
    
    return f"Chi phí bảo hiểm dự đoán: {prediction:.2f} USD"

import gradio as gr

inputs = [
    gr.Slider(18, 100, step=1, label="Tuổi"),
    gr.Radio(["female", "male"], label="Giới tính"),
    gr.Slider(15, 50, label="BMI"),
    gr.Number(label="Số con đã sinh", precision=0, value=0, minimum=0),
    gr.Radio(["no", "yes"], label="Hút thuốc"),
    gr.Radio(["northeast", "northwest", "southeast", "southwest"], label="Vùng")
]

outputs = gr.Textbox(label="Kết quả")
accuracy_text = f"Độ chính xác của mô hình: MAE = {mae:.2f} USD và R-squared = {r2:.2f}"

# Tạo một component Textbox để hiển thị thông tin
accuracy_display = gr.Textbox(accuracy_text, label="Đánh giá mô hình", interactive=False)
# Sửa đổi hàm launch
gr.Interface(
    fn=predict_insurance_cost, 
    inputs=inputs, 
    outputs=outputs, 
    title="Dự đoán Chi phí Bảo hiểm Y tế",
    # Thêm mô tả hoặc thông tin chính xác
    description=accuracy_text
).launch()
