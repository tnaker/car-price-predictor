import joblib
import os
from data_processing import preprocess_new_car

def predict_price(new_car: dict) -> float:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'car_price_data_origin.csv')
    model_dir = os.path.join(base_dir, 'model')

    # Load model và danh sách cột
    model = joblib.load(os.path.join(model_dir, 'car_price_model.pkl'))
    feature_list = joblib.load(os.path.join(model_dir, 'features.pkl'))

    # Xử lý dữ liệu xe mới
    df_encoded = preprocess_new_car(new_car, feature_list)

    # Dự đoán
    predicted_price = model.predict(df_encoded)[0]
    return round(predicted_price, 2)
