from src.predict_model import predict_price

new_car = {
    'Year': 2011,
    'Engine HP': 335.0,
    'Engine Cylinders': 6.0,
    'highway MPG': 26,
    'city mpg': 19,
    'Popularity': 3916,
    'Transmission Type': 'MANUAL',
    'Driven_Wheels': 'rear wheel drive',
    'Vehicle Size': 'Compact',
    'Vehicle Style': 'Coupe',
    'Engine Fuel Type': 'premium unleaded (required)'
}

price = predict_price(new_car)
print(f" Giá xe dự đoán: {price} USD")
