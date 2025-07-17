from predict_model import predict_price

new_car = {
    'Year': 2020,
    'Engine HP': 300.0,
    'Engine Cylinders': 6.0,
    'highway MPG': 30,
    'city mpg': 22,
    'Popularity': 5000,
    'Transmission Type': 'AUTOMATIC',
    'Driven_Wheels': 'front wheel drive',
    'Vehicle Size': 'Midsize',
    'Vehicle Style': 'Sedan',
    'Engine Fuel Type': 'regular unleaded'
}

price = predict_price(new_car)
print(f" Giá xe dự đoán: {price} USD")
