from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse
from pydantic import BaseModel
from predict_model import predict_price
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, "static")

app = FastAPI()

app.mount("/static", StaticFiles(directory=data_path), name="static")

class CarData(BaseModel):
    Year: int
    Engine_HP: float
    Engine_Cylinders: float
    highway_MPG: int
    city_mpg: int
    Popularity: int
    Transmission_Type: str
    Driven_Wheels: str
    Vehicle_Size: str
    Vehicle_Style: str
    Engine_Fuel_Type: str

@app.get("/", response_class=HTMLResponse)
def serve_web():
    return FileResponse(os.path.join(data_path, "index.html"))

@app.post("/predict_price/")
def predict_car_price(car: CarData):
    input_data = {
        "Year": car.Year,
        "Engine HP": car.Engine_HP, 
        "Engine Cylinders": car.Engine_Cylinders,
        "highway MPG": car.highway_MPG,
        "city mpg": car.city_mpg,
        "Popularity": car.Popularity,
        "Transmission Type": car.Transmission_Type,
        "Driven Wheels": car.Driven_Wheels,
        "Vehicle Size": car.Vehicle_Size,
        "Vehicle Style": car.Vehicle_Style,
        "Engine Fuel Type": car.Engine_Fuel_Type
    }

    price = predict_price(input_data)
    return {"predicted_price": price}