from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

# Charger le modèle
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

# Schéma d'entrée
class WeatherInput(BaseModel):
    MinTemp: float
    MaxTemp: float
    Humidity9am: float
    Humidity3pm: float
    Pressure9am: float
    Pressure3pm: float
    RainToday: int  # 0 = No, 1 = Yes

# Endpoint santé
@app.get("/health")
def health():
    return {"status": "ok"}

# Endpoint prédiction
@app.post("/predict")
def predict(data: WeatherInput):
    X = [[
        data.MinTemp, data.MaxTemp,
        data.Humidity9am, data.Humidity3pm,
        data.Pressure9am, data.Pressure3pm,
        data.RainToday
    ]]
    prediction = model.predict(X)[0]
    result = "Oui 🌧️" if prediction == 1 else "Non ☀️"
    return {"RainTomorrow": result}