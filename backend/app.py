from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import math
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load ML models
try:
    station_model = joblib.load("range_model.joblib")
    battery_model = joblib.load("battery_range_model.joblib")
except Exception as e:
    print("⚠️ Error loading model files:", e)

# Load station dataset
try:
    stations = pd.read_csv("../data/stations.csv")
    stations.columns = (
        stations.columns.str.strip()
        .str.replace(" ", "_")
        .str.replace(r"[\(\)]", "", regex=True)
    )
except Exception as e:
    print("⚠️ Warning: Could not load stations.csv:", e)

# Initialize FastAPI
app = FastAPI(title="EV Recommender + Gemini AI API", version="2.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Input Models
# ---------------------------
class StationRequest(BaseModel):
    latitude: float
    longitude: float

class BatteryRequest(BaseModel):
    Battery_Capacity_kWh: float
    Charging_Duration_hours: float
    Charging_Rate_kW: float
    State_of_Charge_Start: float
    State_of_Charge_End: float
    Temperature_C: float
    Vehicle_Age_years: float
    Energy_Consumed_kWh: float
    Vehicle_Model: str
    Charger_Type: str
    User_Type: str

class ChatRequest(BaseModel):
    message: str

# ---------------------------
# Utility Function
# ---------------------------
def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two GPS coordinates (km)."""
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# ---------------------------
# API Routes
# ---------------------------
@app.get("/")
def root():
    return {"message": "🚗 EV Backend with Gemini AI is running successfully!"}

@app.post("/recommend")
def recommend(req: StationRequest):
    """Recommend nearest optimal EV charging station."""
    if "Latitude" not in stations.columns or "Longitude" not in stations.columns:
        raise HTTPException(status_code=500, detail="Stations dataset not loaded correctly.")

    # Compute distance & score
    stations["Distance_km"] = stations.apply(
        lambda row: haversine(req.latitude, req.longitude, row.Latitude, row.Longitude),
        axis=1,
    )
    stations["Score"] = stations["Power_kW"] / (stations["Distance_km"] + 0.1)
    best = stations.sort_values("Score", ascending=False).iloc[0]

    # Predict approximate range
    features = np.array([[best.Power_kW, best.Latitude, best.Longitude]])
    predicted_range = station_model.predict(features)[0]

    return {
        "recommended_station": best.Station_Name,
        "city": best.City,
        "operator": best.Operator,
        "connector": best.Connector_Type,
        "power_kW": best.Power_kW,
        "distance_km": round(best.Distance_km, 2),
        "predicted_range_km": round(predicted_range, 2),
    }

@app.post("/predict_battery_range")
def predict_battery_range(req: BatteryRequest):
    """Predict the EV driving range based on session parameters."""
    try:
        df = pd.DataFrame([req.dict()])
        predicted_range = battery_model.predict(df)[0]
        return {
            "predicted_distance_km": round(predicted_range, 2),
            "note": "Estimated distance your EV can drive under current charge and conditions."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting range: {e}")

@app.post("/chat")
def chat(req: ChatRequest):
    """Gemini-powered conversational assistant."""
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise HTTPException(status_code=500, detail="Gemini API key not found. Set GEMINI_API_KEY in .env.")

    prompt = f"You are an EV expert assistant helping users with battery range, charging, and station queries.\nUser: {req.message}"

    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-latest:generateContent?key={gemini_api_key}",
            json={
                "contents": [
                    {"parts": [{"text": prompt}]}
                ]
            },
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()

        data = response.json()
        answer = data["candidates"][0]["content"]["parts"][0]["text"]
        return {"reply": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {e}")
