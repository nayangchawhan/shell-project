import joblib
import pandas as pd

# Load trained model
model = joblib.load("battery_range_model.joblib")
print("✅ Model loaded successfully!")

# Create a test input (example: BMW i3)
test_data = {
    "Battery_Capacity_kWh": 108.0,
    "Charging_Duration_hours": 0.6,
    "Charging_Rate_kW": 36.3,
    "State_of_Charge_Start": 29.3,
    "State_of_Charge_End": 86.1,
    "Temperature_°C": 27.9,
    "Vehicle_Age_years": 2.0,
    "Energy_Consumed_kWh": 60.7,
    "Vehicle_Model": "BMW i3",
    "Charger_Type": "DC Fast Charger",
    "User_Type": "Commuter"
}

# Convert to DataFrame
df = pd.DataFrame([test_data])

# Predict
predicted_range = model.predict(df)[0]
print(f"🔋 Predicted Battery Range (km): {predicted_range:.2f}")
