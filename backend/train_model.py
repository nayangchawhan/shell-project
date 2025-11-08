import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
data = pd.read_csv("../data/stations.csv")

# Clean and rename columns
data.columns = [c.strip().replace(" ", "_").replace("(", "").replace(")", "") for c in data.columns]

# Drop missing or invalid rows
data = data.dropna(subset=["Latitude", "Longitude", "Power_kW"])

# For ML — simulate "range" (km) as a function of power
# In real case, we’d use real vehicle + trip data
data["Estimated_Range_km"] = data["Power_kW"] * 3  # simple assumption: more kW => higher range

# Features and target
X = data[["Power_kW", "Latitude", "Longitude"]]
y = data["Estimated_Range_km"]

# Model pipeline
model = Pipeline([
    ("rf", RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X, y)
joblib.dump(model, "range_model.joblib")

print("✅ Model trained successfully!")
print(f"Trained on {len(data)} stations.")
