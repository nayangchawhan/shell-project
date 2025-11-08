import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# ==========================
# 1️⃣ Load and clean dataset
# ==========================
data = pd.read_csv("../data/charging_sessions.csv")

# Normalize column names
data.columns = (
    data.columns.str.strip()
    .str.replace(" ", "_")
    .str.replace(r"[\(\)%]", "", regex=True)
)

print("✅ Columns after cleaning:\n", data.columns.tolist())

# Rename problematic columns
data.rename(columns={
    "State_of_Charge_Start_": "State_of_Charge_Start",
    "State_of_Charge_End_": "State_of_Charge_End"
}, inplace=True)

# ==========================
# 2️⃣ Drop or fix missing target
# ==========================
if data["Distance_Driven_since_last_charge_km"].isna().sum() > 0:
    print(f"⚠️ Found {data['Distance_Driven_since_last_charge_km'].isna().sum()} missing target values — dropping them.")
    data = data.dropna(subset=["Distance_Driven_since_last_charge_km"])

# ==========================
# 3️⃣ Feature & target selection
# ==========================
required_cols = [
    "Battery_Capacity_kWh",
    "Charging_Duration_hours",
    "Charging_Rate_kW",
    "State_of_Charge_Start",
    "State_of_Charge_End",
    "Temperature_°C",
    "Vehicle_Age_years",
    "Energy_Consumed_kWh",
]

missing = [c for c in required_cols if c not in data.columns]
if missing:
    raise KeyError(f"❌ Missing columns in dataset: {missing}")

X = data[required_cols].copy()
y = data["Distance_Driven_since_last_charge_km"]

cat_cols = ["Vehicle_Model", "Charger_Type", "User_Type"]
for col in cat_cols:
    if col in data.columns:
        X[col] = data[col]

# ==========================
# 4️⃣ Handle Missing Values in Features
# ==========================
numeric_features = required_cols
categorical_features = cat_cols

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ==========================
# 5️⃣ Build the Model Pipeline
# ==========================
model = Pipeline([
    ("preproc", preprocessor),
    ("rf", RandomForestRegressor(n_estimators=150, random_state=42))
])

# ==========================
# 6️⃣ Split & Train
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"📊 Training samples: {len(X_train)}, Test samples: {len(X_test)}")
model.fit(X_train, y_train)

# ==========================
# 7️⃣ Evaluate & Save
# ==========================
score = model.score(X_test, y_test)
print(f"✅ Battery range model trained successfully! R² = {score:.3f}")

joblib.dump(model, "battery_range_model.joblib")
print("💾 Model saved as battery_range_model.joblib ✅")
