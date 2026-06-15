import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Read file manually
data = []

with open("auto-mpg.csv", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()

        if len(parts) < 8:
            continue

        try:
            mpg = float(parts[0])
            cylinders = int(parts[1])
            displacement = float(parts[2])

            horsepower = parts[3]
            if horsepower == "?":
                continue

            horsepower = float(horsepower)
            weight = float(parts[4])
            acceleration = float(parts[5])

            data.append([
                mpg,
                cylinders,
                displacement,
                horsepower,
                weight,
                acceleration
            ])
        except:
            continue

# Create DataFrame
df = pd.DataFrame(data, columns=[
    "mpg",
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration"
])

print("Dataset Shape:", df.shape)
print(df.head())

# Features
X = df[[
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration"
]]

# Target
y = df["mpg"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

print("\nMean Absolute Error:",
      mean_absolute_error(y_test, predictions))

print("R² Score:",
      r2_score(y_test, predictions))

# Sample prediction
new_car = pd.DataFrame({
    "cylinders": [4],
    "displacement": [140],
    "horsepower": [90],
    "weight": [2500],
    "acceleration": [15]
})

predicted_mpg = model.predict(new_car)

print("\nPredicted Fuel Consumption (MPG):",
      round(predicted_mpg[0], 2))
import joblib

joblib.dump(model, "fuel_model.pkl")
print("Model Saved Successfully!")