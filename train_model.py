import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os
import re

# Load dataset
df = pd.read_csv("Shoe prices.csv")

# Drop missing values in used columns
df = df.dropna(subset=["Brand", "Size", "Color", "Price (USD)"])

# Clean 'Size' column: extract number from strings like "US 7"
def extract_size(val):
    match = re.search(r"(\d+\.?\d*)", str(val))
    return float(match.group(1)) if match else np.nan

df["Size"] = df["Size"].apply(extract_size)
df = df.dropna(subset=["Size"])  # Drop rows where size couldn't be extracted

# Clean 'Price (USD)' column: remove dollar sign and spaces
df["Price (USD)"] = df["Price (USD)"].replace({r'[$,]': ''}, regex=True).astype(float)

# Select features and target
X = df[["Brand", "Size", "Color"]].copy()
y = df["Price (USD)"]

# Encode categorical features
le_brand = LabelEncoder()
le_color = LabelEncoder()

X.loc[:, "Brand"] = le_brand.fit_transform(X["Brand"])
X.loc[:, "Color"] = le_color.fit_transform(X["Color"])

le_color = LabelEncoder()
color_classes = np.append(le_color.fit(["red", "blue", "green", "yellow", "black", "white"]), 'unknown')
le_color.classes_ = color_classes

# Save encoders
os.makedirs("model", exist_ok=True)
joblib.dump(le_brand, "model/brand_encoder.pkl")
joblib.dump(le_color, "model/color_encoder.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))

# Save model
joblib.dump(model, "model/sneaker_model.h5")
print("✅ Model trained and saved to 'model/sneaker_model.h5'")
