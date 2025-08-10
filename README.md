# 👟 Shoe Price Prediction using Machine Learning

Predict the price of a shoe given its brand, size, and color using a machine learning model.  
This project combines data preprocessing, model training with Random Forest Regressor, and a user-friendly Flask web app for real-time predictions.

---

## 📌 Features

- Cleans and preprocesses real shoe pricing data (brands, sizes, colors, prices).
- Encodes categorical variables automatically.
- Trains a Random Forest regression model.
- Evaluates model accuracy (RMSE).
- Provides a web interface for user input (brand, size, color) and live price predictions.
- Handles unseen colors via the `'unknown'` fallback.
- Modular code: model training and web serving are independent.
- Tested for robust predictions on a variety of input brands and colors.

---

## 🚀 How It Works

### 1. Data Preprocessing & Model Training (`train_model.py`)
- **Reads** raw shoe price data from `Shoe-prices.csv`
- **Cleans:** 
    - Extracts numeric sizes
    - Removes missing values
    - Normalizes price values
- **Encodes** brands and colors using LabelEncoder
- **Splits** into training and test sets (80:20)
- **Trains** RandomForestRegressor on features [`Brand`, `Size`, `Color`]
- **Evaluates** using Root Mean Squared Error (prints to console)
- **Saves** the model and encoders in `/model/` folder as `.h5` and `.pkl` files

### 2. Web Application (`app.py` + `index.html`)
- **Flask web server** loads the trained model and encoders
- **Input form** (`index.html`): brand, size, color  
- **Prediction:**  
    - Encodes form input using the trained encoders  
    - If a new color is given, replaces with 'unknown' (falls back gracefully)  
    - Passes encoded input to the model for price prediction  
    - Renders the predicted price on the webpage

---



