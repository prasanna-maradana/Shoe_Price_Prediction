from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the saved model and encoders
model = joblib.load("model/sneaker_model.h5")
le_brand = joblib.load("model/brand_encoder.pkl")
le_color = joblib.load("model/color_encoder.pkl")

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get user input
        brand = request.form['brand']
        size = float(request.form['size'])
        color = request.form['color'].lower()  # Convert to lowercase to match the training data format
        
        # Check if color is seen in training data, else assign a default label
        if color not in le_color.classes_:
            color = 'unknown'  # Assign a default value if color is unseen
        
        # Encode categorical variables
        brand_encoded = le_brand.transform([brand])[0]
        color_encoded = le_color.transform([color])[0]
        
        # Prepare the input for the model
        input_data = np.array([[brand_encoded, size, color_encoded]])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Show result
        return render_template('index.html', prediction=round(prediction[0], 2))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
