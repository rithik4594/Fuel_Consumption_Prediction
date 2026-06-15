from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("fuel_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    cylinders = float(request.form['cylinders'])
    displacement = float(request.form['displacement'])
    horsepower = float(request.form['horsepower'])
    weight = float(request.form['weight'])
    acceleration = float(request.form['acceleration'])

    data = pd.DataFrame({
        'cylinders':[cylinders],
        'displacement':[displacement],
        'horsepower':[horsepower],
        'weight':[weight],
        'acceleration':[acceleration]
    })

    prediction = model.predict(data)[0]

    return render_template(
        'index.html',
        prediction_text=f"Predicted Fuel Consumption: {prediction:.2f} MPG"
    )

if __name__ == "__main__":
    app.run(debug=True)