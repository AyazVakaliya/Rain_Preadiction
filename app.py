from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)


model = joblib.load("rain_model.pkl")


feature_names = [
    'TempHighF', 'TempAvgF', 'TempLowF', 'DewPointHighF', 'DewPointAvgF', 'DewPointLowF',
    'HumidityHighPercent', 'HumidityAvgPercent', 'HumidityLowPercent',
    'SeaLevelPressureHighInches', 'SeaLevelPressureAvgInches', 'SeaLevelPressureLowInches',
    'VisibilityHighMiles', 'VisibilityAvgMiles', 'VisibilityLowMiles',
    'WindHighMPH', 'WindAvgMPH', 'WindGustMPH'
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            input_data = [float(request.form.get(feature)) for feature in feature_names]
            input_array = np.array([input_data])
            prediction = model.predict(input_array)[0]
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template("index.html", prediction=prediction, features=feature_names)

if __name__ == "__main__":
    app.run(debug=True)
