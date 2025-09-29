from flask import Flask, render_template, request
import pickle
import numpy as np
import os
import joblib
import pandas as pd

app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "..", "best_model", "transformed_object", "preprocessing.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "..", "best_model", "model.pkl")


with open(PREPROCESSOR_PATH, "rb") as f:
    preprocessor = pickle.load(f)

model = joblib.load(MODEL_PATH)


FEATURE_COLUMNS = [
    "transmission",
    "fuel_type",
    "drivetrain",
    "body_type",
    "make",
    "mileage",
    "engine_hp",
    "vehicle_age"
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Collect user input
            user_input = {
                "transmission": request.form["transmission"],
                "fuel_type": request.form["fuel_type"],
                "drivetrain": request.form["drivetrain"],
                "body_type": request.form["body_type"],
                "make": request.form["make"],
                "mileage": float(request.form["mileage"]),
                "engine_hp": float(request.form["engine_hp"]),
                "vehicle_age": float(request.form["vehicle_age"])
            }


            input_df = pd.DataFrame([user_input], columns=FEATURE_COLUMNS)


            input_transformed = preprocessor.transform(input_df)

            # Predict
            pred = model.predict(input_transformed)[0]
            prediction = round(pred, 2)

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
