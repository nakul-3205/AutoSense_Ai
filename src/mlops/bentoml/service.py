from __future__ import annotations
import os
import pandas as pd
import pickle
import joblib
import bentoml
from bentoml.io import JSON

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "best_model", "transformed_object", "preprocessing.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "best_model", "model.pkl")

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

with open(PREPROCESSOR_PATH, "rb") as f:
    preprocessor = pickle.load(f)

model = joblib.load(MODEL_PATH)


@bentoml.Service()
class CarPriceService:

    @bentoml.api
    def predict(self, input_data):
        df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)
        transformed = preprocessor.transform(df)
        prediction = model.predict(transformed)[0]
        return {"prediction": round(prediction, 2)}
