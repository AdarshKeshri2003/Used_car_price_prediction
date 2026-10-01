
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(
    title="Used Car Price Prediction API",
    description="Predict used-car selling prices using a trained Random Forest model.",
    version="1.0.0",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "final_car_price_model.pkl"


try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from: {MODEL_PATH}")

    # Verify the expected input columns
    expected_columns = list(model.feature_names_in_)
    print("Expected model columns:", expected_columns)

except Exception as exc:
    model = None
    expected_columns = []
    print(f"Error loading model: {exc}")


class CarDetails(BaseModel):
    year: int = Field(..., ge=1900, le=2100)
    km_driven: float = Field(..., ge=0)
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: float = Field(..., ge=0)
    engine: float = Field(..., ge=0)
    max_power: float = Field(..., ge=0)
    torque_value: float = Field(..., ge=0)
    seats: float = Field(..., ge=1)
    brand: str = Field(..., min_length=1)


@app.get("/")
def home():
    return {
        "message": "Used Car Price Prediction API is running",
        "docs": "/docs",
        "model_loaded": model is not None,
    }


@app.get("/health")
def health():
    return {
        "status": "running",
        "model_loaded": model is not None,
        "expected_columns": expected_columns,
    }


@app.post("/predict")
def predict_price(car: CarDetails):

    if model is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Model is not loaded. Check that "
                "final_car_price_model.pkl exists and "
                "that the installed scikit-learn version "
                "matches the training environment."
            ),
        )

    try:
        # Convert the validated request to a dictionary
        car_data = car.model_dump()

        # Create DataFrame with the exact model column order
        input_data = pd.DataFrame(
            [car_data],
            columns=expected_columns,
        )

        # Predict using the complete saved pipeline
        prediction = model.predict(input_data)

        predicted_price = float(prediction[0])

        return {
            "predicted_price": round(predicted_price, 2),
            "currency": "INR",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )