from fastapi import FastAPI
from pydantic import BaseModel, Field, validator
import mlflow.sklearn
import pandas as pd

app = FastAPI()

mlflow.set_tracking_uri(
    '../mlflow/mlruns'
)


# Model mapping: name -> run_id
model_mapping = {
    "XGB": "908adc7932d84b7186197fe117f70ae4",
    "RF": "4293e4d7a04d4b6fbb9764101bb032c5",
    # Add other models here
}


def load_pipeline(model_name: str):
    if model_name not in model_mapping:
        raise ValueError(f"Model {model_name} not found.")
    run_id = model_mapping[model_name]
    return mlflow.sklearn.load_model(f"runs:/{run_id}/model")


class ModelInput(BaseModel):
    flight_cycle: float
    flight_phase: str = Field(..., example="CLIMB")
    egt_probe_average: float
    fuel_flw: float
    core_spd: float
    zpn12p: float
    vib_n1_1_bearing: float = Field(..., alias="vib_n1_#1_bearing")
    vib_n2_1_bearing: float = Field(..., alias="vib_n2_#1_bearing")
    vib_n2_turbine_frame: float

    class Config:
        allow_population_by_field_name = True

    @validator('flight_phase')
    def validate_flight_phase(cls, v):
        v = v.strip().upper()
        if v not in {"CLIMB", "CRUISE", "DESCENT"}:
            raise ValueError("Flight phase must be CLIMB/CRUISE/DESCENT")
        return v


@app.post("/predict/{model_name}")
async def predict(model_name: str, input_data: ModelInput):
    try:
        pipeline = load_pipeline(model_name)
    except ValueError as e:
        return {"error": str(e)}

    # Convert input to dictionary with correct column names
    input_dict = input_data.dict(by_alias=True)

    # Create DataFrame with proper column order
    input_df = pd.DataFrame([input_dict])[[
        'flight_cycle',
        'flight_phase',
        'egt_probe_average',
        'fuel_flw',
        'core_spd',
        'zpn12p',
        'vib_n1_#1_bearing',
        'vib_n2_#1_bearing',
        'vib_n2_turbine_frame'
    ]]

    try:
        # Get prediction and convert numpy types to native Python types
        prediction = pipeline.predict(input_df)
        result = float(prediction[0])  # Convert numpy.float32 to Python float
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

    return {
        "model": model_name,
        "prediction": result
    }
