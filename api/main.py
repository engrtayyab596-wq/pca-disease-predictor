from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import load_model, predict as run_prediction


app = FastAPI(title='Disease Risk Predictor')
model = load_model()


class PatientData(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float
    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    symmetry_se: float
    fractal_dimension_se: float
    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float


@app.get('/health')
def health():
    return {
        'status': 'ok',
        'model': 'loaded',
        'description': 'Disease Risk Predictor API'
    }


@app.post('/predict')
def predict(data: PatientData):
    features = [[
        data.radius_mean, data.texture_mean,
        data.perimeter_mean, data.area_mean,
        data.smoothness_mean, data.compactness_mean,
        data.concavity_mean, data.concave_points_mean,
        data.symmetry_mean, data.fractal_dimension_mean,
        data.radius_se, data.texture_se,
        data.perimeter_se, data.area_se,
        data.smoothness_se, data.compactness_se,
        data.concavity_se, data.concave_points_se,
        data.symmetry_se, data.fractal_dimension_se,
        data.radius_worst, data.texture_worst,
        data.perimeter_worst, data.area_worst,
        data.smoothness_worst, data.compactness_worst,
        data.concavity_worst, data.concave_points_worst,
        data.symmetry_worst, data.fractal_dimension_worst
    ]]
    return run_prediction(model, features)
