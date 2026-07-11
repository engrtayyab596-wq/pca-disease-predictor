from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)
#test 1
def test_health_endpoint():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok', 
                               'model': 'loaded',
                               'description': 'Disease Risk Predictor API'
                              }
#test 2
def test_prediction_endpoint_returns_correct_fields():
    response = client.post('/predict', json={
        "radius_mean": 12.47,
        "texture_mean": 18.6,
        "perimeter_mean": 81.09,
        "area_mean": 481.9,
        "smoothness_mean": 0.09965,
        "compactness_mean": 0.1058,
        "concavity_mean": 0.08005,
        "concave_points_mean": 0.03821,
        "symmetry_mean": 0.1925,
        "fractal_dimension_mean": 0.06373,
        "radius_se": 0.3961,
        "texture_se": 1.044,
        "perimeter_se": 2.497,
        "area_se": 30.29,
        "smoothness_se": 0.006953,
        "compactness_se": 0.01911,
        "concavity_se": 0.02701,
        "concave_points_se": 0.01037,
        "symmetry_se": 0.01782,
        "fractal_dimension_se": 0.003586,
        "radius_worst": 14.97,
        "texture_worst": 24.64,
        "perimeter_worst": 96.05,
        "area_worst": 677.9,
        "smoothness_worst": 0.1426,
        "compactness_worst": 0.2378,
        "concavity_worst": 0.2671,
        "concave_points_worst": 0.1015,
        "symmetry_worst": 0.3014,
        "fractal_dimension_worst": 0.0875
    })
    assert response.status_code == 200
    assert 'prediction' in response.json()
    assert 'probability_benign' in response.json()
    assert 'probability_malignant' in response.json()

#test 3

def test_prediction_value_is_valid():
    response = client.post('/predict', json={
        "radius_mean": 12.47,
        "texture_mean": 18.6,
        "perimeter_mean": 81.09,
        "area_mean": 481.9,
        "smoothness_mean": 0.09965,
        "compactness_mean": 0.1058,
        "concavity_mean": 0.08005,
        "concave_points_mean": 0.03821,
        "symmetry_mean": 0.1925,
        "fractal_dimension_mean": 0.06373,
        "radius_se": 0.3961,
        "texture_se": 1.044,
        "perimeter_se": 2.497,
        "area_se": 30.29,
        "smoothness_se": 0.006953,
        "compactness_se": 0.01911,
        "concavity_se": 0.02701,
        "concave_points_se": 0.01037,
        "symmetry_se": 0.01782,
        "fractal_dimension_se": 0.003586,
        "radius_worst": 14.97,
        "texture_worst": 24.64,
        "perimeter_worst": 96.05,
        "area_worst": 677.9,
        "smoothness_worst": 0.1426,
        "compactness_worst": 0.2378,
        "concavity_worst": 0.2671,
        "concave_points_worst": 0.1015,
        "symmetry_worst": 0.3014,
        "fractal_dimension_worst": 0.0875
    })
    
    prediction = response.json().get('prediction')
    assert prediction in ['Malignant', 'Benign']