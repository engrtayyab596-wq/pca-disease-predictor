import joblib
import numpy as np


def load_model(path='models/pipeline.pkl'):
    model = joblib.load(path)
    return model


def predict(model, features):
    input_data = np.array(features)
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0]
    result = 'Malignant' if prediction[0] == 1 else 'Benign'
    return {
        'prediction': result,
        'probability_benign': round(float(probability[0]), 4),
        'probability_malignant': round(float(probability[1]), 4)
    }
