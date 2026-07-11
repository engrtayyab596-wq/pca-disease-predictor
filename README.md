# Disease Risk Predictor

A production-grade machine learning system that predicts breast cancer risk 
from cell nucleus measurements using PCA dimensionality reduction and 
Logistic Regression, served via a REST API.

---

## Project Overview

This project demonstrates an end-to-end ML engineering pipeline built on the 
Wisconsin Breast Cancer dataset. It covers the full stack from raw data 
exploration to a containerised, tested, and automatically verified API service.

---

## Results

| Metric | Score |
|---|---|
| Cross-validation accuracy | 97.58% |
| Test set accuracy | 98.00% |
| Malignant recall | 98.00% |
| False negatives | 1 / 43 |
| PCA components | 10 (captures 95% variance) |

---

## Tech Stack

| Layer | Tools |
|---|---|
| Machine Learning | scikit-learn, PCA, Logistic Regression |
| Data Analysis | pandas, numpy, seaborn, matplotlib |
| Experiment Tracking | MLflow |
| API | FastAPI, uvicorn, pydantic |
| Testing | pytest |
| Containerisation | Docker |
| CI/CD | GitHub Actions |

---

## Project Structure

pca-disease-predictor/
├── data/                    # Raw dataset
├── notebooks/               # EDA and model training
│   └── 01_eda_and_pca.ipynb
├── src/                     # Core ML pipeline scripts
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── visualize.py
├── api/                     # FastAPI application
│   └── main.py
├── models/                  # Saved model (generated locally)
├── tests/                   # pytest test suite
│   └── test_pipeline.py
├── Dockerfile
├── requirements.txt
└── .github/
└── workflows/
└── ci.yml

---

## Why PCA?

The correlation heatmap revealed strong multicollinearity among size-related 
features (radius, perimeter, area) across all three measurement groups. PCA 
compressed the original 30 features into 10 principal components while 
retaining 95% of the variance — reducing model complexity without sacrificing 
predictive power.

---

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/engrtayyab596-wq/pca-disease-predictor.git
cd pca-disease-predictor
```

### 2. Create conda environment

```bash
conda create -n ML_P python=3.11
conda activate ML_P
pip install -r requirements.txt
```

### 3. Generate the model

Open and run all cells in `notebooks/01_eda_and_pca.ipynb`. This trains 
the pipeline and saves it to `models/pipeline.pkl`.

### 4. Start the API

```bash
uvicorn api.main:app --reload
```

### 5. Open the interactive docs
http://127.0.0.1:8000/docs

---

## How to Run with Docker

```bash
docker build -t disease-risk-predictor .
docker run -p 8000:8000 disease-risk-predictor
```

---

## How to Run Tests

```bash
pytest tests/ -v
```

---

## API Endpoints

### GET /health

Returns API status.

```json
{
  "status": "ok",
  "model": "loaded",
  "description": "Disease Risk Predictor API"
}
```

### POST /predict

Accepts 30 cell nucleus measurements and returns a prediction.

Example request:
```json
{
  "radius_mean": 12.47,
  "texture_mean": 18.6,
  "perimeter_mean": 81.09,
  "area_mean": 481.9,
  "smoothness_mean": 0.09965
}
```

Example response:
```json
{
  "prediction": "Benign",
  "probability_benign": 0.9010,
  "probability_malignant": 0.0990
}
```

---

## Dataset

Wisconsin Breast Cancer Dataset (UCI Machine Learning Repository)
- 569 patients
- 30 numerical features computed from cell nucleus images
- Binary classification: Malignant (212) vs Benign (357)
- No missing values

---

## Key Design Decisions

- StandardScaler inside Pipeline prevents data leakage during cross validation
- 5-fold cross validation on training set only — test set touched once
- n_components=10 selected from scree plot confirming 95% variance threshold
- joblib used for model serialisation to enable Docker deployment
- MLflow tracks all experiment parameters and metrics

---

## Author

Engr.Tayyab
ML/AI Engineering Portfolio Project

