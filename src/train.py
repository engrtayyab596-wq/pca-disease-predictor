import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def build_pipeline(n_components=10):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components)),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    return pipeline


def train_model(pipeline, X_train, y_train):
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    print(f"Mean CV accuracy: {cv_scores.mean():.4f}")
    print(f"Std CV accuracy:  {cv_scores.std():.4f}")
    pipeline.fit(X_train, y_train)
    return pipeline, cv_scores


def save_model(pipeline, path='models/pipeline.pkl'):
    joblib.dump(pipeline, path)
    print(f"Model saved to {path}")