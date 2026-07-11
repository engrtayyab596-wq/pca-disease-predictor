import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(filepath):
    df = pd.read_excel(filepath)
    return df


def clean_data(df):
    df = df.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df


def split_data(df):
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test