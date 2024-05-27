# src/preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = load_data("data/boston_housing.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df)
    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)
