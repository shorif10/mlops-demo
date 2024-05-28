import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn
import joblib
import os

def load_data(path):
    return pd.read_csv(path)

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse

if __name__ == "__main__":
    mlflow.start_run()
    X_train = load_data("data/X_train.csv")
    y_train = load_data("data/y_train.csv")
    X_test = load_data("data/X_test.csv")
    y_test = load_data("data/y_test.csv")

    model = train_model(X_train, y_train)
    mse = evaluate_model(model, X_test, y_test)

    mlflow.log_param("model", "LinearRegression")
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")

    # Ensure the model directory exists
    os.makedirs("model", exist_ok=True)
    # Save the model to the model directory
    joblib.dump(model, "model/model.joblib")

    mlflow.end_run()


# # src/train.py
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# import mlflow
# import mlflow.sklearn
#
# def load_data(path):
#     return pd.read_csv(path)
#
# def train_model(X_train, y_train):
#     model = LinearRegression()
#     model.fit(X_train, y_train)
#     return model
#
# def evaluate_model(model, X_test, y_test):
#     predictions = model.predict(X_test)
#     mse = mean_squared_error(y_test, predictions)
#     return mse
#
# if __name__ == "__main__":
#     mlflow.start_run()
#     X_train = load_data("data/X_train.csv")
#     y_train = load_data("data/y_train.csv")
#     X_test = load_data("data/X_test.csv")
#     y_test = load_data("data/y_test.csv")
#
#     model = train_model(X_train, y_train)
#     mse = evaluate_model(model, X_test, y_test)
#
#     mlflow.log_param("model", "LinearRegression")
#     mlflow.log_metric("mse", mse)
#     mlflow.sklearn.log_model(model, "model")
#     mlflow.end_run()
