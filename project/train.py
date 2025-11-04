import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    # Load data
    df = pd.read_csv("../data/winequality-white.csv", sep=";")

    X = df.drop("quality", axis=1)
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("wines_rf_tuning")

    # Define three different hyperparameter sets
    runs = [
        {"n_estimators": 50, "max_depth": 5, "min_samples_split": 2},
        {"n_estimators": 100, "max_depth": 10, "min_samples_split": 4},
        {"n_estimators": 200, "max_depth": 8, "min_samples_split": 3},
    ]

    mlflow.autolog(disable=True)
    for params in runs:
        with mlflow.start_run(
            run_name=f"rf_{params['n_estimators']}trees_{params['max_depth']}depth",
            nested=True):
            model = RandomForestRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                random_state=42,
                n_jobs=-1
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            mlflow.log_params(params)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2_score", r2)

            example = X_test[:1]
            mlflow.sklearn.log_model(model, name="model", input_example=example)

            print(f"✅ Run logged: n={params['n_estimators']}, depth={params['max_depth']}, mse={mse:.4f}, r2={r2:.4f}")