import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Obesity Classification Experiment")

train_df = pd.read_csv("dataset/obesity_data_train_preprocessing.csv")
test_df = pd.read_csv("dataset/obesity_data_test_preprocessing.csv")

# Memisahkan fitur dan target
X_train = train_df.drop("ObesityCategory", axis=1)
y_train = train_df["ObesityCategory"]
X_test = test_df.drop("ObesityCategory", axis=1)
y_test = test_df["ObesityCategory"]

input_example = X_train.iloc[:5]

with mlflow.start_run():
    n_estimators = 100
    max_depth = 10
    # Aktifkan autolog
    mlflow.sklearn.autolog()

    # Inisialisasi model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )
    
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    mlflow.log_metric("accuracy", accuracy)
    print(f"Model accuracy: {accuracy:.4f}")