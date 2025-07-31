import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Obesity Classification Tuning")

train_df = pd.read_csv("dataset/obesity_data_train_preprocessing.csv")
test_df = pd.read_csv("dataset/obesity_data_test_preprocessing.csv")

# Memisahkan fitur dan target
X_train = train_df.drop("ObesityCategory", axis=1)
y_train = train_df["ObesityCategory"]
X_test = test_df.drop("ObesityCategory", axis=1)
y_test = test_df["ObesityCategory"]

input_example = X_train[0:5]

n_estimators_range = np.linspace(50, 300, 4, dtype=int)
max_depth_range = np.linspace(5, 20, 4, dtype=int)

best_accuracy = 0.0
best_params = {}

for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:
        with mlflow.start_run(run_name=f"RF_{n_estimators}_{max_depth}"):
            # Inisialisasi model
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

            # Training
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Evaluasi
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            # Logging parameter dan metric
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            # estimator.html
            mlflow.log_text(f"{str(model)}", artifact_file="estimator.html")

            # metric_info.json
            metrics_dict = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
            mlflow.log_dict(metrics_dict, artifact_file="metric_info.json")


            # confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            plt.tight_layout()
            mlflow.log_figure(fig, artifact_file="confusion_matrix.png")
            plt.close(fig)

            # Simpan model yang terbaik
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth
                }
                best_model = model
                # log model terbaik secara eksplisit
                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    artifact_path="model",
                    input_example=input_example
                )

            print(f"[INFO] n_estimators={n_estimators}, max_depth={max_depth} → Accuracy={accuracy:.4f}")
