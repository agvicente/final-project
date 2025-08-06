import mlflow
import mlflow.sklearn
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import time

def compare_algorithms():

    algorithms = {
        'IsolationForest': IsolationForest(contamination=0.1, random_state=42),
        'OneClassSvm': OneClassSVM(nu=0.1, kernel='rbf'),
        #TODO: estudar viabilidade de usar o LOF no lugar do MoT
        'LocalOutlierFactor': LocalOutlierFactor(n_neighbors=20, contamination=0.1),
        'EllipticEnvelope': EllipticEnvelope(contamination=0.1, random_state=42)
    }

    from baseline_experiment import generate_iot_like_data
    X, y_true = generate_iot_like_data(n_samples=5000)

    results = []

    for name, model in algorithms.items():
        with mlflow.start_run(run_name=f"{name}_comparison"):
            start_time = time.time()

            if name == 'LocalOutlierFactor':
                y_pred = model.fit_predict(X)
            else:
                model.fit(X)
                y_pred = model.predict(X)

            training_time = time.time() - start_time

            accuracy = sum(y_pred == y_true) / len(y_true)

            mlflow.log_param("algorithm", name)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("training_time", training_time)

            if name != 'LocalOutlierFactor':
                mlflow.sklearn.log_model(model, f"{name}_model")
            
            results.append({
                'algorithm': name,
                'accuracy': accuracy,
                'time': training_time
            })

            print(f"{name}: Acc={accuracy:.3f}, Time{training_time:.2f}s")

    return results

if __name__ == "__main__":
    run_id = int(time.time())
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(f"Algorithm-Comparison-{run_id}")
    compare_algorithms()