import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import time

def generate_iot_like_data(n_samples=10000, contamination=0.1):
    np.random.seed(42)
    
    normal_data = np.random.multivariate_normal(
        mean=[0,0,0,0],
        cov=np.eye(4),
        size=int(n_samples * (1-contamination))
    )
    
    anomaly_data = np.random.multivariate_normal(
        mean = [3,3,3,3],
        cov = np.eye(4) * 2,
        size = int(n_samples * contamination)
    )

    #TODO: entender melhor os sinais negativos
    X = np.vstack([normal_data, anomaly_data])
    y = np.hstack([
        np.ones(len(normal_data)),
        -np.ones(len(anomaly_data))
    ])

    indices = np.random.permutation(len(X))
    return X[indices], y[indices]
    

def run_isolation_forest_experiment(contamination=0.1, n_estimators=100):
    
    with mlflow.start_run(run_name=f"isolationForest_cont{contamination}"):
        mlflow.log_param("algorithm", "IsolationForest")
        mlflow.log_param("contamination", contamination)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("n_samples", 10000)

        X, y_true = generate_iot_like_data(contamination=contamination)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_true, test_size=0.3, random_state=42
        )

        model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42
        )

        model.fit(X_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_accuracy = sum(y_pred_train == y_train) / len(y_train)
        test_accuracy = sum(y_pred_test == y_test) / len(y_train)

        scores_test = model.decision_function(X_test)
        auc_score = roc_auc_score(y_test == -1, -scores_test)

        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("auc_score", auc_score)


        plt.figure(figsize=(12,4))
    
        # Plot 1: Distribuição de scores
        plt.subplot(1, 3, 1)
        plt.hist(scores_test[y_test == 1], alpha=0.7, label='Normal', bins=30)
        plt.hist(scores_test[y_test == -1], alpha=0.7, label='Anomaly', bins=30)
        plt.xlabel('Isolation Score')
        plt.ylabel('Frequency')
        plt.title('Score Distribution')
        plt.legend()
        
        # Plot 2: Confusion Matrix visual
        plt.subplot(1, 3, 2)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred_test)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Plot 3: Features scatter (primeiras 2 dimensões)
        plt.subplot(1, 3, 3)
        normal_mask = y_test == 1
        anomaly_mask = y_test == -1
        plt.scatter(X_test[normal_mask, 0], X_test[normal_mask, 1], 
                alpha=0.6, label='Normal', s=20)
        plt.scatter(X_test[anomaly_mask, 0], X_test[anomaly_mask, 1], 
                alpha=0.8, label='Anomaly', s=20, c='red')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Feature Space')
        plt.legend()
        
        plt.tight_layout()
        #TODO: salvar figuras na pasta de tracking e inserir um id para cada run (timestamp)
        plt.savefig('experiment_visualization.png', dpi=150, bbox_inches='tight')

        dataset_info = {
            "n_samples": len(X),
            "n_features": X.shape[1],
            "contamination_real": sum(y_true == -1) / len(y_true),
            "train_size": len(X_train),
            "test_size": len(X_test)
        }

        for key, value in dataset_info.items():
            mlflow.log_param(f"data_{key}", value)
        
        print(f"Experiment finished")
        print(f"Test Accuracy: {test_accuracy:.3f}",)
        print(f"AUC score: {auc_score:.3f}")

        return model, test_accuracy, auc_score


if __name__ == "__main__":
    run_id = int(time.time())

    #TODO: usar postgresql ao invés de sqlite
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(f"IoT-IDS-Baseline-{run_id}")

    results = []

    contaminations = [0.05, 0.1, 0.15, 0.2]
    n_estimators_list = [50, 100, 200]

    for cont in contaminations:
        for n_est in n_estimators_list:
            print(f"\n Running: contamination={cont}, n_estimators={n_est}")
            model, acc, auc = run_isolation_forest_experiment(cont, n_est)
            results.append({
                "contamination": cont,
                "n_estimators": n_est,
                "accuracy": acc,
                "auc": auc
            })
    
    results_df = pd.DataFrame(results)
    print("\n Experiments summary")
    print(results_df.round(3))
    print(f"\n Best setup: {results_df.loc[results_df['auc'].idxmax()]}")
