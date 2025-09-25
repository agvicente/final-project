import mlflow
import mlflow.sklearn
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import time

'''
Atualize o @algorithm_comparison para que ele implemente a comparação entre os algoritmos, 
mas usando os dados de saida do preprocessing ao inves dos dados gerados. 

Cada algoritmo de ml deve ser rodado com parametros diferentes, e cada rodada com parametros 
diferentes deve ser rodado um certo numero de vezes para rigor estatistico. 

O mlflow deve ser usado, junto com o run_id baseado 
em tipestamp para cada execução, da forma que está no exemplo do arquivo. 

Crie um arquivo dvc_baseline_experiment, que será incluido no pipeline dvc e irá rodar o experimento. 

Os resultados serão para escrever um artigo, os gráficos tabelas e dados relacionados aos experimentos 
e comparação entre modelos devem ser salvos na pasta experiments/results. Gere gráficos combinados apenas se eles forem 
correlacionados, se não for, gere graficos e tabelas infividuais. 

Baseie-se principalmente em accuracy, recall, precision e f1-score, mas verifique possiveis graficos 
que podem enriquecer as analises. 
Em um primeiro momento rode todos os algoritmos com parametros simples e uma pequena parte dos dados, 
certifique-se de que todos os algoritmos rodam e depois rode completo.
'''

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