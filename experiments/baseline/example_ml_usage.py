#!/usr/bin/env python3
"""
🚀 EXEMPLO PRÁTICO: Como usar os dados pré-processados para Machine Learning

Este script demonstra como usar os dados gerados pelo preprocessing correto
(separação treino/teste ANTES da normalização) para treinar modelos de ML.
"""

import numpy as np
import pandas as pd
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_preprocessed_data():
    """
    Carrega os dados pré-processados gerados pelo pipeline correto
    """
    print("📂 Carregando dados pré-processados...")
    
    # Verificar se arquivos existem
    required_files = [
        '../data/processed/X_train.npy',
        '../data/processed/X_test.npy', 
        '../data/processed/y_train.npy',
        '../data/processed/y_test.npy',
        '../data/processed/scaler.pkl',
        '../data/processed/preprocessing_metadata.json'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("❌ Arquivos faltando. Execute primeiro: python3 dvc_preprocessing.py")
        print(f"Faltando: {missing_files}")
        return None
    
    # Carregar dados
    X_train = np.load('../data/processed/X_train.npy')
    X_test = np.load('../data/processed/X_test.npy')
    y_train = np.load('../data/processed/y_train.npy')
    y_test = np.load('../data/processed/y_test.npy')
    
    # Carregar metadados
    with open('../data/processed/preprocessing_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Carregar scaler
    scaler = joblib.load('../data/processed/scaler.pkl')
    
    print(f"✅ Dados carregados:")
    print(f"   Treino: {X_train.shape} features, {len(y_train)} labels")
    print(f"   Teste:  {X_test.shape} features, {len(y_test)} labels")
    print(f"   Features: {len(metadata['feature_columns'])}")
    print(f"   Split usado: {metadata['config']['test_size']*100:.0f}% teste")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'metadata': metadata
    }

def train_multiple_models(data):
    """
    Treina múltiplos modelos de ML e compara performance
    """
    print(f"\n🤖 Treinando múltiplos modelos de ML...")
    
    X_train, X_test = data['X_train'], data['X_test']
    y_train, y_test = data['y_train'], data['y_test']
    
    # Definir modelos
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            n_jobs=-1
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42,
            max_iter=1000
        ),
        'SVM (RBF)': SVC(
            random_state=42,
            probability=True  # Para predict_proba
        ),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            random_state=42,
            max_iter=500
        )
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n🔄 Treinando {name}...")
        
        # Treinar modelo
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Fazer predições
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_proba,
            'model': model
        }
        
        print(f"   Acurácia: {accuracy:.4f}")
        print(f"   F1-Score: {f1:.4f}")
    
    return results, trained_models

def analyze_results(results, y_test):
    """
    Analisa e compara os resultados dos modelos
    """
    print(f"\n📊 COMPARAÇÃO DOS MODELOS:")
    print("="*60)
    
    # Criar DataFrame com resultados
    comparison_data = []
    for name, result in results.items():
        comparison_data.append({
            'Modelo': name,
            'Acurácia': result['accuracy'],
            'Precisão': result['precision'], 
            'Recall': result['recall'],
            'F1-Score': result['f1_score']
        })
    
    df_results = pd.DataFrame(comparison_data)
    df_results = df_results.sort_values('F1-Score', ascending=False)
    
    print(df_results.to_string(index=False, float_format='%.4f'))
    
    # Identificar melhor modelo
    best_model_name = df_results.iloc[0]['Modelo']
    print(f"\n🏆 MELHOR MODELO: {best_model_name}")
    print(f"   F1-Score: {df_results.iloc[0]['F1-Score']:.4f}")
    
    # Report detalhado do melhor modelo
    best_result = results[best_model_name]
    print(f"\n📋 RELATÓRIO DETALHADO - {best_model_name}:")
    print(classification_report(y_test, best_result['predictions']))
    
    return best_model_name, best_result

def save_best_model(best_model_name, best_result, data):
    """
    Salva o melhor modelo para uso futuro
    """
    print(f"\n💾 Salvando melhor modelo: {best_model_name}")
    
    # Criar diretório de modelos
    os.makedirs('../data/models', exist_ok=True)
    
    # Salvar modelo
    model_file = '../data/models/best_model.pkl'
    joblib.dump(best_result['model'], model_file)
    
    # Salvar informações do modelo
    model_info = {
        'model_name': best_model_name,
        'accuracy': best_result['accuracy'],
        'f1_score': best_result['f1_score'],
        'precision': best_result['precision'],
        'recall': best_result['recall'],
        'feature_columns': data['metadata']['feature_columns'],
        'preprocessing_config': data['metadata']['config']
    }
    
    info_file = '../data/models/model_info.json'
    with open(info_file, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"   ✅ Modelo salvo: {model_file}")
    print(f"   ✅ Info salva: {info_file}")
    
    return model_file, info_file

def demo_production_usage(data):
    """
    Demonstra como usar o modelo em produção com novos dados
    """
    print(f"\n🔮 DEMO: Uso em Produção")
    print("="*40)
    
    # Simular novos dados (usando algumas amostras do teste)
    print("📝 Simulando novos dados de produção...")
    
    X_test = data['X_test']
    scaler = data['scaler']
    
    # Pegar algumas amostras "originais" (antes da normalização)
    # Para demonstração, vamos "desnormalizar" algumas amostras
    sample_indices = [0, 10, 50, 100, 200]
    X_samples = X_test[sample_indices]
    
    print(f"   📊 Processando {len(sample_indices)} amostras de exemplo...")
    
    # Carregar melhor modelo
    best_model = joblib.load('../data/models/best_model.pkl')
    
    # Fazer predições
    predictions = best_model.predict(X_samples)
    probabilities = best_model.predict_proba(X_samples)
    
    print(f"   ✅ Predições realizadas:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        confidence = prob.max()
        print(f"      Amostra {i+1}: Classe {pred} (confiança: {confidence:.3f})")
    
    return predictions, probabilities

def main():
    """
    Função principal que executa todo o pipeline de ML
    """
    print("🚀 PIPELINE COMPLETO DE MACHINE LEARNING")
    print("="*50)
    print("📌 Usando dados com separação CORRETA (sem data leakage)")
    
    # 1. Carregar dados pré-processados
    data = load_preprocessed_data()
    if data is None:
        return
    
    # 2. Treinar múltiplos modelos
    results, trained_models = train_multiple_models(data)
    
    # 3. Analisar resultados
    best_model_name, best_result = analyze_results(results, data['y_test'])
    
    # 4. Salvar melhor modelo
    model_file, info_file = save_best_model(best_model_name, best_result, data)
    
    # 5. Demo de uso em produção
    demo_production_usage(data)
    
    print(f"\n✅ PIPELINE CONCLUÍDO!")
    print(f"📁 Arquivos criados:")
    print(f"   - Melhor modelo: {model_file}")
    print(f"   - Info do modelo: {info_file}")
    print(f"📖 Veja o guia completo em: docs/preprocessing_guide.md")

if __name__ == "__main__":
    main()
