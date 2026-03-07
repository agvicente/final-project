#!/bin/bash
set -e

# Corrigir permissões do volume montado
if [ -d "/workspace" ]; then
    sudo chown -R researcher:researcher /workspace 2>/dev/null || true
    sudo chmod -R u+w /workspace 2>/dev/null || true
fi

echo "🚀 Iniciando ambiente de pesquisa IoT-IDS..."

# Verificar se é primeira execução
if [ ! -f "/workspace/.initialized" ]; then
    echo "📦 Configuração inicial..."
    
    # Inicializar DVC se não existir
    if [ ! -d "/workspace/.dvc" ]; then
        echo "   Inicializando DVC..."
        cd /workspace && dvc init --no-scm
    fi
    
    # Inicializar Git se não existir
    if [ ! -d "/workspace/.git" ]; then
        echo "   Inicializando Git..."
        cd /workspace && git init
    fi
    
    # Gerar dados sintéticos se não existirem
    if [ ! -f "/workspace/data/raw/iot_traffic_v1_small.csv" ]; then
        echo "   Gerando dados sintéticos..."
        cd /workspace && python generate_dataset.py
    fi
    
    touch /workspace/.initialized
    echo "✅ Configuração inicial concluída!"
fi

# Verificar componentes
echo "🔍 Verificando componentes..."
python -c "
import numpy, pandas, sklearn, mlflow, dvc
print('✅ Dependências principais OK')
"

# Iniciar MLflow server em background se solicitado
if [ "$1" = "with-mlflow" ]; then
    echo "🔬 Iniciando MLflow server..."
    mlflow server --backend-store-uri sqlite:///mlflow.db \
                  --default-artifact-root ./mlruns \
                  --host 0.0.0.0 \
                  --port 5000 &
    
    # Aguardar MLflow iniciar
    sleep 5
    shift  # Remove 'with-mlflow' dos argumentos
fi

# Executar comando solicitado
exec "$@"