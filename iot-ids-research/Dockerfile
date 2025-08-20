# Dockerfile para pesquisa IoT-IDS
FROM python:3.12-slim-bullseye

# Metadados
LABEL maintainer="custodioovicente@gmail.com"
LABEL description="Ambiente para pesquisa de detecção de intrusão IoT"
LABEL version="1.0"

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Criar usuário não-root para segurança
RUN groupadd -r researcher && useradd -r -g researcher researcher

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Definir diretório de trabalho
WORKDIR /workspace

# Copiar requirements primeiro (para cache de layers)
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copiar código da aplicação
COPY . .

# Criar diretórios necessários
RUN mkdir -p data/raw data/processed models experiments logs && \
    chown -R researcher:researcher /workspace

# Mudar para usuário não-root
USER researcher

# Configurar MLflow
ENV MLFLOW_TRACKING_URI=http://localhost:5000

# Expor portas
EXPOSE 8888 5000

# Script de inicialização
COPY docker-entrypoint.sh /usr/local/bin/
USER root
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
USER researcher

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]