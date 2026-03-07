#!/bin/bash
# Helper script para gerenciar Kafka
# Uso: ./kafka.sh [start|stop|status|logs|ui]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

case "$1" in
    start)
        echo "Iniciando Kafka..."
        docker-compose up -d
        echo "Aguardando Kafka ficar healthy..."
        sleep 5
        docker-compose ps
        ;;
    start-ui)
        echo "Iniciando Kafka com UI..."
        docker-compose --profile ui up -d
        echo "Kafka UI disponivel em: http://localhost:8080"
        ;;
    stop)
        echo "Parando Kafka..."
        docker-compose down
        ;;
    status)
        docker-compose ps
        ;;
    logs)
        docker-compose logs -f kafka
        ;;
    topics)
        docker exec iot-kafka kafka-topics --list --bootstrap-server localhost:9092
        ;;
    clean)
        echo "Removendo volumes (dados serao perdidos)..."
        docker-compose down -v
        ;;
    test)
        echo "Testando conexao com Kafka..."
        docker exec iot-kafka kafka-topics --list --bootstrap-server localhost:9092
        if [ $? -eq 0 ]; then
            echo "Kafka esta funcionando!"
        else
            echo "Erro: Kafka nao esta respondendo"
            exit 1
        fi
        ;;
    *)
        echo "Uso: $0 {start|start-ui|stop|status|logs|topics|clean|test}"
        echo ""
        echo "Comandos:"
        echo "  start     - Inicia Kafka (sem UI)"
        echo "  start-ui  - Inicia Kafka com interface web (localhost:8080)"
        echo "  stop      - Para todos os containers"
        echo "  status    - Mostra status dos containers"
        echo "  logs      - Mostra logs do Kafka"
        echo "  topics    - Lista topicos existentes"
        echo "  clean     - Remove containers E volumes (perde dados)"
        echo "  test      - Testa conexao com Kafka"
        exit 1
        ;;
esac
