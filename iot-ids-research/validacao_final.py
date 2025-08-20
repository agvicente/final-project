# validacao_final.py
import subprocess
import sys
import os

def validar_ambiente():
    """Valida se todo o ambiente está configurado corretamente"""
    
    testes = []
    
    # 1. Python e dependências
    try:
        import numpy, pandas, sklearn, mlflow, dvc
        testes.append(("✅", "Dependências Python"))
    except ImportError as e:
        testes.append(("❌", f"Dependências Python: {e}"))
    
    # 2. MLflow
    try:
        import requests
        resp = requests.get("http://localhost:5000", timeout=5)
        if resp.status_code == 200:
            testes.append(("✅", "MLflow server"))
        else:
            testes.append(("❌", "MLflow server não responde"))
    except:
        testes.append(("❌", "MLflow server não acessível"))
    
    # 3. DVC
    if os.path.exists('.dvc'):
        testes.append(("✅", "DVC inicializado"))
    else:
        testes.append(("❌", "DVC não inicializado"))
    
    # 4. Git
    if os.path.exists('.git'):
        testes.append(("✅", "Git inicializado"))
    else:
        testes.append(("❌", "Git não inicializado"))
    
    # 5. Docker
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            testes.append(("✅", "Docker disponível"))
        else:
            testes.append(("❌", "Docker não disponível"))
    except:
        testes.append(("❌", "Docker não encontrado"))
    
    # 6. Estrutura de diretórios
    dirs_necessarios = ['data/raw', 'data/processed', 'models', 'experiments']
    dirs_ok = all(os.path.exists(d) for d in dirs_necessarios)
    if dirs_ok:
        testes.append(("✅", "Estrutura de diretórios"))
    else:
        testes.append(("❌", "Estrutura de diretórios incompleta"))
    
    # Relatório
    print("🔍 Validação do Ambiente de Pesquisa")
    print("=" * 40)
    for status, descricao in testes:
        print(f"{status} {descricao}")
    
    sucessos = sum(1 for status, _ in testes if status == "✅")
    total = len(testes)
    
    print(f"\n📊 Score: {sucessos}/{total} ({sucessos/total*100:.1f}%)")
    
    if sucessos == total:
        print("\n🎉 Parabéns! Ambiente completamente configurado!")
        print("🚀 Você está pronto para começar a pesquisa!")
        return True
    else:
        print(f"\n⚠️  {total-sucessos} itens precisam de atenção.")
        print("📝 Revise a configuração dos itens marcados com ❌")
        return False

if __name__ == "__main__":
    validar_ambiente()