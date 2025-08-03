#!/usr/bin/env python3
"""
Quick Setup Script para Laboratório Python Workspace
Automatiza a criação de ambiente virtual e instalação de dependências

Uso: python quick_setup.py [nome_do_projeto]
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path

def run_command(command, check=True):
    """Executa comando e retorna resultado"""
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {command}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro: {command}")
        print(f"   {e.stderr}")
        return None

def create_project_structure(project_name):
    """Cria estrutura de pastas do projeto"""
    folders = [
        "data/raw",
        "data/processed", 
        "data/samples",
        "notebooks/eda",
        "notebooks/experiments",
        "src/preprocessing",
        "src/models", 
        "src/evaluation",
        "src/utils",
        "results/figures",
        "results/metrics",
        "results/models",
        "docs",
        "tests"
    ]
    
    for folder in folders:
        (Path(project_name) / folder).mkdir(parents=True, exist_ok=True)
        # Criar .gitkeep para pastas vazias
        (Path(project_name) / folder / ".gitkeep").touch()
    
    # Criar __init__.py files
    init_files = [
        "src/__init__.py",
        "src/preprocessing/__init__.py", 
        "src/models/__init__.py",
        "src/evaluation/__init__.py",
        "src/utils/__init__.py"
    ]
    
    for init_file in init_files:
        (Path(project_name) / init_file).touch()

def create_requirements(project_name, project_type="research"):
    """Cria requirements.txt baseado no tipo de projeto"""
    
    if project_type == "research":
        requirements = """# Core Data Science
pandas==1.5.3
numpy==1.24.3
matplotlib==3.7.1
seaborn==0.12.2

# Machine Learning  
scikit-learn==1.2.2
imbalanced-learn==0.10.1

# Experiment Tracking
mlflow==2.3.1

# Jupyter
jupyter==1.0.0
ipywidgets==8.0.6

# Development
pytest==7.3.1
black==23.3.0
flake8==6.0.0

# Data Validation
pandas-profiling==3.6.6
"""
    else:  # basic
        requirements = """# Basic Data Science
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.7.0
jupyter>=1.0.0
pytest>=7.0.0
"""
    
    with open(Path(project_name) / "requirements.txt", "w") as f:
        f.write(requirements)

def create_gitignore(project_name):
    """Cria .gitignore otimizado"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# Virtual Environment
{project_name}_env/

# Data (grandes arquivos)
data/raw/*.csv
data/raw/*.json
data/raw/*.parquet
data/processed/*.csv
data/processed/*.json
data/processed/*.parquet

# Results
results/figures/*.png
results/figures/*.pdf
results/models/*.pkl
results/models/*.joblib

# Jupyter
.ipynb_checkpoints/

# MLflow
mlruns/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
""".format(project_name=project_name)
    
    with open(Path(project_name) / ".gitignore", "w") as f:
        f.write(gitignore_content)

def create_setup_instructions(project_name):
    """Cria instruções de setup para colaboradores"""
    instructions = f"""# {project_name.title()} - Setup Instructions

## Pré-requisitos
- Python 3.8+
- Git (opcional, mas recomendado)

## Setup Rápido

### 1. Clonar/Baixar o Projeto
```bash
# Se usando Git:
git clone <repository-url>
cd {project_name}

# Ou simplesmente navegue até a pasta:
cd {project_name}
```

### 2. Criar Ambiente Virtual
```bash
# Criar ambiente virtual
python -m venv {project_name}_env

# Ativar ambiente
# Linux/macOS:
source {project_name}_env/bin/activate

# Windows (Command Prompt):
{project_name}_env\\Scripts\\activate.bat

# Windows (PowerShell):  
{project_name}_env\\Scripts\\Activate.ps1
```

### 3. Instalar Dependências
```bash
# Instalar todas as dependências
pip install -r requirements.txt

# Verificar instalação
pip list
```

### 4. Validar Setup
```bash
# Testar que tudo está funcionando
python -c "import pandas, numpy, matplotlib; print('✅ Setup completo!')"

# Iniciar Jupyter (se instalado)
jupyter lab
```

## Estrutura do Projeto
```
{project_name}/
├── {project_name}_env/        # Ambiente virtual (não versionado)
├── data/                      # Datasets
│   ├── raw/                   # Dados originais
│   ├── processed/             # Dados processados
│   └── samples/               # Amostras para teste
├── notebooks/                 # Jupyter notebooks
│   ├── eda/                   # Análise exploratória
│   └── experiments/           # Experimentos
├── src/                       # Código fonte
│   ├── preprocessing/         # Pré-processamento
│   ├── models/                # Modelos
│   ├── evaluation/            # Avaliação
│   └── utils/                 # Utilidades
├── results/                   # Resultados
│   ├── figures/               # Gráficos
│   ├── metrics/               # Métricas
│   └── models/                # Modelos salvos
├── docs/                      # Documentação
├── tests/                     # Testes
├── requirements.txt           # Dependências
├── .gitignore                 # Arquivos ignorados pelo Git
└── SETUP_INSTRUCTIONS.md      # Este arquivo
```

## Troubleshooting

### Problema: "python command not found"
**Solução**: Instale Python 3.8+ do site oficial python.org

### Problema: "ModuleNotFoundError" 
**Solução**: 
1. Certifique-se que o ambiente virtual está ativo (deve aparecer `({project_name}_env)` no prompt)
2. Execute: `pip install -r requirements.txt`

### Problema: Jupyter não abre
**Solução**:
1. Verifique se está instalado: `jupyter --version`
2. Se não: `pip install jupyter`
3. Tente: `jupyter lab` ou `jupyter notebook`

### Problema: Permissões no Windows
**Solução**: 
1. Execute o PowerShell como Administrador
2. Execute: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

## Adicionando Novas Dependências

```bash
# Instalar nova biblioteca
pip install nome-da-biblioteca

# Atualizar requirements.txt
pip freeze > requirements.txt

# Commitar mudança (se usando Git)
git add requirements.txt
git commit -m "Add nome-da-biblioteca dependency"
```

## Colaboração

1. **Sempre** trabalhe dentro do ambiente virtual ativado
2. **Sempre** atualize requirements.txt quando instalar novas bibliotecas
3. **Nunca** versione a pasta do ambiente virtual ({project_name}_env/)
4. **Sempre** teste que outros podem reproduzir o setup

---

💡 **Dica**: Este projeto foi criado usando o Lab 01 de Python Workspace Setup. 
Para aprender mais sobre configuração de ambientes Python, consulte o laboratório completo.
"""
    
    with open(Path(project_name) / "SETUP_INSTRUCTIONS.md", "w") as f:
        f.write(instructions)

def main():
    parser = argparse.ArgumentParser(description='Setup automático de projeto Python')
    parser.add_argument('project_name', nargs='?', default='python-project',
                       help='Nome do projeto (default: python-project)')
    parser.add_argument('--type', choices=['basic', 'research'], default='research',
                       help='Tipo de projeto (default: research)')
    parser.add_argument('--no-venv', action='store_true',
                       help='Não criar ambiente virtual automaticamente')
    
    args = parser.parse_args()
    project_name = args.project_name
    
    print(f"🚀 Criando projeto: {project_name}")
    print(f"📁 Tipo: {args.type}")
    print("=" * 50)
    
    # Verificar se pasta já existe
    if Path(project_name).exists():
        response = input(f"⚠️  Pasta '{project_name}' já existe. Continuar? (y/N): ")
        if response.lower() != 'y':
            print("❌ Operação cancelada")
            return
    
    # Criar estrutura do projeto
    print("📁 Criando estrutura de pastas...")
    create_project_structure(project_name)
    
    # Criar arquivos de configuração
    print("📝 Criando arquivos de configuração...")
    create_requirements(project_name, args.type)
    create_gitignore(project_name)
    create_setup_instructions(project_name)
    
    # Navegar para pasta do projeto
    os.chdir(project_name)
    
    # Criar ambiente virtual (se solicitado)
    if not args.no_venv:
        print("🔧 Criando ambiente virtual...")
        env_name = f"{project_name}_env"
        
        result = run_command(f"python -m venv {env_name}")
        if result and result.returncode == 0:
            print(f"✅ Ambiente virtual '{env_name}' criado")
            
            # Instruções para ativação
            print("\n🎯 PRÓXIMOS PASSOS:")
            print(f"1. cd {project_name}")
            print("2. Ativar ambiente virtual:")
            print(f"   # Linux/macOS: source {env_name}/bin/activate")  
            print(f"   # Windows: {env_name}\\Scripts\\activate")
            print("3. pip install -r requirements.txt")
            print("4. jupyter lab  # (para iniciar desenvolvimento)")
        else:
            print("❌ Erro ao criar ambiente virtual")
            print("💡 Execute manualmente: python -m venv {env_name}")
    
    # Inicializar Git (opcional)
    if run_command("git --version", check=False):
        response = input("\n🔧 Inicializar repositório Git? (Y/n): ")
        if response.lower() != 'n':
            run_command("git init")
            run_command("git add .")
            run_command('git commit -m "Initial project setup"')
    
    print("\n🎉 Projeto criado com sucesso!")
    print(f"📋 Consulte '{project_name}/SETUP_INSTRUCTIONS.md' para instruções detalhadas")

if __name__ == "__main__":
    main()