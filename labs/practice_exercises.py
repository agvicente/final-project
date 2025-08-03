#!/usr/bin/env python3
"""
Exercícios Práticos - Lab 01: Python Workspace Setup
Exercícios hands-on para consolidar o aprendizado

Uso: python practice_exercises.py
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

def exercise_1_environment_check():
    """Exercício 1: Verificação de ambiente"""
    print("🔬 EXERCÍCIO 1: Verificação de Ambiente")
    print("=" * 50)
    
    # Verificar se está em ambiente virtual
    if sys.prefix == sys.base_prefix:
        print("❌ Você NÃO está em um ambiente virtual!")
        print("💡 Para criar e ativar:")
        print("   python -m venv test_env")
        print("   source test_env/bin/activate  # Linux/macOS")
        print("   test_env\\Scripts\\activate     # Windows")
        return False
    else:
        print("✅ Você está em um ambiente virtual!")
        print(f"📍 Localização: {sys.prefix}")
        
    # Verificar versão do Python
    print(f"🐍 Python version: {sys.version}")
    
    # Verificar pip
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"📦 pip version: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ pip não encontrado!")
        return False
    
    print("\n✅ Exercício 1 completo!")
    return True

def exercise_2_package_management():
    """Exercício 2: Gerenciamento de pacotes"""
    print("\n🔬 EXERCÍCIO 2: Gerenciamento de Pacotes")
    print("=" * 50)
    
    # Listar pacotes instalados
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                              capture_output=True, text=True, check=True)
        packages = result.stdout.split('\n')
        print(f"📦 Pacotes instalados: {len(packages)-3}")  # -3 para header e linhas vazias
        
        # Verificar se tem pandas
        if 'pandas' in result.stdout:
            print("✅ pandas está instalado")
        else:
            print("❌ pandas não está instalado")
            print("💡 Instale com: pip install pandas")
            
    except subprocess.CalledProcessError:
        print("❌ Erro ao listar pacotes")
        return False
    
    # Verificar requirements.txt
    if Path("requirements.txt").exists():
        print("✅ requirements.txt encontrado")
        with open("requirements.txt", "r") as f:
            reqs = f.readlines()
        print(f"📋 Dependências listadas: {len(reqs)}")
    else:
        print("❌ requirements.txt não encontrado")
        print("💡 Crie com: pip freeze > requirements.txt")
    
    print("\n✅ Exercício 2 completo!")
    return True

def exercise_3_project_structure():
    """Exercício 3: Estrutura de projeto"""
    print("\n🔬 EXERCÍCIO 3: Estrutura de Projeto")
    print("=" * 50)
    
    expected_folders = [
        "data", "notebooks", "src", "results", "docs", "tests"
    ]
    
    current_structure = []
    for item in Path(".").iterdir():
        if item.is_dir() and not item.name.startswith('.') and not item.name.endswith('_env'):
            current_structure.append(item.name)
    
    print(f"📁 Pastas encontradas: {current_structure}")
    
    missing_folders = [folder for folder in expected_folders if folder not in current_structure]
    
    if missing_folders:
        print(f"❌ Pastas em falta: {missing_folders}")
        print("💡 Crie com:")
        for folder in missing_folders:
            print(f"   mkdir {folder}")
    else:
        print("✅ Todas as pastas esperadas estão presentes!")
    
    # Verificar se tem __init__.py em src
    if Path("src").exists():
        if Path("src/__init__.py").exists():
            print("✅ src/__init__.py encontrado")
        else:
            print("❌ src/__init__.py não encontrado")
            print("💡 Crie com: touch src/__init__.py")
    
    print("\n✅ Exercício 3 completo!")
    return True

def exercise_4_git_integration():
    """Exercício 4: Integração com Git"""
    print("\n🔬 EXERCÍCIO 4: Integração com Git")
    print("=" * 50)
    
    # Verificar se Git está disponível
    try:
        result = subprocess.run(['git', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"✅ {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Git não está instalado ou não está no PATH")
        return False
    
    # Verificar se é repositório Git
    if Path(".git").exists():
        print("✅ Repositório Git inicializado")
        
        # Verificar .gitignore
        if Path(".gitignore").exists():
            print("✅ .gitignore encontrado")
            with open(".gitignore", "r") as f:
                gitignore_content = f.read()
            
            important_ignores = ["__pycache__/", "*.pyc", "*_env/", ".ipynb_checkpoints/"]
            missing_ignores = [ignore for ignore in important_ignores if ignore not in gitignore_content]
            
            if missing_ignores:
                print(f"⚠️  Patterns importantes ausentes no .gitignore: {missing_ignores}")
            else:
                print("✅ .gitignore contém patterns importantes")
        else:
            print("❌ .gitignore não encontrado")
    else:
        print("❌ Não é um repositório Git")
        print("💡 Inicialize com: git init")
    
    print("\n✅ Exercício 4 completo!")
    return True

def exercise_5_reproducibility_test():
    """Exercício 5: Teste de reprodutibilidade"""
    print("\n🔬 EXERCÍCIO 5: Teste de Reprodutibilidade")
    print("=" * 50)
    
    # Criar ambiente temporário para teste
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"🧪 Criando ambiente de teste em: {temp_dir}")
        
        # Copiar requirements.txt se existir
        if Path("requirements.txt").exists():
            import shutil
            shutil.copy("requirements.txt", temp_dir)
            
            # Simular criação de novo ambiente
            test_env_path = Path(temp_dir) / "test_env"
            
            try:
                # Criar ambiente virtual de teste
                subprocess.run([sys.executable, '-m', 'venv', str(test_env_path)], 
                             check=True)
                print("✅ Ambiente virtual de teste criado")
                
                # Verificar se requirements podem ser instalados
                if sys.platform.startswith('win'):
                    pip_path = test_env_path / "Scripts" / "pip"
                else:
                    pip_path = test_env_path / "bin" / "pip"
                
                # Instalar requirements (dry-run)
                result = subprocess.run([str(pip_path), 'install', '--dry-run', '-r', 
                                       str(Path(temp_dir) / "requirements.txt")], 
                                      capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("✅ Requirements.txt é válido e instalável")
                else:
                    print("❌ Problemas com requirements.txt")
                    print(f"Erro: {result.stderr[:200]}...")
                    
            except subprocess.CalledProcessError as e:
                print(f"❌ Erro no teste: {e}")
                return False
        else:
            print("❌ requirements.txt não encontrado - impossível testar reprodutibilidade")
            return False
    
    print("\n✅ Exercício 5 completo!")
    return True

def exercise_6_best_practices_check():
    """Exercício 6: Verificação de best practices"""
    print("\n🔬 EXERCÍCIO 6: Best Practices Check")
    print("=" * 50)
    
    score = 0
    total_checks = 10
    
    # Check 1: Ambiente virtual ativo
    if sys.prefix != sys.base_prefix:
        print("✅ Trabalhando em ambiente virtual")
        score += 1
    else:
        print("❌ Não está em ambiente virtual")
    
    # Check 2: requirements.txt existe
    if Path("requirements.txt").exists():
        print("✅ requirements.txt presente")
        score += 1
    else:
        print("❌ requirements.txt ausente")
    
    # Check 3: .gitignore existe
    if Path(".gitignore").exists():
        print("✅ .gitignore presente")
        score += 1
    else:
        print("❌ .gitignore ausente")
    
    # Check 4: Estrutura de pastas
    essential_folders = ["data", "src"]
    if all(Path(folder).exists() for folder in essential_folders):
        print("✅ Estrutura de pastas básica presente")
        score += 1
    else:
        print("❌ Estrutura de pastas incompleta")
    
    # Check 5: __init__.py em src
    if Path("src/__init__.py").exists():
        print("✅ src é um pacote Python válido")
        score += 1
    else:
        print("❌ src/__init__.py ausente")
    
    # Check 6: Documentação
    doc_files = ["README.md", "SETUP_INSTRUCTIONS.md", "setup.py"]
    if any(Path(doc).exists() for doc in doc_files):
        print("✅ Documentação presente")
        score += 1
    else:
        print("❌ Documentação ausente")
    
    # Check 7: Sem arquivos temporários
    temp_patterns = [".pyc", "__pycache__", ".DS_Store"]
    temp_found = []
    for pattern in temp_patterns:
        if any(Path(".").rglob(f"*{pattern}*")):
            temp_found.append(pattern)
    
    if not temp_found:
        print("✅ Sem arquivos temporários")
        score += 1
    else:
        print(f"❌ Arquivos temporários encontrados: {temp_found}")
    
    # Check 8: Requirements com versões fixas
    if Path("requirements.txt").exists():
        with open("requirements.txt", "r") as f:
            reqs = f.read()
        if "==" in reqs:
            print("✅ Requirements com versões fixas")
            score += 1
        else:
            print("❌ Requirements sem versões fixas")
    else:
        print("❌ Requirements.txt não encontrado")
    
    # Check 9: Separação data/code
    data_in_src = any(Path("src").rglob("*.csv")) or any(Path("src").rglob("*.json"))
    if not data_in_src:
        print("✅ Dados separados do código")
        score += 1
    else:
        print("❌ Dados misturados com código")
    
    # Check 10: Jupyter notebooks organizados
    notebooks_in_root = any(Path(".").glob("*.ipynb"))
    if not notebooks_in_root or Path("notebooks").exists():
        print("✅ Notebooks organizados")
        score += 1
    else:
        print("❌ Notebooks desorganizados")
    
    print(f"\n📊 SCORE: {score}/{total_checks} ({score/total_checks*100:.0f}%)")
    
    if score >= 8:
        print("🏆 EXCELENTE! Você domina as best practices!")
    elif score >= 6:
        print("👍 BOM! Algumas melhorias ainda podem ser feitas")
    else:
        print("📚 PRECISA MELHORAR! Revise o laboratório")
    
    print("\n✅ Exercício 6 completo!")
    return score >= 6

def main():
    """Executa todos os exercícios"""
    print("🎓 EXERCÍCIOS PRÁTICOS - PYTHON WORKSPACE SETUP")
    print("=" * 60)
    print("Este script testa seu conhecimento dos conceitos do Lab 01")
    print("Execute dentro do ambiente de projeto que você criou")
    print("=" * 60)
    
    exercises = [
        exercise_1_environment_check,
        exercise_2_package_management, 
        exercise_3_project_structure,
        exercise_4_git_integration,
        exercise_5_reproducibility_test,
        exercise_6_best_practices_check
    ]
    
    results = []
    
    for exercise in exercises:
        try:
            result = exercise()
            results.append(result)
        except Exception as e:
            print(f"❌ Erro no exercício: {e}")
            results.append(False)
        
        input("\n⏸️  Pressione Enter para continuar...")
    
    # Resultado final
    print("\n🎯 RESULTADO FINAL")
    print("=" * 30)
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Exercícios aprovados: {passed}/{total}")
    print(f"📊 Taxa de sucesso: {passed/total*100:.0f}%")
    
    if passed == total:
        print("🏆 PARABÉNS! Você dominou completamente o setup de workspace Python!")
        print("🚀 Você está pronto para aplicar no seu projeto de mestrado!")
    elif passed >= total * 0.8:
        print("👍 MUITO BOM! Você tem uma base sólida.")
        print("💡 Revise os pontos que falharam para aperfeiçoamento.")
    else:
        print("📚 PRECISA REVISAR! Recomendamos estudar novamente o Lab 01.")
        print("🔄 Execute os exercícios novamente após as correções.")
    
    print("\n💡 Para aprofundar, consulte:")
    print("- Real Python: https://realpython.com/python-virtual-environments-a-primer/")
    print("- Python.org: https://docs.python.org/3/tutorial/venv.html")
    print("- Lab 01 completo: labs/lab01_python_workspace_setup.md")

if __name__ == "__main__":
    main()