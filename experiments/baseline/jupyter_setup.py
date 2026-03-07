import subprocess
import sys

def install_extension(extension_name, pip_package=None):
    """Instala extensão do Jupyter Lab"""
    try:
        if pip_package:
            subprocess.run([sys.executable, '-m', 'pip', 'install', pip_package], check=True)
        
        subprocess.run(['jupyter', 'labextension', 'install', extension_name], check=True)
        print(f"✅ {extension_name} instalado com sucesso")
    except subprocess.CalledProcessError:
        print(f"❌ Falha ao instalar {extension_name}")

def setup_jupyter_lab():
    """Configura Jupyter Lab com extensões para pesquisa"""
    
    print("🔧 Configurando Jupyter Lab para pesquisa científica...")
    
    # Extensões essenciais
    extensions = [
        # Produtividade
        ('jupyterlab-git', 'jupyterlab-git'),  # Integração Git
        ('jupyterlab_code_formatter', 'jupyterlab_code_formatter'),  # Formatação de código
        ('@jupyterlab/debugger', None),  # Debugger visual
        
        # Visualização
        ('jupyterlab-plotly', 'jupyterlab-plotly'),  # Plotly interativo
        ('@jupyter-widgets/jupyterlab-manager', None),  # Widgets
        ('jupyterlab-matplotlib', 'ipympl'),  # Matplotlib interativo
        
        # Data Science
        ('@lckr/jupyterlab_variableinspector', None),  # Inspetor de variáveis
        ('jupyterlab-spreadsheet', None),  # Visualizar Excel
        
        # Sistema
        ('jupyterlab-system-monitor', 'jupyterlab-system-monitor'),  # Monitor CPU/RAM
        ('jupyterlab-topbar-extension', 'jupyterlab-topbar'),  # Barra superior
        
        # Documentação
        ('@jupyterlab/latex', 'jupyterlab-latex'),  # LaTeX
        ('@jupyterlab/toc', None),  # Índice automático
        
        # ML/AI específico
        ('jupyterlab-tensorboard', 'jupyterlab_tensorboard'),  # TensorBoard
    ]
    
    for ext_name, pip_pkg in extensions:
        install_extension(ext_name, pip_pkg)
    
    # Configurações personalizadas
    config = {
        "CodeCell": {
            "cm_config": {
                "lineNumbers": True,
                "foldCode": True,
                "highlightSelectionMatches": True
            }
        },
        "NotebookApp": {
            "nbserver_extensions": {
                "jupyterlab": True,
                "jupyterlab_git": True
            }
        }
    }
    
    print("\n🎨 Aplicando configurações personalizadas...")
    
    # Configurar formatação automática
    formatter_config = """
c.JupyterLabCodeFormatter.black_config = {
    'line_length': 88,
    'target_versions': ['py38'],
    'include': '\\.pyi?$',
    'exclude': '''
    /(
        \\.eggs
        | \\.git
        | \\.mypy_cache
        | \\.tox
        | \\.venv
        | _build
        | buck-out
        | build
        | dist
    )/
    '''
}
"""
    
    with open('jupyter_lab_config.py', 'w') as f:
        f.write(formatter_config)
    
    print("✅ Configuração do Jupyter Lab concluída!")
    print("\n📝 Para usar:")
    print("   1. Jupyter Lab: jupyter lab")
    print("   2. Formatação: Ctrl+Shift+I (Black)")
    print("   3. Git: Aba lateral esquerda")
    print("   4. Debugger: Aba lateral direita")
    print("   5. Variáveis: View > Inspector")

if __name__ == "__main__":
    setup_jupyter_lab()