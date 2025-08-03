# 🚀 Quick Start - Laboratório Python Workspace

## ⚡ Setup em 5 Minutos

### Para Usuários Experientes
```bash
# 1. Crie projeto automaticamente
python labs/quick_setup.py iot-research --type research

# 2. Entre e ative ambiente
cd iot-research
source iot-research_env/bin/activate  # Linux/macOS
# ou iot-research_env\Scripts\activate  # Windows

# 3. Instale dependências
pip install -r requirements.txt

# 4. Inicie Jupyter
jupyter lab

# 5. Valide (opcional)
python ../labs/practice_exercises.py
```

### Para Usuários Iniciantes
```bash
# 1. Leia o laboratório completo primeiro
open labs/lab01_python_workspace_setup.md

# 2. Siga os exercícios passo a passo
# 3. Use o script de setup quando se sentir confortável
```

---

## 📋 Checklist Rápido

### Antes de Começar
- [ ] Python 3.8+ instalado (`python --version`)
- [ ] pip funcionando (`pip --version`)  
- [ ] 2GB de espaço livre em disco
- [ ] Conexão com internet para downloads

### Após Setup
- [ ] Prompt mostra `(nome_env)` quando ativo
- [ ] `pip list` mostra pacotes instalados
- [ ] `jupyter lab` abre no navegador
- [ ] `python -c "import pandas, numpy; print('OK')"` funciona

---

## 🎯 Para Seu Projeto de Mestrado

### Setup Específico IoT-IDS
```bash
# Comando único para setup completo
python labs/quick_setup.py iot-ids-mestrado --type research

cd iot-ids-mestrado  
source iot-ids-mestrado_env/bin/activate

# Adicionar dependências específicas da Fase 1
pip install river==0.15.0 mlflow==2.3.1 dvc[all]==2.58.2
pip freeze > requirements.txt

# Inicializar Git
git init
git add .
git commit -m "Initial setup for IoT-IDS research"

# Validar tudo funcionando
python ../labs/practice_exercises.py
```

### Resultado Esperado
```
iot-ids-mestrado/
├── iot-ids-mestrado_env/     # Ambiente virtual
├── data/                     # Para CICIoT2023 dataset
├── notebooks/                # Jupyter notebooks  
├── src/                      # Código dos algoritmos
├── results/                  # Outputs dos experimentos
├── requirements.txt          # Dependências fixas
└── SETUP_INSTRUCTIONS.md     # Para colaboradores
```

---

## 🆘 Troubleshooting Express

| Problema | Solução Rápida |
|----------|----------------|
| `python: command not found` | Instale Python do python.org |
| Ambiente não ativa | Use o comando correto para seu SO |
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| Jupyter não abre | `pip install jupyter && jupyter lab` |
| Git não funciona | `git config --global user.name "Nome"` |

---

## 📞 Quando Precisar de Ajuda

1. **Leia primeiro**: `labs/lab01_python_workspace_setup.md`
2. **Teste automatizado**: `python labs/practice_exercises.py`  
3. **Documentação completa**: `labs/README.md`
4. **Recursos online**: Links no laboratório principal

---

**🏁 Meta: Ambiente profissional funcionando em <30 minutos!**