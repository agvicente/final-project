# 🧪 Laboratórios de Pesquisa Acadêmica

Este diretório contém laboratórios práticos desenvolvidos para apoiar o projeto de mestrado em **Detecção de Intrusão Baseada em Anomalias em Sistemas IoT com Clustering Evolutivo**.

## 📚 Laboratórios Disponíveis

### Lab 01: Python Workspace Setup
**Arquivo**: `lab01_python_workspace_setup.md`  
**Duração**: 4-6 horas  
**Nível**: Iniciante a Avançado  

**O que você aprenderá:**
- Configuração profissional de ambientes virtuais Python
- Gerenciamento de dependências reproduzível  
- Estruturação de projetos acadêmicos
- Integração com ferramentas de pesquisa (MLflow, Jupyter)
- Workflows colaborativos para ciência reproduzível

**Pré-requisitos:**
- Python 3.8+ instalado
- Conhecimento básico de linha de comando

---

## 🛠️ Ferramentas de Apoio

### Quick Setup Script
**Arquivo**: `quick_setup.py`  
**Uso**: `python quick_setup.py [nome_do_projeto]`

Script automatizado que cria:
- Estrutura completa de projeto
- Ambiente virtual configurado
- Requirements.txt otimizado
- .gitignore profissional
- Documentação de setup

**Exemplos de uso:**
```bash
# Projeto básico
python quick_setup.py meu-projeto --type basic

# Projeto de pesquisa completo  
python quick_setup.py iot-ids-research --type research

# Sem criar ambiente virtual automaticamente
python quick_setup.py meu-projeto --no-venv
```

### Exercícios Práticos
**Arquivo**: `practice_exercises.py`  
**Uso**: `python practice_exercises.py`

Sistema de validação interativo que testa:
- ✅ Configuração do ambiente virtual
- ✅ Gerenciamento de pacotes
- ✅ Estrutura de projeto
- ✅ Integração com Git
- ✅ Reprodutibilidade
- ✅ Best practices

**Score final:** 10 verificações automatizadas

---

## 🚀 Como Começar

### Opção 1: Laboratório Completo (Recomendado)
```bash
# 1. Leia o laboratório completo
open labs/lab01_python_workspace_setup.md

# 2. Siga todos os exercícios práticos
# 3. Valide com os exercícios de teste
cd seu-projeto
python ../labs/practice_exercises.py
```

### Opção 2: Setup Rápido
```bash
# Crie projeto automaticamente
python labs/quick_setup.py meu-projeto-mestrado

# Entre na pasta e ative ambiente
cd meu-projeto-mestrado
source meu-projeto-mestrado_env/bin/activate  # Linux/macOS

# Instale dependências
pip install -r requirements.txt

# Valide setup
python ../labs/practice_exercises.py
```

### Opção 3: Aplicação Direta ao Mestrado
```bash
# Setup específico para projeto IoT-IDS
python labs/quick_setup.py iot-ids-research --type research

cd iot-ids-research
source iot-ids-research_env/bin/activate

# Adicionar dependências específicas da Fase 1
pip install river mlflow wandb dvc great-expectations
pip freeze > requirements.txt

# Iniciar desenvolvimento
jupyter lab
```

---

## 📊 Cronograma de Uso

### Para Iniciantes (2-3 dias)
```
Dia 1: Lab 01 - Exercícios 1-3 (fundamentos)
Dia 2: Lab 01 - Exercícios 4-6 (aplicação prática)  
Dia 3: Validação com practice_exercises.py + setup real do projeto
```

### Para Experientes (1 dia)
```
Manhã: Review rápido do Lab 01 + quick_setup.py
Tarde: Setup do projeto real + validação + customizações
```

### Para o Projeto de Mestrado
```
Semana 1 da Fase 1: Setup usando este laboratório
- Aplicar lab01_python_workspace_setup.md
- Usar quick_setup.py para estrutura inicial
- Adaptar para necessidades específicas (CICIoT2023, MLflow, etc.)
```

---

## 🎯 Objetivos de Aprendizagem

Ao completar os laboratórios, você será capaz de:

### Conhecimentos Técnicos
- [ ] Criar e gerenciar ambientes virtuais Python profissionalmente
- [ ] Implementar dependency management reproduzível  
- [ ] Estruturar projetos seguindo padrões acadêmicos
- [ ] Integrar ferramentas de pesquisa (MLflow, DVC, Jupyter)
- [ ] Configurar workflows de colaboração científica

### Habilidades Práticas
- [ ] Setup de ambiente em <30 minutos
- [ ] Reprodução de ambiente em qualquer máquina
- [ ] Colaboração eficiente em projetos de pesquisa
- [ ] Debugging de problemas comuns de ambiente
- [ ] Aplicação de best practices industriais

### Aplicação ao Mestrado
- [ ] Ambiente robusto para experimentos da Fase 1
- [ ] Base sólida para desenvolvimento das Fases 2-4
- [ ] Infraestrutura para publicações reproduzíveis
- [ ] Setup adequado para colaboração com orientadores

---

## 🔍 Validação e Certificação

### Critérios de Aprovação
Para considerar o laboratório concluído com sucesso:

1. **Score ≥80% nos exercícios práticos**
2. **Ambiente reproduzível funcionando**
3. **Projeto estruturado seguindo padrões**
4. **Documentação completa e clara**
5. **Git integrado com .gitignore apropriado**

### Auto-Avaliação
```bash
# Execute para validação completa
python labs/practice_exercises.py

# Score esperado: ≥8/10 (80%+)
# Resultado "EXCELENTE" ou "BOM"
```

### Checklist Final
- [ ] Ambiente virtual ativo e funcionando
- [ ] Requirements.txt com versões fixas
- [ ] Estrutura de pastas profissional
- [ ] .gitignore configurado
- [ ] Jupyter Lab iniciando sem erros
- [ ] Código executa sem ModuleNotFoundError
- [ ] Outro computador pode reproduzir o setup
- [ ] Documentação permite setup independente

---

## 📚 Recursos Complementares

### Documentação Oficial
- [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)
- [Pip User Guide](https://pip.pypa.io/en/stable/user_guide/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

### Tutoriais Recomendados  
- [Real Python - Virtual Environments](https://realpython.com/python-virtual-environments-a-primer/)
- [Corey Schafer - Python venv Tutorial](https://www.youtube.com/watch?v=Kg1Yvry_Ydk)
- [Data Science Project Structure](https://drivendata.github.io/cookiecutter-data-science/)

### Ferramentas Avançadas
- **Poetry**: Gerenciamento moderno de dependências
- **Conda**: Ambientes para ciência de dados  
- **Docker**: Containerização para máxima reprodutibilidade
- **DVC**: Versionamento de dados e pipelines

---

## 🐛 Troubleshooting Comum

### Problema: "python command not found"
```bash
# Solução: Instalar Python
# Windows: Download do python.org
# macOS: brew install python3
# Ubuntu: sudo apt install python3 python3-pip
```

### Problema: Ambiente virtual não ativa
```bash
# Verifique o comando correto:
# Linux/macOS: source venv/bin/activate
# Windows CMD: venv\Scripts\activate.bat  
# Windows PowerShell: venv\Scripts\Activate.ps1
```

### Problema: ModuleNotFoundError
```bash
# 1. Verifique se ambiente está ativo (prompt deve mostrar (venv))
# 2. Instale dependências: pip install -r requirements.txt
# 3. Verifique se está no diretório correto
```

### Problema: Permissões no Windows
```bash
# Execute PowerShell como Administrador:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 🤝 Contribuição e Feedback

### Como Melhorar os Laboratórios
1. **Relate problemas**: Abra issues com erros encontrados
2. **Sugira melhorias**: Propose exercícios adicionais
3. **Compartilhe experiência**: Documente casos de uso específicos
4. **Contribua código**: PRs com correções e enhancements

### Contato
- **Projeto**: Mestrado em Detecção de Intrusão IoT
- **Contexto**: Fase 1 - Fundamentos e MVP
- **Objetivo**: Baseline científico reproduzível

---

## 📈 Roadmap

### Laboratórios Futuros Planejados
- **Lab 02**: MLflow para Tracking de Experimentos
- **Lab 03**: Análise Exploratória de Dados IoT  
- **Lab 04**: Implementação de Concept Drift Detection
- **Lab 05**: Clustering Evolutivo com Mixture of Typicalities
- **Lab 06**: Arquitetura de Streaming com Kafka

### Integração com Cronograma
Os laboratórios estão alinhados com o cronograma da Fase 1:
- **Semanas 1-2**: Lab 01 (Setup e fundamentos)
- **Semanas 3-4**: Lab 02 + Lab 03 (EDA e tracking)
- **Semanas 5-6**: Lab 04 (Baseline e avaliação)
- **Semanas 7-8**: Lab 05 (Concept drift)
- **Semanas 9-10**: Lab 06 (Aplicação prática)

---

**🎓 Este conjunto de laboratórios estabelece a base técnica sólida para todo o projeto de mestrado, garantindo reprodutibilidade científica e desenvolvimento eficiente das fases subsequentes.**