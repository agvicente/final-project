# 🚀 Quick Start Guide - Sistema de Pesquisa Acelerada

## O Que Foi Criado

Sistema completo de desenvolvimento acelerado para sua pesquisa de mestrado:

### ✅ 9 Skills Especializadas
1. **iot-ids-research-context** - Contexto completo do projeto (sempre ativa)
2. **evolutionary-clustering-guide** - Ensina clustering evolutivo (K-means → Maia et al.)
3. **kafka-streaming-iot** - Guia de streaming com Kafka
4. **paper-reading-accelerator** - Resume papers rapidamente
5. **experiment-design-validator** - Valida rigor científico
6. **scientific-paper-writer** - Escreve papers incrementalmente
7. **dissertation-writer** - Escreve dissertação (PT + EN)
8. **overleaf-formatter-artigo** - Mantém formatação do artigo1
9. **overleaf-formatter-dissertation** - Mantém formatação da dissertação

### ✅ 3 Hooks de Automação
1. **session-start** - Carrega contexto ao abrir Claude Code
2. **session-end** - Salva progresso ao fechar
3. **auto-save** - Salva a cada 10min (proteção contra travamento)

### ✅ 4 Comandos Úteis
1. **/resume** - Mostra contexto atual e próximos passos
2. **/start-sprint** - Inicia nova semana de trabalho
3. **/finalize-week** - Gera relatório para reunião com orientador
4. **/paper-summary [nome]** - Resume paper do Zotero

### ✅ Documentação Evolutiva
- **SESSION_CONTEXT.md** - "Cérebro" do projeto (contexto permanente)
- **current-week.md** - Relatório semanal vivo
- **progress/** - Logs de cada sessão
- **decisions/** - Decisões técnicas importantes

---

## Como Usar (Primeiros Passos)

### 1. Configurar Zotero (5 minutos)

**Leia:** `/Users/augusto/mestrado/ZOTERO_SETUP.md`

**Ação:** Instale Better BibTeX e configure auto-export para:
```
/Users/augusto/mestrado/references.bib
```

### 2. Testar Sistema (2 minutos)

Abra nova sessão do Claude Code neste diretório e digite:

```
/resume
```

Você deve ver um resumo do contexto atual.

### 3. Iniciar Primeira Sprint (5 minutos)

```
/start-sprint
```

Claude vai perguntar qual o objetivo da semana e criar um plano.

---

## Workflow Diário

### Segunda-feira (Início da semana):
```
/start-sprint
```
Define objetivo e plano da semana.

### Durante a semana (Cada sessão):
```
/resume
```
Carrega contexto e continua de onde parou.

### Quando precisar ler um paper:
```
/paper-summary Maia et al 2020
```
Resume o paper focando em implementação.

### Sexta ou antes da reunião:
```
/finalize-week
```
Gera relatório completo para o orientador.

---

## Proteção Contra Travamento

**Auto-save roda a cada 10 minutos automaticamente.**

Se o terminal travar:
1. Reinicie a máquina
2. Abra Claude Code novamente
3. Digite: `/resume`
4. Claude detecta sessão interrompida e pergunta se quer recuperar
5. Diga "sim" para continuar de onde parou

Trabalho salvo em: branch `wip/auto-save`

---

## Estrutura de Arquivos

```
mestrado/
├── final-project/
│   ├── docs/
│   │   ├── SESSION_CONTEXT.md          ← LEIA ESTE ARQUIVO SEMPRE
│   │   ├── weekly-reports/
│   │   │   └── current-week.md         ← Relatório vivo da semana
│   │   ├── progress/                   ← Logs de sessões
│   │   └── decisions/                  ← Decisões técnicas
│   ├── .claude/
│   │   ├── skills/                     ← 9 skills criadas
│   │   ├── hooks/                      ← 3 hooks de automação
│   │   └── commands/                   ← 4 comandos úteis
│   └── iot-ids-research/               ← Código da pesquisa
│       ├── src/
│       │   ├── clustering/             ← Fase 2 (novo)
│       │   └── streaming/              ← Fase 3 (novo)
│       └── experiments/                ← Experimentos
├── artigo1/                            ← Paper baseline
├── dissertation/                       ← Dissertação
└── references.bib                      ← Zotero auto-export
```

---

## Skills Se Ativam Automaticamente

Você não precisa chamar as skills manualmente. Elas se ativam baseado no contexto:

- **Falar sobre clustering?** → `evolutionary-clustering-guide` ativa
- **Mencionar Kafka/streaming?** → `kafka-streaming-iot` ativa
- **Pedir resumo de paper?** → `paper-reading-accelerator` ativa
- **Editar artigo1?** → `overleaf-formatter-artigo` ativa
- **Editar dissertação?** → `overleaf-formatter-dissertation` ativa

**iot-ids-research-context está sempre ativa** = Claude sempre lembra do seu projeto.

---

## Próximos Passos Imediatos

1. ✅ Sistema configurado
2. ⏳ **Você:** Setup Zotero (ZOTERO_SETUP.md)
3. ⏳ **Você:** Testar com `/resume`
4. ⏳ **Você:** Rodar `/start-sprint` para Semana 1 da Fase 2
5. ⏳ Começar estudos de clustering (Week 1 - teoria 30%)

---

## Se Algo Der Errado

**Sessão travou?**
- `/resume` recupera automaticamente

**Perdeu contexto?**
- Leia: `docs/SESSION_CONTEXT.md`

**Esqueceu o que fazer?**
- `/resume` sempre te orienta

**Hook não funciona?**
- Ainda funciona sem hooks, só perde automação
- Continue trabalhando normalmente

---

## Comandos Rápidos para Copiar

```bash
# Testar sistema
/resume

# Iniciar semana
/start-sprint

# Resumir paper
/paper-summary Maia et al 2020

# Finalizar semana
/finalize-week

# Ver contexto completo
cat docs/SESSION_CONTEXT.md

# Ver progresso da semana
cat docs/weekly-reports/current-week.md
```

---

## 🎯 Agora Você Está Pronto!

**Sistema configurado e funcionando.**

**Próxima ação:** Configure Zotero (5 min) e rode `/start-sprint`

**Lembre-se:**
- 30% teoria / 60% prática / 10% revisão
- Sprints semanais
- Relatórios automáticos
- Proteção contra perda de dados

**Boa pesquisa! 🚀**
