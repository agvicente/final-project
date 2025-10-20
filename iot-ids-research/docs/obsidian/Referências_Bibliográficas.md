# Referências Bibliográficas

> **Curadoria:** Outubro 2025  
> **Contexto:** Pesquisa de Mestrado em IDS para IoT  
> **Tema:** Avaliação de Modelos com Balanced Accuracy

---

## 📚 Organização

As referências estão organizadas por:
1. **Artigo Principal** - [[The_Balanced_Accuracy_and_Its_Posterior_Distribution]]
2. **Livros Fundamentais** - Estatística e Probabilidade
3. **Livros de Inferência Bayesiana**
4. **Livros de Machine Learning**
5. **Papers Importantes**
6. **Recursos Online**
7. **Livros Disponíveis Localmente**

---

## 🎯 Artigo Principal

### The Balanced Accuracy (2010)

**Brodersen, K.H., Ong, C.S., Stephan, K.E., & Buhmann, J.M.** (2010). The balanced accuracy and its posterior distribution. *20th International Conference on Pattern Recognition (ICPR)*, pp. 3121-3124. IEEE.

**BibTeX:**
```bibtex
@inproceedings{brodersen2010balanced,
  title={The balanced accuracy and its posterior distribution},
  author={Brodersen, Kay H and Ong, Cheng Soon and Stephan, Klaas E and Buhmann, Joachim M},
  booktitle={2010 20th International Conference on Pattern Recognition},
  pages={3121--3124},
  year={2010},
  organization={IEEE},
  doi={10.1109/ICPR.2010.764}
}
```

**Onde encontrar:**
- Arquivo local: `../papers/The_Balanced_Accuracy_and_Its_Posterior_Distribution.pdf`
- [IEEE Xplore](https://ieeexplore.ieee.org/)

**Por que é importante:**
- ⭐⭐⭐⭐⭐ Base teórica do seu projeto
- Resolve problemas de [[Intervalos_de_Confiança]] inadequados
- Propõe uso de [[Distribuição_Beta]] para [[Acurácia_Balanceada]]

---

## 📖 Livros Fundamentais

### Estatística Geral

#### All of Statistics (2004)
**Wasserman, Larry.** *All of Statistics: A Concise Course in Statistical Inference*. Springer, 2004.

**Capítulos relevantes:**
- Cap. 2-3: [[Média_Desvio_Padrão_Erro_Padrão]]
- Cap. 5: Bootstrap e métodos não-paramétricos
- Cap. 11: Inferência estatística

**Por que ler:**
- ✅ Conciso e direto
- ✅ Cobre tópicos essenciais rapidamente
- ✅ Bom para revisão

**Nível:** Intermediário

---

#### Mathematical Statistics and Data Analysis (2006)
**Rice, John A.** *Mathematical Statistics and Data Analysis* (3rd ed.). Duxbury Press, 2006.

**Capítulos relevantes:**
- Cap. 7: [[Média_Desvio_Padrão_Erro_Padrão|Estimação paramétrica]]
- Cap. 8: [[Intervalos_de_Confiança]]

**Por que ler:**
- ✅ Rigor matemático
- ✅ Muitos exemplos práticos
- ✅ Exercícios bem elaborados

**Nível:** Intermediário-Avançado

**Disponível:** `../books/MathematicalStatisticsandDataAnalysis3ed.pdf` ✅

---

#### Statistical Inference (2002)
**Casella, George & Berger, Roger L.** *Statistical Inference* (2nd ed.). Duxbury, 2002.

**Capítulos relevantes:**
- Cap. 3: [[Distribuições_de_Probabilidade]] comuns
- Cap. 5: Propriedades de estimadores
- Cap. 9: [[Intervalos_de_Confiança]]

**Por que ler:**
- ✅ Referência definitiva em inferência
- ✅ Provas matemáticas completas
- ✅ Muito usado em pós-graduação

**Nível:** Avançado

---

### Métodos Paramétricos e Não-Paramétricos

#### All of Nonparametric Statistics (2006)
**Wasserman, Larry.** *All of Nonparametric Statistics*. Springer, 2006.

**Capítulos relevantes:**
- Cap. 1: [[Métodos_Paramétricos_vs_Não_Paramétricos|Comparação de abordagens]]

**Por que ler:**
- ✅ Complemento ao "All of Statistics"
- ✅ Explica trade-offs

**Nível:** Avançado

---

#### Handbook of Parametric and Nonparametric Statistical Procedures (2011)
**Sheskin, David J.** *Handbook of Parametric and Nonparametric Statistical Procedures* (5th ed.). CRC Press, 2011.

**Por que ler:**
- ✅ Referência enciclopédica
- ✅ Testes estatísticos passo a passo
- ✅ Comparação direta de métodos

**Nível:** Todos os níveis (referência)

**Disponível:** `../books/Handbook_of_Parametric_and_Nonparametric_Statistical_Procedures_SecondEdition.pdf` ✅

---

## 🔵 Livros de Inferência Bayesiana

### Statistical Rethinking (2020) ⭐ COMECE AQUI!
**McElreath, Richard.** *Statistical Rethinking: A Bayesian Course with Examples in R and Stan* (2nd ed.). CRC Press, 2020.

**Capítulos relevantes:**
- Cap. 2: [[Inferência_Bayesiana]] introdução
- Cap. 3: Amostragem da posterior
- Cap. 6: Overfitting e regularização

**Por que ler:**
- ⭐⭐⭐⭐⭐ MAIS DIDÁTICO de todos!
- ✅ Exemplos práticos com código
- ✅ Visualizações excelentes
- ✅ Humor e analogias memoráveis
- ✅ Sem jargão desnecessário

**Nível:** Iniciante-Intermediário

**Disponível:** `../books/RM-StatRethink-Bayes.pdf` ✅

**Vídeo-aulas:** [YouTube Playlist](https://www.youtube.com/playlist?list=PLDcUM9US4XdMROZ57-OIRtIK0aOynbgZN)

---

### Bayesian Data Analysis (2013) ⭐ REFERÊNCIA DEFINITIVA
**Gelman, Andrew, Carlin, John B., Stern, Hal S., Dunson, David B., Vehtari, Aki, & Rubin, Donald B.** *Bayesian Data Analysis* (3rd ed.). CRC Press, 2013.

**Capítulos relevantes:**
- Cap. 2: [[Distribuição_Beta]] e conjugação
- Cap. 4: [[Intervalos_de_Confiança|Intervalos de credibilidade]]
- Cap. 5: Seleção de modelos

**Por que ler:**
- ⭐⭐⭐⭐⭐ Bíblia do Bayesiano
- ✅ Cobertura completa
- ✅ Exemplos reais
- ✅ Computação moderna

**Nível:** Intermediário-Avançado

**Disponível:** `../books/Bayesian_Data_Analysis.pdf` ✅

---

### Doing Bayesian Data Analysis (2014)
**Kruschke, John K.** *Doing Bayesian Data Analysis: A Tutorial with R, JAGS, and Stan* (2nd ed.). Academic Press, 2014.

**Capítulos relevantes:**
- Cap. 6: [[Distribuição_Beta|Inferência para probabilidades]]
- Cap. 12: Modelo Bayesiano hierárquico

**Por que ler:**
- ✅ Muito didático ("Puppies book")
- ✅ Foco em implementação
- ✅ MCMC explicado claramente

**Nível:** Iniciante-Intermediário

---

### Data Analysis: A Bayesian Tutorial (2006)
**Sivia, D.S. & Skilling, J.** *Data Analysis: A Bayesian Tutorial* (2nd ed.). Oxford University Press, 2006.

**Por que ler:**
- ✅ Curto e focado
- ✅ Ótimo para físicos/engenheiros
- ✅ Filosofia Bayesiana clara

**Nível:** Intermediário

---

## 🤖 Livros de Machine Learning

### Pattern Recognition and Machine Learning (2006) ⭐ CITADO NO ARTIGO!
**Bishop, Christopher M.** *Pattern Recognition and Machine Learning*. Springer, 2006.

**Capítulos relevantes:**
- Seção 1.5.4: Model selection
- **Seção 2.2 (pp. 68-74):** [[Distribuição_Beta]] e conjugação ⭐ **Citado no artigo!**
- Cap. 4: Linear models for classification

**Por que ler:**
- ⭐⭐⭐⭐⭐ Clássico de ML
- ✅ Perspectiva Bayesiana
- ✅ Matemática rigorosa mas acessível
- ✅ Figuras excelentes

**Nível:** Intermediário-Avançado

**Disponível:** `../books/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf` ✅

---

### The Elements of Statistical Learning (2009)
**Hastie, Trevor, Tibshirani, Robert, & Friedman, Jerome.** *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer, 2009.

**Capítulos relevantes:**
- Cap. 7: Model assessment and selection
- Cap. 18: High-dimensional problems

**Por que ler:**
- ⭐⭐⭐⭐⭐ Referência em ML estatístico
- ✅ Cobertura enciclopédica
- ✅ **Gratuito online!**

**Nível:** Avançado

**Download:** [PDF oficial gratuito](https://hastie.su.domains/ElemStatLearn/)

---

### Introduction to Machine Learning (2020)
**Alpaydin, Ethem.** *Introduction to Machine Learning* (4th ed.). MIT Press, 2020.

**Capítulos relevantes:**
- Cap. 19: Assessing and comparing classification algorithms

**Por que ler:**
- ✅ Introdução acessível
- ✅ Foco em conceitos
- ✅ Menos matemática pesada

**Nível:** Iniciante-Intermediário

---

### Machine Learning: The Art and Science of Algorithms (2012)
**Flach, Peter.** *Machine Learning: The Art and Science of Algorithms that Make Sense of Data*. Cambridge University Press, 2012.

**Capítulos relevantes:**
- Cap. 9: Evaluating classification

**Por que ler:**
- ✅ Ênfase em evaluation metrics
- ✅ Trade-offs bem explicados

**Nível:** Intermediário

---

## 📄 Papers Importantes

### Sobre Balanced Accuracy

**Velez et al. (2007)**
Velez, D.R., White, B.C., Motsinger, A.A., Bush, W.S., Ritchie, M.D., Williams, S.M., & Moore, J.H. "A balanced accuracy function for epistasis modeling in imbalanced datasets using multifactor dimensionality reduction". *Genetic Epidemiology*, 31(4), 306-315.

**BibTeX:**
```bibtex
@article{velez2007balanced,
  title={A balanced accuracy function for epistasis modeling in imbalanced datasets},
  author={Velez, Diane R and White, Bill C and Motsinger, Alison A and others},
  journal={Genetic Epidemiology},
  volume={31},
  number={4},
  pages={306--315},
  year={2007}
}
```

---

### Sobre Intervalos de Confiança para Proporções

**Agresti & Coull (1998)**
Agresti, A. & Coull, B.A. "Approximate is better than 'exact' for interval estimation of binomial proportions". *The American Statistician*, 52(2), 119-126.

**Por que importante:**
- Mostra que métodos "exatos" podem ser piores
- Propõe melhorias sobre IC tradicional

---

**Brown, Cai & DasGupta (2001)**
Brown, L.D., Cai, T.T., & DasGupta, A. "Interval estimation for a binomial proportion". *Statistical Science*, 16(2), 101-133.

**Por que importante:**
- Revisão abrangente de métodos
- Comparação empírica extensiva

---

### Sobre Classes Desbalanceadas

**Japkowicz & Stephen (2002)**
Japkowicz, N. & Stephen, S. "The class imbalance problem: A systematic study". *Intelligent Data Analysis*, 6(5), 429-449.

**BibTeX:**
```bibtex
@article{japkowicz2002class,
  title={The class imbalance problem: A systematic study},
  author={Japkowicz, Nathalie and Stephen, Shaju},
  journal={Intelligent Data Analysis},
  volume={6},
  number={5},
  pages={429--449},
  year={2002}
}
```

---

**Chawla et al. (2002) - SMOTE**
Chawla, N.V., Bowyer, K.W., Hall, L.O., & Kegelmeyer, W.P. "SMOTE: synthetic minority over-sampling technique". *Journal of Artificial Intelligence Research*, 16(3), 321-357.

---

### Sobre Evaluation Metrics

**Sokolova & Lapalme (2009)**
Sokolova, M. & Lapalme, G. "A systematic analysis of performance measures for classification tasks". *Information Processing & Management*, 45(4), 427-437.

**Por que importante:**
- Taxonomia de métricas
- Quando usar cada uma

---

## 🌐 Recursos Online

### Visualizações Interativas

**Seeing Theory** ⭐⭐⭐⭐⭐
URL: https://seeing-theory.brown.edu/

**Conteúdo:**
- [[Distribuições_de_Probabilidade]] interativas
- [[Inferência_Bayesiana]] visual
- [[Intervalos_de_Confiança]] animados

**Por que usar:**
- Melhor recurso visual disponível
- Gratuito
- Intuitivo

---

**Distribution Explorer**
URL: https://distribution-explorer.github.io/

**Conteúdo:**
- Explore [[Distribuições_de_Probabilidade]]
- Ajuste parâmetros e veja efeitos
- Compare distribuições

---

### Vídeos e Cursos

**3Blue1Brown** - Grant Sanderson
YouTube: https://www.youtube.com/c/3blue1brown

**Vídeos relevantes:**
- "Bayes theorem" - [[Inferência_Bayesiana]]
- "Binomial distributions" - [[Distribuições_de_Probabilidade#Binomial]]

**Por que assistir:**
- Visualizações INCRÍVEIS
- Intuição geométrica
- Muito didático

---

**StatQuest** - Josh Starmer
YouTube: https://www.youtube.com/c/joshstarmer

**Vídeos relevantes:**
- "Sensitivity and Specificity"
- "ROC and AUC"
- "Bayesian Statistics"

**Por que assistir:**
- Explicações simples e claras
- Bom para iniciantes
- Cobertura ampla

---

### Khan Academy
URL: https://www.khanacademy.org/math/statistics-probability

**Conteúdo:**
- [[Média_Desvio_Padrão_Erro_Padrão|Estatística descritiva]]
- [[Distribuições_de_Probabilidade]] básicas
- Testes de hipótese

**Por que usar:**
- Gratuito
- Exercícios interativos
- Bom para fundamentos

---

### Documentação de Bibliotecas

**Scipy.stats** ⭐
URL: https://docs.scipy.org/doc/scipy/reference/stats.html

**Conteúdo:**
- [[Distribuição_Beta|Beta distribution]]
- Todas as distribuições principais
- Exemplos de código

---

**Scikit-learn: Model Evaluation**
URL: https://scikit-learn.org/stable/modules/model_evaluation.html

**Conteúdo:**
- [[Acurácia]]
- [[Acurácia_Balanceada]] (`balanced_accuracy_score`)
- Confusion matrix

---

**PyMC** ⭐
URL: https://www.pymc.io/

**Conteúdo:**
- [[Inferência_Bayesiana]] probabilistic programming
- Tutoriais interativos
- Exemplos práticos

---

## 📚 Livros em Tópicos Específicos

### Classes Desbalanceadas

**He, H. & Ma, Y.** (eds.) (2013). *Imbalanced Learning: Foundations, Algorithms, and Applications*. Wiley-IEEE Press.

**Por que ler:**
- Cobertura completa do problema
- Múltiplas soluções
- Relevante para [[Aplicação_ao_IoT_IDS]]

---

### Probabilidade Avançada

**DeGroot, M.H. & Schervish, M.J.** (2012). *Probability and Statistics* (4th ed.). Pearson.

**Capítulos relevantes:**
- Cap. 3-5: [[Distribuições_de_Probabilidade]]
- Cap. 7: Estimação

---

**Ross, S.M.** (2014). *Introduction to Probability Models* (11th ed.). Academic Press.

**Por que ler:**
- Processos estocásticos
- Aplicações práticas

---

## 🎓 Para Citar na Dissertação

### Formato Básico

**ABNT:**
```
BRODERSEN, K. H. et al. The balanced accuracy and its posterior distribution. 
In: INTERNATIONAL CONFERENCE ON PATTERN RECOGNITION, 20., 2010. 
Proceedings... IEEE, 2010. p. 3121-3124.
```

**APA:**
```
Brodersen, K. H., Ong, C. S., Stephan, K. E., & Buhmann, J. M. (2010). 
The balanced accuracy and its posterior distribution. In 2010 20th 
International Conference on Pattern Recognition (pp. 3121-3124). IEEE.
```

---

## 📋 Checklist de Leitura Recomendada

### Ordem Sugerida para seu Projeto

**Semana 1-2: Fundamentos**
- [ ] Wasserman - *All of Statistics* (Caps. 2-3, 11)
- [ ] Bishop - *Pattern Recognition* (Seção 2.2) ⭐
- [ ] Brodersen et al. - Artigo principal ⭐⭐⭐

**Semana 3-4: Bayesiano**
- [ ] McElreath - *Statistical Rethinking* (Caps. 1-3) ⭐⭐⭐
- [ ] Gelman - *Bayesian Data Analysis* (Cap. 2-4)

**Semana 5-6: Aplicação**
- [ ] Japkowicz & Stephen - Class imbalance paper
- [ ] Flach - *ML: Art and Science* (Cap. 9)
- [ ] He & Ma - *Imbalanced Learning* (Cap. relevantes)

**Contínuo: Recursos Online**
- [ ] Seeing Theory - visualizações
- [ ] 3Blue1Brown - vídeos sobre Bayes
- [ ] Scipy/Scikit-learn docs

---

## 🔗 Links para Conceitos

- [[Acurácia]]
- [[Acurácia_Balanceada]]
- [[Distribuição_Beta]]
- [[Distribuições_de_Probabilidade]]
- [[Inferência_Bayesiana]]
- [[Intervalos_de_Confiança]]
- [[Média_Desvio_Padrão_Erro_Padrão]]
- [[Métodos_Paramétricos_vs_Não_Paramétricos]]
- [[The_Balanced_Accuracy_and_Its_Posterior_Distribution]]
- [[Aplicação_ao_IoT_IDS]]

---

**Tags:** #references #bibliography #books #papers #resources #reading-list

**Voltar para:** [[INDEX]]

