# Agrupamento Profundo Semi-Supervisionado em Dados scRNA-seq de Câncer de Mama

## Descrição

Este projeto apresenta a implementação de um pipeline de **clustering semi-supervisionado** aplicado ao conjunto de dados **GSE75688**, composto por perfis de expressão gênica obtidos por **Single-Cell RNA Sequencing (scRNA-seq)** de células de câncer de mama.

O trabalho investiga se um pequeno conjunto de informações prévias, representadas por restrições **Must-Link** e **Cannot-Link**, pode auxiliar na organização do espaço latente e produzir agrupamentos mais coerentes do que abordagens totalmente não supervisionadas.

Todo o desenvolvimento foi realizado em um único notebook no **Google Colab**, reunindo desde o pré-processamento dos dados até a avaliação final dos modelos.

---

## Objetivo

O objetivo deste trabalho é comparar diferentes abordagens de clustering para dados de expressão gênica e analisar o impacto da utilização de restrições par-a-par durante o aprendizado das representações latentes.

Foram comparados métodos clássicos de agrupamento com um autoencoder semi-supervisionado, avaliando em quais situações a incorporação de conhecimento parcial melhora a qualidade dos clusters.

---

## Dataset

Foi utilizado o conjunto de dados **GSE75688**, contendo perfis de expressão gênica de células individuais de câncer de mama.

Após o pré-processamento foram mantidas apenas células do tipo **Single Cell (SC)** presentes simultaneamente na matriz de expressão e no metadata, resultando em **515 células**, distribuídas entre cinco tipos celulares:

* Tumor
* Stromal
* Myeloid
* Tcell
* Bcell

---

## Metodologia

O notebook está organizado em etapas sequenciais:

### 1. Preparação dos dados

* leitura da matriz de expressão gênica e do metadata;
* seleção das células do tipo SC;
* interseção entre matriz de expressão e metadata;
* remoção de genes pouco expressos;
* transformação `log1p`;
* seleção dos 1000 genes mais variáveis;
* padronização utilizando `StandardScaler`.

### 2. Geração das restrições

São construídos pares:

* **Must-Link**, indicando células que devem permanecer no mesmo grupo;
* **Cannot-Link**, indicando células que devem pertencer a grupos diferentes.

A geração dos pares utiliza **amostragem estratificada**, reduzindo o efeito do desbalanceamento entre os tipos celulares.

Foram avaliados três cenários de restrições:

| Cenário     | Must-Link | Cannot-Link |
| ----------- | --------: | ----------: |
| 25ML_25CL   |        25 |          25 |
| 50ML_75CL   |        50 |          75 |
| 100ML_150CL |       100 |         150 |

### 3. Baselines

Foram implementados três métodos de referência:

* PCA + KMeans;
* COP-KMeans;
* Autoencoder Não Supervisionado + KMeans.

### 4. Autoencoder Semi-Supervisionado

O modelo principal combina:

* perda de reconstrução;
* perda baseada nas restrições Must-Link e Cannot-Link.

Além disso, o treinamento utiliza:

* divisão treino/validação dos pares;
* early stopping;
* seleção automática do melhor modelo segundo a perda de validação.

### 5. Avaliação

Os modelos são avaliados utilizando as métricas:

* Adjusted Rand Index (ARI);
* Normalized Mutual Information (NMI);
* Homogeneity;
* V-Measure.

Também são apresentadas:

* matrizes de contingência;
* visualizações dos espaços latentes utilizando t-SNE;
* comparação entre diferentes quantidades de restrições.

---

## Como utilizar

Todo o projeto está concentrado em um único notebook Jupyter.

Para executar os experimentos:

1. Abra o notebook no **Google Colab**;
2. Faça o download do dataset **GSE75688** no repositório GEO (https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE75688). Em seguida, faça o upload dos arquivos para a pasta esperada pelo notebook ou ajuste os caminhos utilizados no código;
3. Execute as células na ordem em que aparecem.

O notebook realiza automaticamente todas as etapas do pipeline, desde o pré-processamento até a geração das métricas e visualizações finais.

---

## Principais Resultados

Os experimentos mostraram que:

* o **PCA + KMeans** fornece um baseline competitivo para o conjunto de dados;
* o **COP-KMeans** apresenta pequenas melhorias ao incorporar restrições par-a-par;
* o **Autoencoder Não Supervisionado** aprende boas representações para reconstrução, mas nem sempre produz o melhor espaço para clustering;
* o **Autoencoder Semi-Supervisionado** permite incorporar conhecimento parcial durante o aprendizado do espaço latente, possibilitando analisar o impacto das restrições na organização dos agrupamentos.

O foco do trabalho não é demonstrar que um único método supera todos os demais, mas investigar em quais condições as restrições par-a-par contribuem para melhorar o processo de clustering.

---

## Tecnologias Utilizadas

* Python
* Google Colab
* NumPy
* Pandas
* Scikit-learn
* PyTorch
* Matplotlib

---

## Autores

* Beatriz Rangel Cerutti
* Renato Alves de Sousa
