# Early Stopping com Validação Dinâmica em Blocos

Enquanto o early stopping tradicional depende de uma partição fixa de validação e fica limitado a uma única divisão dos dados, a validação dinâmica reamostra os conjuntos a cada época para aproveitar melhor o dataset. Contudo, essa reamostragem frequente gera muito ruído na curva de loss, tornando o critério de parada instável. Como alternativa intermediária para mitigar esse problema, este estudo investiga a Validação Dinâmica em Blocos, técnica que congela as partições por um período de $X$ épocas antes de realizar uma nova reamostragem, equilibrando estabilidade e aproveitamento de dados.

---

## Como executar

Para reproduzir o experimento basta instalar as dependências listadas em `requirements.txt`, então executar todas as células do jupyter notebook `early_stopping_vdb.ipynb` em ordem.

Caso queira apenas analisar as métricas e visualizar as curvas sem passar pelo processo demorado de treinamento, as células da seção **"Avaliação dos resultados"** podem ser executadas de forma independente. O notebook está configurado para carregar os históricos e métricas pré-calculados diretamente dos arquivos `.json` e `.csv` inclusos no repositório.

### Estrutura do Projeto

```text
tema01/
│
├── graphs/                    Gráficos gerados
├── early_stopping_vdb.ipynb   Código fonte do experimento
│
├── loss_hist_cla.json         Histórico perda (classificação)
│
├── loss_hist_reg.json         Histórico perda (regressão)
│
├── results_cla.csv            Desempenho teste dos modelos (classificação)
│
└── results_reg.csv            Desempenho teste dos modelos (regressão)
|
└── requirements.txt           Bibliotecas necessárias para execução
```

---

## Metodologia Experimental

### Conjunto de dados

* Regressão: [dataset Student](https://doi.org/10.24432/C5TG7T) com variável-alvo "G3"
* Classificação: [dataset Titanic](https://github.com/datasciencedojo/datasets/blob/master/titanic.csv)  com variável-alvo "Class"

### Arquitetura da rede

Para ambos os problemas, utilizou-se uma rede neural artificial do tipo *Multi-Layer Perceptron* (MLP) da biblioteca daltoolbox, modificada para incluir a separação dinâmica entre treino e validação em blocos por meio da introdução do parâmetro **B**.

### Hiperparâmetros gerais:

| Hiperparâmetro | Valor |
| :--- | :--- |
| Hidden Sizes | (64, 32) |
| Dropout | 0,1 |
| Proporção de Validação | 0,2 |
| Batch size | 64 |
| Learning rate | 0,001 |
| Activation | Relu |
| Min Delta | $1 \times 10^{-4}$ |
| Epochs | 10.000 |

Estratégias Early Stopping implementadas:

1. **Paciência (*patience*)**: Treinamento interrompido caso a perda de validação não apresente melhora por um período consecutivo de 100 épocas.
2. **Teste de Hipótese ($ h $)**: Aplica-se um teste $t$ de Student para avaliar a estagnação do aprendizado, configurado com uma janela (*test_window*) de 60 épocas e um nível de significância ($p$-value) de 0,1.

Para cada estratégia de early stopping foi treinado um modelo *static*, o qual não sofre reamostragem de dados para comparação. Assim foi aplicada a estratégia de validação em blocos testando os valores **[1,5,10,20,30,50]** para reamostragem. Assim, cada variação de modelo foi testada em um conjunto de 30 seeds aleatórias.

---

## Resultados obtidos

### Nomenclatura dos modelos

Regras early stopping:

* h - teste de hipótese
* patience - paciência

Tipo validação:

* static - treino e validação fixos
* dynamic B - treino e validação reamostrados a cada B épocas

### Métricas nos modelos de Regressão

| Modelo | RMSE | $R^2$ | Épocas | ValLoss std |
| --- | --- | --- | --- | --- |
| h dynamic 01 | $2,566 \pm 0,11$ | $0,678 \pm 0,03$ | $1419,9 \pm 276,7$ | $3,24 \pm 0,34$ |
| h dynamic 05 | $2,574 \pm 0,12$ | $0,676 \pm 0,03$ | $1394,6 \pm 347,9$ | $3,27 \pm 0,41$ |
| **h dynamic 10** | **$2,545 \pm 0,11$** | **$0,684 \pm 0,03$** | **$1408,8 \pm 364,7$** | **$3,33 \pm 0,42$** |
| h dynamic 20 | $2,584 \pm 0,14$ | $0,673 \pm 0,03$ | $1513,9 \pm 419,4$ | $3,35 \pm 0,44$ |
| h dynamic 30 | $2,585 \pm 0,11$ | $0,673 \pm 0,03$ | $1385,6 \pm 490,6$ | $3,60 \pm 0,64$ |
| h dynamic 50 | $2,572 \pm 0,10$ | $0,677 \pm 0,03$ | $1282,0 \pm 362,5$ | $3,75 \pm 0,68$ |
| h static | $2,608 \pm 0,12$ | $0,667 \pm 0,03$ | $777,1 \pm 456,5$ | $3,55 \pm 0,90$ |
|  |  |  |  |  |
| patience dynamic 01 | $2,644 \pm 0,10$ | $0,659 \pm 0,02$ | $715,7 \pm 169,0$ | $4,30 \pm 0,46$ |
| patience dynamic 05 | $2,638 \pm 0,11$ | $0,660 \pm 0,03$ | $795,0 \pm 216,1$ | $4,11 \pm 0,45$ |
| patience dynamic 10 | $2,634 \pm 0,10$ | $0,661 \pm 0,03$ | $676,2 \pm 192,3$ | $4,45 \pm 0,57$ |
| patience dynamic 20 | $2,657 \pm 0,09$ | $0,655 \pm 0,02$ | $670,9 \pm 190,0$ | $4,59 \pm 0,59$ |
| patience dynamic 30 | $2,652 \pm 0,11$ | $0,656 \pm 0,03$ | $606,3 \pm 181,0$ | $4,85 \pm 0,76$ |
| patience dynamic 50 | $2,665 \pm 0,10$ | $0,653 \pm 0,03$ | $647,0 \pm 203,9$ | $4,80 \pm 0,87$ |
| patience static | $2,600 \pm 0,13$ | $0,670 \pm 0,03$ | $446,9 \pm 177,2$ | $4,20 \pm 0,98$ |


### Métricas nos modelos de Classificação

| Modelo | Acc | Prec | Rec | F1 | Épocas | ValLoss std |
| --- | --- | --- | --- | --- | --- | --- |
| h dynamic 01 | $0,681 \pm 0,01$ | $0,670 \pm 0,01$ | $0,632 \pm 0,01$ | $0,632 \pm 0,01$ | $594,7 \pm 262,0$ | $0,048 \pm 0,002$ |
| h dynamic 05 | $0,681 \pm 0,01$ | $0,670 \pm 0,01$ | $0,631 \pm 0,01$ | $0,631 \pm 0,01$ | $915,1 \pm 509,6$ | $0,050 \pm 0,004$ |
| h dynamic 10 | $0,685 \pm 0,01$ | $0,675 \pm 0,01$ | $0,636 \pm 0,01$ | $0,637 \pm 0,01$ | $972,5 \pm 614,2$ | $0,050 \pm 0,004$ |
| h dynamic 20 | $0,687 \pm 0,01$ | $0,678 \pm 0,01$ | $0,639 \pm 0,01$ | $0,640 \pm 0,01$ | $577,5 \pm 414,4$ | $0,047 \pm 0,006$ |
| h dynamic 30 | $0,686 \pm 0,01$ | $0,677 \pm 0,01$ | $0,637 \pm 0,01$ | $0,637 \pm 0,02$ | $582,8 \pm 302,2$ | $0,047 \pm 0,008$ |
| h dynamic 50 | $0,686 \pm 0,01$ | $0,676 \pm 0,01$ | $0,638 \pm 0,01$ | $0,640 \pm 0,01$ | $690,0 \pm 542,1$ | $0,047 \pm 0,011$ |
| h static | $0,688 \pm 0,01$ | $0,677 \pm 0,02$ | $0,642 \pm 0,02$ | $0,644 \pm 0,02$ | $253,0 \pm 108,7$ | $0,018 \pm 0,005$ |
|  |  |  |  |  |  |  |
| patience dynamic 01 | $0,682 \pm 0,01$ | $0,670 \pm 0,01$ | $0,636 \pm 0,02$ | $0,637 \pm 0,02$ | $295,6 \pm 116,7$ | $0,047 \pm 0,002$ |
| patience dynamic 05 | $0,687 \pm 0,01$ | $0,677 \pm 0,01$ | $0,640 \pm 0,01$ | $0,641 \pm 0,02$ | $261,5 \pm 92,8$ | $0,046 \pm 0,004$ |
| patience dynamic 10 | $0,687 \pm 0,01$ | $0,677 \pm 0,01$ | $0,640 \pm 0,01$ | $0,641 \pm 0,02$ | $229,5 \pm 65,5$ | $0,046 \pm 0,005$ |
| patience dynamic 20 | $0,687 \pm 0,01$ | $0,676 \pm 0,01$ | $0,640 \pm 0,02$ | $0,641 \pm 0,02$ | $229,9 \pm 80,7$ | $0,045 \pm 0,008$ |
| patience dynamic 30 | $0,690 \pm 0,01$ | $0,681 \pm 0,02$ | $0,643 \pm 0,02$ | $0,645 \pm 0,02$ | $235,7 \pm 102,0$ | $0,046 \pm 0,010$ |
| patience dynamic 50 | $0,686 \pm 0,01$ | $0,676 \pm 0,01$ | $0,639 \pm 0,01$ | $0,640 \pm 0,02$ | $254,1 \pm 76,4$ | $0,048 \pm 0,013$ |
| **patience static** | **$0,694 \pm 0,02$** | **$0,682 \pm 0,02$** | **$0,651 \pm 0,02$** | **$0,654 \pm 0,02$** | **$205,2 \pm 103,6$** | **$0,020 \pm 0,006$** |

---

## Conclusão

A validação dinâmica em blocos demonstrou ser uma alternativa promissora, superando ligeiramente a abordagem estática no cenário de regressão. Em contra partida, perdeu para a abordagem estática nos testes de classificação, mesmo que também por uma pequena margem. Estes resultados indicam que, quando a validação dinâmica é viável, a reamostragem por blocos é uma estratégia recomendável para o ajuste fino do modelo. Contudo, a técnica não se mostrou eficaz na redução do ruído da curva de validação, uma vez que a volatilidade e a amplitude dos picos permaneceram semelhantes.
