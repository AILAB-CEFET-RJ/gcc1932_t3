# Modelos e model selection

Este README documenta o pre-processamento, os modelos baseline e a selecao de
modelos para estimacao de redshift fotometrico.

## Estrutura

```text
src/redshift/
  data/
    preprocess.py
    preprocess_gate_err.py
    gate_err.py
  evaluation/
    plots.py
    slices.py
  models/
    linear_regression.py
    polynomial_ridge.py
  training/
    search_polynomial_ridge.py
    search_gate_strength_linear_regression.py
  utils/
    modeling.py
```

## Visao geral

Tres baselines de features sao comparados nos dois datasets COIN:

- **Datasets**: `happyT` (COIN/Happy) e `teddyT` (COIN/Teddy).
- **Modelos**: Regressao Linear e Polynomial Ridge (`PolynomialFeatures` + `Ridge`).
- **Conjuntos de features**:
  - `mag`: apenas magnitudes;
  - `mag_err`: magnitudes + erros fotometricos;
  - `gate_err_manual`: cinco magnitudes ponderadas pelos respectivos erros.

Fluxo de trabalho:

1. Pre-processar MAG/MAG_ERR com `data/preprocess.py` ou GATE-ERR com
   `data/preprocess_gate_err.py`.
2. Selecionar hiperparametros na **validacao**: o Polynomial Ridge busca
   `degree`/`alpha`; o GATE-ERR ainda busca `gate_strength` (na Regressao Linear ha
   um script dedicado para isso).
3. Avaliar no teste externo, separadamente em `B`, `C` e `D`.

`utils/modeling.py` concentra o codigo comum: leitura de `data/processed/` para
MAG/MAG_ERR ou de `data/processed Gate-Err/` para GATE-ERR, selecao de features e
split de avaliacao, metricas na escala original do redshift, e os caminhos
padronizados de modelos, metricas, figuras e tabelas.

`evaluation/plots.py` salva o grafico de avaliacao (real vs previsto e residuos).
A geracao da figura e **nao-fatal**: se o backend do matplotlib estiver bloqueado
pelo sistema operacional, a execucao apenas avisa e continua, preservando metricas,
modelos e tabelas.

`evaluation/slices.py` calcula as metricas por faixa (redshift, magnitude `r` e
erro fotometrico medio), usadas apenas na avaliacao final (`--eval-split test`).

## Pre-processamento (`data/preprocess.py`)

**Os dois modelos consomem os splits ja preprocessados em subpastas por modo:**

```text
data/processed/Val/<dataset>/   -> model selection
data/processed/Test/<dataset>/  -> avaliacao final
```

| Grupo de colunas      | Transformacao                  |
| --------------------- | ------------------------------ |
| magnitudes `u,g,r,i,z`| `StandardScaler`               |
| erros `uErr..zErr`    | `log1p` e depois `RobustScaler`|
| alvo `redshift`       | escala original (sem scaler)   |

- O `log1p` nos **erros** reduz a assimetria das medidas.
- O **alvo** permanece diretamente na escala original de redshift; nao ha `log1p`
  no alvo e nao ha `expm1` nas predicoes.
- O `log1p` e aplicado nas **features brutas** de erro, **antes** de qualquer
  expansao polinomial. As metricas sao calculadas diretamente na escala original.

Split dos dados:

- conjunto `A` -> `train`/`validation` (80%/20%, estratificado por faixa de redshift);
- conjuntos `B`, `C`, `D` -> testes externos separados.

Gerar os dados processados (uma vez por dataset):

```powershell
.\.venv\Scripts\python.exe src\redshift\data\preprocess.py --dataset happyT
.\.venv\Scripts\python.exe src\redshift\data\preprocess.py --dataset teddyT
```

Por padrao, cada comando gera os dois modos:

```text
Val:
  fit dos scalers em: train
  salva: X_train, y_train, X_val, y_val, preprocessors.joblib

Test:
  fit dos scalers em: train + validation
  salva: X_train, y_train, X_test_B/C/D, y_test_B/C/D, preprocessors.joblib
```

No modo `Test`, o arquivo `X_train.csv` representa `train + validation` ja
preprocessado com scalers ajustados nesse conjunto combinado. Os arquivos
`X_test_B.csv`, `X_test_C.csv` e `X_test_D.csv` mantem os tres testes externos
separados para avaliacao individual.

Gerar apenas um modo, se necessario:

```powershell
.\.venv\Scripts\python.exe src\redshift\data\preprocess.py --dataset happyT --mode val
.\.venv\Scripts\python.exe src\redshift\data\preprocess.py --dataset happyT --mode test
```

Esse script tradicional altera somente `data/processed/` e continua sendo o
preprocessing reproduzivel de `mag` e `mag_err`. Ele nao gera nem atualiza dados
do GATE-ERR.

## Baseline GATE-ERR manual

O `gate_err_manual` e um terceiro conjunto de features. Diferentemente de
`mag_err`, os erros fotometricos nao sao concatenados as magnitudes e nao chegam
diretamente ao estimador. Cada erro e usado somente para reduzir a contribuicao
da magnitude da mesma banda.

Para uma banda fotometrica `j`:

```text
weight_j = exp(-gate_strength * error_norm_j)
j_gate   = j_scaled * weight_j
```

As cinco features finais recebidas pelo modelo sao:

```text
u_gate, g_gate, r_gate, i_gate, z_gate
```

Como `error_norm` pertence ao intervalo `[0, 1]`, os pesos pertencem a
`[exp(-gate_strength), 1]`. Uma banda com erro normalizado proximo de zero preserva quase
toda a magnitude escalada. Uma banda com erro alto recebe peso menor. Os erros
`uErr..zErr` nunca entram como features finais.

O `gate_strength` controla a intensidade do gate:

- `gate_strength = 0`: todos os pesos valem 1, portanto o baseline equivale as magnitudes
  escaladas;
- `gate_strength > 0`: bandas mais incertas sao atenuadas;
- quanto maior o `gate_strength`, maior a atenuacao provocada pelo erro.

### Pre-processamento intermediario

O preprocessing do GATE-ERR e isolado do preprocessing de `mag` e `mag_err`. Ele
possui um script e uma raiz de dados proprios:

```text
script: src/redshift/data/preprocess_gate_err.py
raiz:   data/processed Gate-Err/
```

Gerar `Val` e `Test` sem alterar nada em `data/processed/`:

```powershell
.\.venv\Scripts\python.exe src\redshift\data\preprocess_gate_err.py --dataset happyT
.\.venv\Scripts\python.exe src\redshift\data\preprocess_gate_err.py --dataset teddyT
```

Estrutura produzida:

```text
data/processed Gate-Err/
  Val/<dataset>/
    X_train.csv
    y_train.csv
    X_val.csv
    y_val.csv
    preprocessors.joblib
  Test/<dataset>/
    X_train.csv
    y_train.csv
    X_test_B.csv
    y_test_B.csv
    X_test_C.csv
    y_test_C.csv
    X_test_D.csv
    y_test_D.csv
    preprocessors.joblib
```

No modo `Val`, o conjunto A e dividido de forma reproduzivel com 80% para treino,
20% para validacao, estratificacao por redshift e `random_state=42`. Todos os
transformadores sao ajustados somente no treino.

No modo `Test`, o ajuste usa o conjunto A completo (`train + validation`) e B, C
e D recebem somente transformacoes. O script nunca ajusta transformadores nos
testes externos.

Em cada conjunto usado para ajustar o preprocessing:

1. cada magnitude recebe `StandardScaler`;
2. cada erro fotometrico recebe `log1p`;
3. calcula-se o percentil 99 de cada coluna de erro transformada;
4. valores acima do p99 da respectiva banda sao saturados, sem remover objetos;
5. um `MinMaxScaler` e ajustado nos erros transformados e clipados;
6. o resultado do scaler e limitado ao intervalo `[0, 1]`.

O resultado e uma base intermediaria independente de `gate_strength`, com dez colunas:

```text
u_scaled, g_scaled, r_scaled, i_scaled, z_scaled
uErr_norm, gErr_norm, rErr_norm, iErr_norm, zErr_norm
```

Os cinco primeiros valores sao as magnitudes padronizadas. Os cinco ultimos sao
os erros ja transformados e normalizados. Nao existe `RobustScaler` nesse
pipeline, e as features finais `u_gate..z_gate` ainda nao existem nos CSVs.

Somente depois que um `gate_strength` e informado, as cinco features finais sao
construidas a partir da base intermediaria:

```text
j_gate = j_scaled * exp(-gate_strength * jErr_norm)
```

Assim, a mesma base processada pode ser reutilizada para comparar diferentes
valores de `gate_strength`, sem refazer `log1p`, clipping, escalas ou divisao dos dados.

O arquivo `preprocessors.joblib` registra os transformadores e parametros que
permitem reproduzir a base:

```text
magnitude_scaler
error_gate_minmax_scaler
error_gate_p99
gate_strength = None
gate_feature_columns
magnitude_columns
magnitude_error_columns
magnitude_scaled_columns
error_normalized_columns
```

O `gate_strength` permanece como `None` porque nao faz parte do preprocessing. Ele sera
definido posteriormente para construir as features finais.

### Controle de vazamento

No modo `Val`:

```text
fit dos transformadores: train
transform: validation
```

No modo `Test`:

```text
fit dos transformadores: conjunto A completo (train + validation)
transform: testes externos B, C e D
```

Nenhum p99 ou scaler e ajustado usando validacao no modo `Val`, nem usando B, C
ou D no modo `Test`.

Quando `--feature-set gate_err_manual` e usado, os scripts de modelo e busca
selecionam automaticamente `data/processed Gate-Err/`. Para `mag` e `mag_err`,
continuam usando `data/processed/`.

## Feature sets

`mag` usa apenas magnitudes:

```text
u, g, r, i, z
```

`mag_err` usa magnitudes e erros fotometricos:

```text
u, g, r, i, z, uErr, gErr, rErr, iErr, zErr
```

`gate_err_manual` usa as cinco magnitudes ponderadas descritas na secao
**Baseline GATE-ERR manual**.

## Splits de avaliacao

`--eval-split val` deve ser usado para model selection.

```text
dados lidos de: data/processed/Val/<dataset>/
fit dos scalers em: train
treina em: train
avalia em: validation
```

`--eval-split test` deve ser usado apenas para avaliacao final.

```text
dados lidos de: data/processed/Test/<dataset>/
fit dos scalers em: train + validation
treina em: train + validation
avalia em: um teste externo por vez (B, C ou D)
```

Em outras palavras:

- `val`: o modelo e ajustado apenas em `X_train` e comparado em `X_val`;
- `test`: depois da selecao do modelo, o ajuste final usa o `X_train` salvo em
  `data/processed/Test/<dataset>/`, que ja representa `train + validation`, e a
  avaliacao acontece em `X_test_B`, `X_test_C` ou `X_test_D`, escolhido com
  `--test-set`. Tambem e possivel usar `--test-set all` para treinar uma vez e
  avaliar os tres testes externos na mesma execucao.

Se o split escolhido estiver vazio ou incompleto, a execucao falha com erro explicito.

## Metricas

As metricas sao calculadas sempre na **escala original do redshift**. Sao iguais
para os dois modelos, pois ambos usam a mesma funcao `evaluate_predictions`.

```text
mae
rmse
r2
bias
n
negative_redshift_predictions
catastrophic_outlier_fraction
catastrophic_outlier_threshold
nmad
nmad_se
```

Erro normalizado de redshift:

```text
dz = (z_pred - z_real) / (1 + z_real)
```

- **Fracao de outliers catastroficos**: proporcao de objetos com `|dz| > 0.15`.
- **NMAD** (dispersao robusta de photo-z): `1.4826 * mediana(|dz - mediana(dz)|)`.
  O fator `1.4826` faz o NMAD coincidir com o desvio-padrao sob residuos Gaussianos,
  mas sem ser afetado pelas caudas pesadas / outliers catastroficos.
- **`nmad_se`**: erro-padrao do NMAD estimado por **bootstrap** (reamostragem com
  reposicao, seed fixa), pois nao tem formula fechada simples sob a distribuicao de
  cauda pesada de `dz`. E usado na regra de 1 desvio-padrao da busca.

Observacao: quando `rmse` fica muito maior que `nmad`, isso normalmente indica
**poucos pontos catastroficos** (o RMSE e dominado por eles; o NMAD, baseado na
mediana, ignora a cauda).

## Metricas por faixa (apenas na avaliacao final)

Quando `--eval-split test`, alem das metricas globais cada JSON em
`reports/metrics/Tests/` recebe um bloco `slice_metrics` com o desempenho **dentro
de subconjuntos** dos objetos. O calculo esta em `evaluation/slices.py` e so roda
no teste, nunca na validacao.

Sao tres recortes:

```text
by_redshift            -> faixas fixas de redshift verdadeiro
by_r_magnitude         -> quartis da magnitude r (escala original)
by_photometric_error   -> quartis do erro fotometrico medio das 5 bandas
```

- `by_redshift` usa bordas fixas `[0, 0.1, 0.2, 0.4, 0.6, inf)` (intervalos
  fechados a esquerda) sobre o **redshift verdadeiro**.
- `by_r_magnitude` e `by_photometric_error` usam **quartis** (4 faixas de tamanho
  parecido) das respectivas variaveis.
- Para criar faixas interpretaveis, as features escaladas sao revertidas para a
  escala original com o `preprocessors.joblib` do dataset (StandardScaler e
  RobustScaler/MinMax invertidos, depois `expm1` para desfazer o `log1p` dos erros).
  Vale tambem para o GATE-ERR, cuja base intermediaria guarda magnitudes escaladas
  e erros normalizados.

Em cada faixa sao calculadas **apenas** estas metricas:

```text
mae, rmse, bias, negative_redshift_predictions, nmad
```

Nao ha `nmad_se`, `r2` nem fracao de outliers por faixa. O NMAD de cada faixa
recentraliza na **mediana daquela faixa**, entao mede so a dispersao interna e nao
enxerga o vies entre faixas (que aparece na coluna `bias`). Faixas com poucos
objetos (ex.: `z 0..0.1`) tem NMAD estatisticamente fragil, ainda mais por nao
terem erro-padrao estimado.

## Organizacao de metricas, modelos, figuras e tabelas

```text
reports/metrics/<Model Selection ou Tests>/<dataset>/<feature_set>/<modelo>/
models/<Model Selection ou Tests>/<dataset>/<feature_set>/<modelo>/
reports/figures/<Model Selection ou Tests>/<dataset>/<feature_set>/<modelo>/
reports/tables/<Model Selection ou Tests>/<dataset>/<feature_set>/<modelo>/
```

Mapeamento de nomes de dataset nas pastas:

```text
happyT -> happy
teddyT -> teddy
```

Exemplos:

```text
reports/metrics/Model Selection/happy/mag/linear_regression/
reports/metrics/Tests/teddy/mag_err/polynomial_ridge/
reports/tables/Model Selection/happy/mag/polynomial_ridge/
```

## Regressao Linear

Script:

```text
src/redshift/models/linear_regression.py
```

Pipeline:

```text
data/processed/Val/<dataset> ou data/processed/Test/<dataset> -> LinearRegression
```

Le diretamente os arquivos do modo correspondente ao `--eval-split`: `Val` para
validacao e `Test` para avaliacao final. As magnitudes ja chegam com
`StandardScaler`, os erros com `log1p` + `RobustScaler`, e o alvo em escala
original. Nao ha hiperparametro a buscar.

Rodar MAG em validacao:

```powershell
.\.venv\Scripts\python.exe src\redshift\models\linear_regression.py --dataset happyT --feature-set mag --eval-split val
```

Rodar MAG+ERR em validacao:

```powershell
.\.venv\Scripts\python.exe src\redshift\models\linear_regression.py --dataset happyT --feature-set mag_err --eval-split val
```

Rodar GATE-ERR em validacao (exige `--gate-strength`; le de `data/processed Gate-Err/`,
que precisa ter sido gerado antes com `preprocess_gate_err.py`):

```powershell
.\.venv\Scripts\python.exe src\redshift\models\linear_regression.py --dataset happyT --feature-set gate_err_manual --eval-split val --gate-strength 0.1
```

Rodar avaliacao final em teste:

```powershell
.\.venv\Scripts\python.exe src\redshift\models\linear_regression.py --dataset happyT --feature-set mag --eval-split test --test-set B
```

Rodar os tres testes externos com um unico treino:

```powershell
.\.venv\Scripts\python.exe src\redshift\models\linear_regression.py --dataset happyT --feature-set mag --eval-split test --test-set all
```

Saidas por execucao: modelo `.joblib`, metricas `.json` e grafico `.png`.

## Polynomial Ridge

Script:

```text
src/redshift/models/polynomial_ridge.py
```

Assim como a regressao linear, o Polynomial Ridge consome a pasta processada do
modo correspondente ao `--eval-split` (nao usa mais dados brutos nem aplica
`log1p` ao alvo ou depois da expansao). O pipeline do modelo e:

```text
data/processed/Val/<dataset> ou data/processed/Test/<dataset>
  -> PolynomialFeatures -> StandardScaler -> Ridge
```

```python
PolynomialFeatures(degree=degree, include_bias=False)
StandardScaler()
Ridge(alpha=alpha)
```

### Escala em duas etapas

As features passam por `StandardScaler` **duas vezes**, em momentos diferentes e com
propositos diferentes:

1. **Antes do poly** (no pre-processamento, `data/processed`): magnitudes recebem
   `StandardScaler` e erros `log1p` + `RobustScaler`. Isso deixa as features ~O(1),
   entao termos como `u^2` ficam ~O(1) em vez de ~O(centenas) — o que reduz muito a
   tendencia de explosao na extrapolacao.
2. **Depois do poly** (no pipeline do modelo): o `StandardScaler` padroniza os termos
   **novos** (`u^2`, `u*g`, ...), para que a penalidade L2 do `Ridge` seja uniforme
   entre eles.

As duas passagens nao conflitam: re-padronizar os termos lineares (ja escalados na
etapa 1) e praticamente um no-op; o ganho real da etapa 2 e sobre os quadrados e
produtos. O alvo permanece em escala original no pre-processamento.

Rodar uma configuracao especifica (MAG):

```powershell
.\.venv\Scripts\python.exe src\redshift\models\polynomial_ridge.py --dataset happyT --feature-set mag --eval-split val --degree 2 --alpha 1
```

Com MAG+ERR:

```powershell
.\.venv\Scripts\python.exe src\redshift\models\polynomial_ridge.py --dataset happyT --feature-set mag_err --eval-split val --degree 2 --alpha 1
```

Com GATE-ERR (exige `--gate-strength`; le de `data/processed Gate-Err/`):

```powershell
.\.venv\Scripts\python.exe src\redshift\models\polynomial_ridge.py --dataset happyT --feature-set gate_err_manual --eval-split val --degree 2 --alpha 1 --gate-strength 0.1
```

Avaliacao final em teste, ja com o `degree` e o `alpha` escolhidos na validacao:

```powershell
.\.venv\Scripts\python.exe src\redshift\models\polynomial_ridge.py --dataset happyT --feature-set mag --eval-split test --test-set B --degree 2 --alpha 5
```

Rodar os tres testes externos com um unico treino:

```powershell
.\.venv\Scripts\python.exe src\redshift\models\polynomial_ridge.py --dataset happyT --feature-set mag --eval-split test --test-set all --degree 2 --alpha 5
```

## Busca de hiperparametros do Polynomial Ridge

Script:

```text
src/redshift/training/search_polynomial_ridge.py
```

Grade padrao:

```python
degrees = [1, 2]
alphas = [0.1, 0.5, 1, 2, 5, 10, 50, 100, 1000]
gate_strengths = [0.1, 0.5, 1.0, 5.0]   # apenas para o feature set gate_err_manual
```

Para `mag` e `mag_err`, a busca varre apenas `degree` x `alpha`. Para
`gate_err_manual`, ela varre `degree` x `alpha` x `gate_strength`, com a grade de
intensidades acima (ajustavel por `--gate-strengths`).

A busca treina cada combinacao em `train`, avalia em `validation` e salva uma tabela
resumo. A escolha do melhor conjunto de hiperparametros usa a **regra de 1 desvio-padrao**
(em ingles, *one-standard-error rule*) aplicada sobre o **NMAD** — nao o simples menor erro.

### Por que nao pegar simplesmente o menor erro

A metrica de validacao e uma **estimativa ruidosa**: se a amostra de validacao fosse um
pouco diferente, o valor mudaria. Pegar cegamente a config de menor erro quase sempre
premia a mais complexa (maior `degree`, menor `alpha`) por uma diferenca que cabe **dentro
do ruido**, sem ganho real de generalizacao. Pior: modelos pouco regularizados extrapolam
mal no teste externo (`B, C, D`). Queremos o oposto: o modelo **mais simples que ainda seja
estatisticamente equivalente ao melhor**.

### O que e o "1 desvio-padrao" aqui

Para saber se duas configs sao "iguais dentro do ruido", precisamos medir o quanto a
metrica oscilaria por puro acaso — o **desvio-padrao da estimativa** do NMAD (em ingles,
*standard error*, SE). Como a distribuicao do erro normalizado `dz` tem cauda pesada e o
NMAD nao tem formula fechada simples, esse desvio-padrao e estimado por **bootstrap**:
reamostramos os residuos com reposicao varias vezes, recalculamos o NMAD em cada
reamostra, e o desvio-padrao desses valores e o `nmad_se`. Ele responde: "se a amostra de
validacao fosse outra, quanto o NMAD do melhor modelo balancaria?".

### Passo a passo da regra

1. acha a config de **menor NMAD** (o "melhor"; coluna `nmad`);
2. pega o **desvio-padrao** do NMAD dessa config (`nmad_se`, via bootstrap);
3. define a faixa de tolerancia: `limite = melhor_NMAD + 1 desvio-padrao`
   (coluna `selection_threshold`). Toda config com `nmad <= limite` recebe
   `within_1se = True` — sao **indistinguiveis** do melhor, pois a diferenca cabe dentro
   do desvio-padrao;
4. dentre as indistinguiveis, escolhe a **mais simples / mais regularizada**: menor
   `degree` e, em seguida, maior `alpha` (coluna `selected_1se = True`).

Ou seja, abrimos mao de um ganho de erro que e so ruido em troca de **robustez**: o modelo
escolhido tem NMAD de validacao praticamente igual ao do minimo, porem mais regularizado,
o que generaliza melhor sob mudanca de distribuicao.

### Exemplo concreto (teddy / mag_err)

Com os numeros reais da tabela:

```text
melhor NMAD               = 0.018033   (degree 2, alpha 0.1)
desvio-padrao (bootstrap) = 0.000198
limite 1-SE               = 0.018033 + 0.000198 = 0.018231
```

5 configs ficam dentro do limite (todas `degree 2`, com `alpha` de 0.1 a 5). Como todas
tem o mesmo degree, o desempate vai pelo maior `alpha`: a escolhida e **degree 2, alpha 5**
— 50x mais regularizada que o minimo cru, com NMAD praticamente identico.

### Exemplo concreto (happy / mag_err)

Outro caso comum e quando varios `alpha` empatam praticamente no mesmo `degree`:

```text
degree  alpha     nmad     rmse      mae  within_1se  selected_1se
     2   10.0 0.032525 0.065706 0.042136        True          True
     2    1.0 0.032389 0.065556 0.041971        True         False
     2    0.5 0.032390 0.065548 0.041962        True         False
     2    0.1 0.032394 0.065541 0.041954        True         False
     2    2.0 0.032427 0.065573 0.041990        True         False
     2    5.0 0.032437 0.065623 0.042046        True         False
```

O menor NMAD bruto e `degree 2, alpha 1.0`, com `nmad = 0.032389`. Mesmo assim,
a regra de 1-SE escolhe `degree 2, alpha 10.0`, porque ele tambem esta dentro da
margem de erro (`within_1se = True`) e e o mais regularizado entre os empatados.
A diferenca de NMAD e pequena (`0.000136`), entao a escolha troca um ganho
minimo de validacao por um modelo mais conservador.

### Por que NMAD e nao RMSE

O RMSE e dominado por poucos outliers catastroficos e tende a apontar sempre o maior
`degree` e o menor `alpha` por margem dentro do ruido. O NMAD e robusto (baseado na
mediana) e seleciona pela qualidade do **nucleo** da distribuicao, que e o que queremos
comparar entre modelos.

Colunas relevantes na tabela: `nmad`, `nmad_se`, `selection_metric`, `selection_se`,
`selection_threshold`, `within_1se`, `selected_1se`. A config recomendada fica na
primeira linha (`selected_1se = True`).

Rodar busca em MAG:

```powershell
.\.venv\Scripts\python.exe src\redshift\training\search_polynomial_ridge.py --dataset happyT --feature-set mag --eval-split val
```

Rodar busca em MAG+ERR:

```powershell
.\.venv\Scripts\python.exe src\redshift\training\search_polynomial_ridge.py --dataset happyT --feature-set mag_err --eval-split val
```

Rodar busca em GATE-ERR (varre tambem `gate_strength`):

```powershell
.\.venv\Scripts\python.exe src\redshift\training\search_polynomial_ridge.py --dataset happyT --feature-set gate_err_manual --eval-split val
```

Customizar a grade:

```powershell
.\.venv\Scripts\python.exe src\redshift\training\search_polynomial_ridge.py --dataset happyT --feature-set mag --eval-split val --degrees 1 2 3 --alphas 1 10 100
```

A busca salva: JSON individual por combinacao, grafico individual por combinacao e a
tabela resumo ordenada pela regra de 1-SE em `reports/tables/...`.

## Busca da intensidade do GATE-ERR (Regressao Linear)

Script:

```text
src/redshift/training/search_gate_strength_linear_regression.py
```

A Regressao Linear nao tem hiperparametro proprio, mas o `gate_err_manual`
introduz o `gate_strength`. Este script testa varias intensidades **somente na
validacao** e escolhe a melhor.

Grade padrao:

```python
gate_strengths = [0.1, 0.5, 1.0, 5.0]
```

Para cada intensidade, treina em `train`, avalia em `validation` e salva uma linha
na tabela resumo. Diferente do Polynomial Ridge, a selecao **nao usa a regra de 1
desvio-padrao**: escolhe diretamente o **menor NMAD** (desempate por menor `mae` e,
em seguida, menor `gate_strength`). A configuracao escolhida fica marcada com
`selected = True` na primeira linha.

Rodar a busca:

```powershell
.\.venv\Scripts\python.exe src\redshift\training\search_gate_strength_linear_regression.py --dataset happyT
```

Customizar as intensidades (inclui `0`, que reproduz o baseline `mag`):

```powershell
.\.venv\Scripts\python.exe src\redshift\training\search_gate_strength_linear_regression.py --dataset happyT --gate-strengths 0 0.1 0.5 1 5
```

A tabela e salva em
`reports/tables/Model Selection/<dataset>/gate_err_manual/linear_regression/`.

## Teddy

Para rodar qualquer comando no Teddy, troque `--dataset happyT` por `--dataset teddyT`.

## Aviso sobre limitacao conhecida

Nos testes externos (`B`, `C` e `D`), modelos polinomiais podem produzir predicoes
extremas para objetos que caem fora do suporte do treino (extrapolacao sob mudanca de
distribuicao). Mesmo com o alvo em escala original, extrapolacoes ainda podem inflar
RMSE/MAE enquanto o NMAD permanece comportado. Esse efeito nao e corrigido pela
selecao de hiperparametros; um guarda-corpo possivel e limitar predicoes a uma faixa
fisica de redshift antes da avaliacao.
```

