# Redshift fotométrico — baselines nos datasets COIN (Happy & Teddy)

Estimativa de **redshift fotométrico** (*photo-z*): prever o redshift de galáxias a
partir apenas de **fotometria** (brilho em poucas bandas), sem espectroscopia. É um
problema de **regressão** — features são as magnitudes (e seus erros), o alvo é o
redshift — e o foco deste repositório é construir **baselines simples, corretos e
reprodutíveis** e avaliá-los sob **mudança de distribuição** entre treino e teste.

## O problema

Medir redshift por espectroscopia é caro e lento; por fotometria é barato e
massivo, porém menos preciso. O desafio não é só ajustar bem o conjunto de treino,
mas **generalizar para populações de galáxias diferentes** das vistas no treino —
exatamente o cenário dos levantamentos reais. Por isso a avaliação aqui usa
conjuntos de teste com graus crescentes de descasamento de distribuição.

## Os dados

Dois datasets do tipo COIN/SDSS ficam em [data/raw/](data/raw):

- **`happyT`** (Happy) — arquivos `happyT_*.txt`
- **`teddyT`** (Teddy) — arquivos `teddyT_*.cat`

Cada dataset tem quatro conjuntos `A`, `B`, `C`, `D`, na lógica de *realistic
validation* (Beck et al. 2017):

- **`A`** → usado para **treino/validação** (split 80/20 estratificado por redshift);
- **`B`, `C`, `D`** → **testes externos** separados, com descasamento de distribuição
  **crescente** em relação ao treino (`B` parecido com o treino; `D` o mais
  diferente).

Cada objeto tem 5 magnitudes (`u, g, r, i, z`), seus erros fotométricos
(`uErr…zErr`), o `redshift` (alvo) e `redshiftErr`.

## Modelos e baselines

São comparados **2 modelos** × **3 conjuntos de features**:

**Modelos**
- **Regressão Linear** — baseline mais simples e robusto à extrapolação.
- **Polynomial Ridge** — `PolynomialFeatures` (grau 1–2) + `Ridge`, com busca de
  hiperparâmetros na validação.

**Conjuntos de features**
- **`mag`** — só as 5 magnitudes.
- **`mag_err`** — magnitudes + os 5 erros fotométricos (10 features).
- **`gate_err_manual`** (GATE-ERR) — 5 magnitudes **ponderadas pelo próprio erro**:
  `j_gate = j_scaled · exp(−gate_strength · jErr_norm)`. Bandas mais incertas pesam
  menos; os erros não entram como features separadas.

## Como o desempenho é medido

As métricas são calculadas na **escala original do redshift**, sobre o erro
normalizado `dz = (z_pred − z_real) / (1 + z_real)`. As principais:

- **NMAD** — dispersão robusta do núcleo (`1.4826 · mediana(|dz − mediana(dz)|)`),
  a métrica de seleção;
- **RMSE / MAE / R² / bias**;
- **fração de outliers catastróficos** (`|dz| > 0.15`) e **predições negativas**.

Regra de ouro do projeto: **validação escolhe o modelo; o teste só estima o
desempenho final** — nunca se usa `B/C/D` para escolher hiperparâmetro. O controle
de vazamento (fit só no treino) é tratado em todo o pipeline.

## Mapa do repositório

| Caminho | O que é |
|---|---|
| [src/redshift/](src/redshift) | Código-fonte: pré-processamento, modelos, buscas e avaliação |
| [src/redshift/README.md](src/redshift/README.md) | **Documentação detalhada do pipeline** (comandos, fórmulas, métricas) |
| [data/raw/](data/raw) | Catálogos brutos `happyT` e `teddyT` (A/B/C/D) |
| [data/processed/](data/processed) | Splits pré-processados de `mag` / `mag_err` (Val e Test) |
| `data/processed Gate-Err/` | Splits pré-processados isolados do GATE-ERR |
| [models/](models) | Modelos treinados (`.joblib`), por `Model Selection` / `Tests` |
| [reports/metrics/](reports/metrics) | Métricas `.json` por experimento |
| [reports/metrics/Results Table/](reports/metrics/Results%20Table) | Tabelas comparativas em Markdown (global e por faixa de redshift) |
| [reports/tables/](reports/tables) | Tabelas `.csv` das buscas de hiperparâmetros |
| [reports/figures/](reports/figures) | Gráficos de avaliação (real × previsto e resíduos) |
| [notebooks/](notebooks) | Análise exploratória (EDA) dos dados brutos e processados |
| [codex.md](codex.md) | Guia de boas práticas de ML do projeto (anti-vazamento, reprodutibilidade) |
| [requirements.txt](requirements.txt) | Dependências Python |

Dentro de `models/`, `reports/metrics/`, `reports/tables/` e `reports/figures/` a
organização segue sempre o mesmo padrão:

```text
<Model Selection ou Tests>/<dataset>/<feature_set>/<modelo>/
```

## Começando

```powershell
# 1. Ambiente
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. Pré-processar um dataset (gera os modos Val e Test)
.\.venv\Scripts\python.exe src\redshift\data\preprocess.py --dataset teddyT
.\.venv\Scripts\python.exe src\redshift\data\preprocess_gate_err.py --dataset teddyT

# 3. Rodar um baseline na validação
.\.venv\Scripts\python.exe src\redshift\models\linear_regression.py --dataset teddyT --feature-set mag --eval-split val

# 4. Avaliação final nos testes externos B, C e D
.\.venv\Scripts\python.exe src\redshift\models\linear_regression.py --dataset teddyT --feature-set mag --eval-split test --test-set all
```

Cada execução salva modelo (`.joblib`), métricas (`.json`) e gráfico (`.png`) nos
diretórios correspondentes.

## Documentação detalhada

Para o passo a passo completo — pré-processamento, fórmulas do GATE-ERR, métricas
por faixa de redshift, busca de hiperparâmetros (regra de 1 desvio-padrão) e todos
os comandos por modelo e dataset — veja:

➡️ **[src/redshift/README.md](src/redshift/README.md)**

As conclusões comparativas entre baselines e modelos estão consolidadas em
[reports/metrics/Results Table/](reports/metrics/Results%20Table).
