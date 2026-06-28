# Boas práticas de Machine Learning para o Codex

> **Objetivo deste arquivo**: servir como guia de projeto para o Codex ao gerar, revisar ou refatorar código de Machine Learning.  
> Prioridade: **evitar data leakage**, manter experimentos **reprodutíveis** e criar scripts **reutilizáveis**, legíveis e fáceis de testar.

---

## 1. Regras de ouro

Ao mexer em qualquer projeto de ML, siga estas regras antes de pensar em modelo complexo:

1. **Separe os dados antes de aprender qualquer transformação**.  
   `train`, `validation` e `test` devem existir antes de scaler, imputação, encoder, PCA, feature selection, oversampling, target encoding ou tuning.

2. **Nunca chame `fit`, `fit_transform` ou ajuste estatísticas usando validação/teste**.  
   Transformações aprendidas devem usar apenas treino. Validação e teste recebem apenas `transform`.

3. **Use `Pipeline`/`ColumnTransformer` sempre que houver pré-processamento**.  
   Isso reduz o risco de aplicar uma etapa no conjunto errado e facilita cross-validation.

4. **Validação escolhe modelo; teste só estima desempenho final**.  
   Não use teste para escolher hiperparâmetro, feature, threshold, técnica de balanceamento ou arquitetura.

5. **Registre tudo que define o experimento**.  
   Seeds, versão dos dados, colunas usadas, split, pré-processamento, modelo, hiperparâmetros, métrica e caminho dos artefatos.

6. **Comece simples e sólido**.  
   Baseline bem feito > modelo complexo com pipeline frágil. Primeiro garanta que o fluxo ponta a ponta está correto.

---

## 2. Checklist anti-data leakage

Antes de entregar qualquer código, verifique estes pontos.

### 2.1 Split

- [ ] O split foi feito **antes** de qualquer pré-processamento aprendido?
- [ ] O `random_state` foi definido quando o split é aleatório?
- [ ] A estratificação foi usada em classificação quando há desbalanceamento relevante?
- [ ] Em dados temporais, o split respeita o tempo? Ex.: treino no passado, validação/teste no futuro.
- [ ] Em dados com grupos repetidos, foi considerado `GroupKFold`, `GroupShuffleSplit` ou split por grupo?
- [ ] O conjunto de teste ficou intocado até a avaliação final?

### 2.2 Pré-processamento

Etapas que **devem ser ajustadas apenas no treino**:

- [ ] `StandardScaler`, `MinMaxScaler`, `RobustScaler`.
- [ ] `SimpleImputer`, `KNNImputer`, imputação por média/mediana/moda.
- [ ] `OneHotEncoder`, `OrdinalEncoder`, `TargetEncoder`.
- [ ] PCA, seleção de features, remoção por correlação, remoção por variância.
- [ ] `PolynomialFeatures`, criação de features dependentes de estatísticas globais.
- [ ] SMOTE, undersampling, oversampling e qualquer balanceamento.
- [ ] Escolha de threshold baseada em métrica.

Use este padrão:

```python
preprocessor.fit(X_train)
X_train_t = preprocessor.transform(X_train)
X_val_t = preprocessor.transform(X_val)
X_test_t = preprocessor.transform(X_test)
```

Ou, preferencialmente:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", model),
])

pipeline.fit(X_train, y_train)
val_pred = pipeline.predict(X_val)
test_pred = pipeline.predict(X_test)
```

### 2.3 Balanceamento de classes

Nunca faça isto:

```python
# ERRADO: aplica SMOTE no dataset inteiro antes do split
X_res, y_res = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res)
```

Faça isto:

```python
# CERTO: split primeiro, balanceamento só dentro do treino
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42
)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
```

Melhor ainda, use `imblearn.pipeline.Pipeline` para garantir que o resampling aconteça apenas nas folds de treino durante cross-validation.

### 2.4 Features temporais e históricas

Para datasets com tempo, como esportes, finanças, séries temporais ou eventos históricos:

- [ ] A feature existia **antes** do momento da previsão?
- [ ] Estatísticas acumuladas foram calculadas só com eventos anteriores?
- [ ] Médias móveis, rankings, streaks e contagens não usam o evento atual nem eventos futuros?
- [ ] O target não aparece disfarçado em uma coluna pós-evento?
- [ ] A data foi usada para ordenar o pipeline e validar a ausência de futuro no treino?

Exemplo de regra para dados de luta/esporte:

> Para prever uma luta, qualquer atributo do atleta precisa representar apenas o histórico disponível **antes da luta**. Não usar estatísticas atualizadas depois da luta-alvo.

### 2.5 Colunas suspeitas

Investigue ou remova colunas que pareçam:

- Resultado pós-evento: `method`, `finish_round`, `total_strikes_after_fight`, `result_details`.
- Identificadores com target embutido: `winner_id`, `loser_id`, nomes de arquivos separados por classe.
- Agregações calculadas no dataset completo.
- Odds, rankings ou probabilidades que podem ter sido coletadas depois do evento.
- Colunas que têm valores nulos ou preenchidos de forma diferente dependendo da classe.

Sobre **odds**: odds podem ser válidas se o objetivo for simular uma previsão feita imediatamente antes do evento. Mesmo assim, rode dois cenários separados:

1. Modelo **sem odds**, para medir capacidade das estatísticas esportivas/históricas.
2. Modelo **com odds**, para medir ganho ao incorporar informação de mercado.

---

## 3. Organização recomendada do projeto

Use uma estrutura parecida com esta:

```text
projeto_ml/
│
├── data/
│   ├── raw/                 # dados originais, nunca editados manualmente
│   ├── interim/             # dados intermediários
│   └── processed/           # dados prontos para modelagem
│
├── notebooks/               # exploração e protótipos, não pipeline final
│   └── 01_eda.ipynb
│
├── reports/
│   ├── figures/             # gráficos gerados
│   └── metrics/             # métricas em json/csv
│
├── models/                  # modelos serializados
│
├── configs/
│   └── baseline.yaml        # caminhos, colunas, hiperparâmetros, seed
│
├── src/
│   └── projeto_ml/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py          # leitura, validação e split
│       ├── features.py      # criação de features seguras
│       ├── preprocessing.py # ColumnTransformer/Pipeline
│       ├── train.py         # treino
│       ├── evaluate.py      # métricas e gráficos
│       ├── predict.py       # inferência
│       └── utils.py
│
├── tests/
│   ├── test_data.py
│   ├── test_features.py
│   └── test_no_leakage.py
│
├── pyproject.toml
├── README.md
└── Makefile
```

### Papel dos notebooks

Use notebooks para:

- EDA.
- Entender distribuições.
- Testar hipóteses rapidamente.
- Fazer gráficos explicativos.

Não deixe no notebook a única versão do treino final. Quando algo funcionar, mova para `src/`.

---

## 4. Contratos dos scripts

Cada script deve ter uma responsabilidade clara.

### `data.py`

Responsável por:

- Ler dados.
- Padronizar nomes de colunas.
- Validar schema mínimo.
- Remover duplicatas óbvias.
- Fazer split seguro.
- Salvar índices ou arquivos de split.

Não deve:

- Treinar modelo.
- Ajustar scaler/encoder.
- Usar target para limpar features sem deixar isso explícito.

### `features.py`

Responsável por:

- Criar features determinísticas.
- Criar diferenças entre atributos, quando fizer sentido.
- Criar features temporais usando apenas passado.

Não deve:

- Usar validação/teste para decidir features.
- Criar estatísticas globais com o dataset inteiro.
- Calcular agregações que incluem o próprio evento previsto.

### `preprocessing.py`

Responsável por:

- Definir `ColumnTransformer`.
- Separar colunas numéricas, categóricas, ordinais e booleanas.
- Definir imputação, encoding e scaling.
- Retornar objetos sklearn compatíveis com `Pipeline`.

Exemplo:

```python
def build_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])
```

### `train.py`

Responsável por:

- Carregar config.
- Carregar split já definido.
- Construir pipeline.
- Treinar no treino.
- Escolher hiperparâmetros usando validação ou cross-validation.
- Salvar modelo, métricas e metadados.

Não deve:

- Usar teste durante seleção de modelo.
- Fazer tuning manual olhando o resultado do teste.

### `evaluate.py`

Responsável por:

- Carregar modelo treinado.
- Avaliar em validação ou teste.
- Gerar métricas.
- Gerar matriz de confusão, curva ROC/PR, resíduos ou gráficos adequados.
- Salvar resultados em `reports/metrics/`.

### `predict.py`

Responsável por:

- Carregar o pipeline inteiro salvo.
- Receber dados novos.
- Aplicar o mesmo pré-processamento.
- Retornar predições/probabilidades.

Nunca recrie scaler/encoder dentro de `predict.py`. Ele deve usar o pipeline treinado.

---

## 5. Configuração em vez de valores hardcoded

Evite caminhos, colunas e hiperparâmetros espalhados pelo código.

Use `configs/baseline.yaml`:

```yaml
seed: 42

data:
  raw_path: data/raw/dataset.csv
  target: Winner
  date_column: date
  split_strategy: chronological
  test_size: 0.15
  val_size: 0.15

features:
  numeric:
    - age_dif
    - reach_dif
    - win_dif
  categorical:
    - weight_class
    - gender
    - B_Stance
    - R_Stance

model:
  name: logistic_regression
  params:
    C: 1.0
    max_iter: 1000

metrics:
  primary: f1_macro
  secondary:
    - accuracy
    - balanced_accuracy
    - roc_auc
```

O Codex deve preferir funções que recebem parâmetros em vez de depender de variáveis globais.

---

## 6. Reprodutibilidade

Todo experimento deve salvar um arquivo de metadados como:

```json
{
  "experiment_name": "baseline_logreg_sem_odds",
  "seed": 42,
  "data_version": "raw_2026_06_14",
  "split_strategy": "chronological",
  "features": ["age_dif", "reach_dif", "win_dif"],
  "target": "Winner",
  "model": "LogisticRegression",
  "hyperparameters": {"C": 1.0, "max_iter": 1000},
  "metrics_val": {"f1_macro": 0.61},
  "metrics_test": null,
  "created_at": "2026-06-14"
}
```

Boas práticas:

- Definir `random_state` em splits e modelos quando disponível.
- Salvar versão dos dados ou hash do arquivo.
- Salvar lista final de colunas.
- Salvar pipeline completo com `joblib`, não só o modelo.
- Salvar métricas em arquivo legível (`json`, `csv` ou `yaml`).
- Não sobrescrever experimentos antigos sem querer.

---

## 7. Métricas e avaliação

Escolha métrica conforme o problema.

### Classificação balanceada

- `accuracy` pode ser aceitável.
- Também reportar matriz de confusão.

### Classificação desbalanceada

Prefira:

- `balanced_accuracy`.
- `f1_macro`.
- `precision_macro` e `recall_macro`.
- `roc_auc` ou `average_precision`, quando houver probabilidades.

Cuidado: uma acurácia alta pode esconder que o modelo ignora a classe minoritária.

### Regressão

Reportar:

- MAE.
- RMSE.
- R².
- Gráfico de resíduos.
- Erro por faixa do target, se fizer sentido.

---

## 8. Testes mínimos para projeto de ML

Crie testes pequenos, mesmo que o projeto seja acadêmico.

### Testes de dados

- [ ] O dataset tem as colunas obrigatórias?
- [ ] O target não tem valores inesperados?
- [ ] Não há duplicatas exatas indesejadas?
- [ ] Datas estão em formato correto?
- [ ] Não há colunas pós-evento na lista de features?

### Testes de split

- [ ] Índices de treino, validação e teste não se sobrepõem.
- [ ] Em split temporal, `max(data_treino) < min(data_val/teste)`.
- [ ] A distribuição do target foi registrada.

### Testes de pipeline

- [ ] O pipeline treina com um subconjunto pequeno.
- [ ] `predict` retorna o mesmo número de linhas da entrada.
- [ ] O pipeline salvo consegue ser carregado e usado.
- [ ] O código não chama `fit` no conjunto de teste.

Exemplo simples:

```python
def test_splits_do_not_overlap(train_idx, val_idx, test_idx):
    assert set(train_idx).isdisjoint(val_idx)
    assert set(train_idx).isdisjoint(test_idx)
    assert set(val_idx).isdisjoint(test_idx)
```

---

## 9. Estilo de código

Ao gerar código, o Codex deve seguir estas convenções:

- Funções pequenas e com nomes claros.
- Type hints quando ajudarem.
- Docstrings curtas explicando entrada e saída.
- Nada de caminhos absolutos do computador do usuário.
- Nada de variável global controlando comportamento importante.
- Não misturar EDA, treino, avaliação e inferência no mesmo script gigante.
- Usar `pathlib.Path` para caminhos.
- Usar `logging` em vez de muitos `print`, quando o script for reutilizável.
- Usar `argparse`, `typer` ou configs para permitir reexecução.

Exemplo de entrada por CLI:

```python
from pathlib import Path
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    return parser.parse_args()
```

---

## 10. Diferença entre notebook e script

### Notebook bom

- Explica raciocínio.
- Mostra estatísticas, gráficos e hipóteses.
- Ajuda a decidir pré-processamento.
- Não precisa ser totalmente modular.

### Script bom

- Roda do começo ao fim.
- Recebe config.
- Salva saídas.
- Pode ser testado.
- Pode ser reexecutado sem depender da ordem manual das células.

Fluxo recomendado:

1. Explorar no notebook.
2. Transformar decisões em funções dentro de `src/`.
3. Rodar treino por script.
4. Usar notebook apenas para analisar os resultados salvos.

---

## 11. Checklist para PR ou entrega final

Antes de considerar a tarefa pronta, o Codex deve verificar:

- [ ] Existe baseline simples.
- [ ] O pipeline roda ponta a ponta.
- [ ] O teste não foi usado para tuning.
- [ ] Pré-processadores são ajustados só no treino.
- [ ] Resampling, PCA e feature selection não vazam validação/teste.
- [ ] Métricas foram salvas.
- [ ] Modelo foi salvo junto com pré-processamento.
- [ ] Colunas usadas foram registradas.
- [ ] Seeds foram definidas.
- [ ] Scripts não dependem de notebook para funcionar.
- [ ] O README explica como rodar.
- [ ] Há pelo menos testes mínimos de dados/split/pipeline.

---

## 12. Instrução direta para o Codex

Quando eu pedir para criar ou alterar código de Machine Learning neste projeto:

1. Primeiro identifique a tarefa: EDA, split, features, treino, avaliação, inferência ou refatoração.
2. Preserve a separação entre treino, validação e teste.
3. Use `Pipeline`/`ColumnTransformer` quando houver pré-processamento.
4. Não use teste para escolher modelo, hiperparâmetro ou feature.
5. Evite scripts monolíticos; prefira funções reutilizáveis em `src/`.
6. Salve métricas, artefatos e metadados.
7. Explique no comentário ou docstring qualquer decisão que possa afetar leakage.
8. Se detectar possível leakage, pare e sinalize antes de continuar.
9. Sempre prefira um baseline correto a uma solução sofisticada e frágil.
10. Ao refatorar, mantenha compatibilidade com os arquivos de entrada já existentes, salvo pedido explícito.

---

## 13. Fontes consultadas

- scikit-learn — Common pitfalls and recommended practices: https://scikit-learn.org/stable/common_pitfalls.html
- imbalanced-learn — Common pitfalls and recommended practices: https://imbalanced-learn.org/stable/common_pitfalls.html
- Google Developers — Rules of Machine Learning / Best Practices for ML Engineering: https://developers.google.com/machine-learning/guides/rules-of-ml
- Cookiecutter Data Science — project structure: https://cookiecutter-data-science.drivendata.org/
- DVC — Data Version Control documentation: https://doc.dvc.org/
- MLflow — experiment tracking documentation: https://mlflow.org/docs/latest/ml/tracking/
- Sasse et al. — On Leakage in Machine Learning Pipelines: https://arxiv.org/abs/2311.04179
- Kapoor & Narayanan — Leakage and the Reproducibility Crisis in ML-based Science: https://arxiv.org/abs/2207.07048
- Kreuzberger, Kühl & Hirschl — Machine Learning Operations: Overview, Definition, and Architecture: https://arxiv.org/abs/2205.02302
