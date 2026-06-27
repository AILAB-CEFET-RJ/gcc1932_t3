# Trabalho 3 — NER no Disque Denúncia

Tema 6 de CIC1205 (Aprendizado de Máquina). Reconhecimento de entidades nomeadas (PESSOA, LOCAL, ORG) em relatos do Disque Denúncia, usando pseudo-rotulagem lexical para aproveitar dados não anotados.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

O `pt_core_news_lg` (modelo spaCy em português) já está listado no `requirements.txt` e é instalado junto com as outras dependências.

## Dados

Os arquivos de `data/raw/DDsmall` e `data/raw/DDlarge` não estão no repositório. Eles devem ser solicitados ao Gustavo Melo e colocados nesses dois diretórios antes de rodar os notebooks 01 a 03.

## Rodando os notebooks

Abra os notebooks em `notebooks/` com Jupyter ou no VS Code e execute na ordem abaixo, sempre com o kernel na raiz do projeto (os notebooks fazem `sys.path.insert` para importar `src/`):

1. **01_eda.ipynb** — análise exploratória do DDsmall.
2. **02_lexicon.ipynb** — extração e filtragem do léxico de entidades a partir do treino.
3. **03_projection.ipynb** — projeção do léxico sobre o DDlarge para gerar o corpus pseudo-anotado.
4. **04_training.ipynb** — treino e avaliação dos 4 modelos (Baseline 1, Proposta 1, Baseline 2, Proposta 2). Veja a tabela de pré-requisitos no topo do notebook: as células de treino (Baseline 2 e Proposta 2) precisam rodar pelo menos uma vez para salvar o modelo em `models/`; as células de avaliação podem ser reexecutadas isoladamente depois.
5. **05_evaluation.ipynb** — métricas comparativas finais e análise qualitativa de erros, lendo os modelos já salvos em `models/`.

Os notebooks 03 e 04 dependem dos artefatos salvos pelos anteriores (`data/lexicon/lexicon_filtered.json` e `data/processed/ddlarge_pseudo.jsonl`, respectivamente). Se esses arquivos e os modelos em `models/` já estiverem presentes, é possível pular direto para o 04 ou 05 caso só queira reproduzir o treino e a avaliação.

## Estrutura

```
data/raw/         dados brutos (DDsmall, DDlarge)
data/lexicon/     léxico filtrado
data/processed/   corpus pseudo-anotado do DDlarge
src/              módulos usados pelos notebooks
notebooks/        pipeline numerado (01 a 05)
models/           modelos treinados
results/          métricas e análise de erros
```
