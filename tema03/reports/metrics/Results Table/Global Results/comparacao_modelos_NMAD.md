# Comparação Linear Regression × Polynomial Ridge — NMAD

Comparação direta dos dois modelos nos baselines `mag`, `mag_err` e `gate-err`, avaliados nos testes externos `B`, `C` e `D`. Esta versão apresenta somente a métrica **NMAD**.

Todos os valores foram extraídos diretamente dos JSONs em `reports/metrics/Tests/`. Em cada teste, o melhor valor entre as seis combinações (modelo × baseline) está em **negrito**.

## Métrica reportada

| Métrica | Interpretação | Melhor direção |
|---|---|---|
| NMAD | métrica robusta de dispersão do erro | menor |

## happyT

| Teste | Baseline | Modelo | N | NMAD |
|:---:|---|---|---:|---:|
| B | `mag` | Linear | 74900 | 0.0369 |
| B | `mag` | Poli | 74900 | 0.0321 |
| B | `mag_err` | Linear | 74900 | 0.0327 |
| B | `mag_err` | Poli | 74900 | **0.0269** |
| B | `gate-err` | Linear | 74900 | 0.0366 |
| B | `gate-err` | Poli | 74900 | 0.0321 |
|  |  |  |  |  |
| C | `mag` | Linear | 60315 | 0.0779 |
| C | `mag` | Poli | 60315 | 0.0732 |
| C | `mag_err` | Linear | 60315 | 0.0735 |
| C | `mag_err` | Poli | 60315 | **0.0636** |
| C | `gate-err` | Linear | 60315 | 0.0778 |
| C | `gate-err` | Poli | 60315 | 0.0730 |
|  |  |  |  |  |
| D | `mag` | Linear | 74642 | 0.1059 |
| D | `mag` | Poli | 74642 | 0.0989 |
| D | `mag_err` | Linear | 74642 | 0.0985 |
| D | `mag_err` | Poli | 74642 | **0.0905** |
| D | `gate-err` | Linear | 74642 | 0.1057 |
| D | `gate-err` | Poli | 74642 | 0.0992 |

## teddyT

| Teste | Baseline | Modelo | N | NMAD |
|:---:|---|---|---:|---:|
| B | `mag` | Linear | 74559 | 0.0216 |
| B | `mag` | Poli | 74559 | 0.0190 |
| B | `mag_err` | Linear | 74559 | 0.0205 |
| B | `mag_err` | Poli | 74559 | **0.0181** |
| B | `gate-err` | Linear | 74559 | 0.0220 |
| B | `gate-err` | Poli | 74559 | 0.0188 |
|  |  |  |  |  |
| C | `mag` | Linear | 97980 | 0.0223 |
| C | `mag` | Poli | 97980 | 0.0195 |
| C | `mag_err` | Linear | 97980 | 0.0212 |
| C | `mag_err` | Poli | 97980 | **0.0185** |
| C | `gate-err` | Linear | 97980 | 0.0226 |
| C | `gate-err` | Poli | 97980 | 0.0193 |
|  |  |  |  |  |
| D | `mag` | Linear | 75925 | 0.0372 |
| D | `mag` | Poli | 75925 | 0.0663 |
| D | `mag_err` | Linear | 75925 | 0.0354 |
| D | `mag_err` | Poli | 75925 | 0.0441 |
| D | `gate-err` | Linear | 75925 | **0.0352** |
| D | `gate-err` | Poli | 75925 | 0.0567 |
