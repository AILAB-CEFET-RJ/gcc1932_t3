# Comparação Linear Regression × Polynomial Ridge

Comparação direta dos dois modelos nos baselines `mag`, `mag_err` e `gate-err`, avaliados nos testes externos `B`, `C` e `D`.

Todos os valores foram extraídos diretamente dos JSONs em `reports/metrics/Tests/`. Em cada teste, o melhor valor entre as seis combinações (modelo × baseline) está em **negrito**.

## Métricas reportadas

| Métrica | Interpretação | Melhor direção |
|---|---|---|
| MAE | erro absoluto médio | menor |
| RMSE | raiz do erro quadrático médio; penaliza mais erros grandes | menor |
| R² | proporção da variância explicada | maior |
| Bias | média do erro de previsão | mais próximo de 0 |
| NMAD | métrica robusta de dispersão do erro | menor |
| Outliers catastróficos | fração de objetos com erro acima do limiar configurado | menor |
| Predições negativas | número de previsões de redshift negativo | menor |

## happyT

| Teste | Baseline | Modelo | N | MAE | RMSE | R² | Bias | NMAD | Out Cat | Prev neg |
|:---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B | `mag` | Linear | 74900 | 0.0484 | 0.0767 | 0.8684 | -0.0003 | 0.0369 | 1.90% | 1631 |
| B | `mag` | Poli | 74900 | 0.0429 | 0.0703 | 0.8892 | -0.0001 | 0.0321 | 1.57% | 805 |
| B | `mag_err` | Linear | 74900 | 0.0438 | 0.0728 | 0.8814 | -0.0002 | 0.0327 | 1.67% | 911 |
| B | `mag_err` | Poli | 74900 | **0.0378** | **0.0660** | **0.9025** | **0.0000** | **0.0269** | **1.31%** | **267** |
| B | `gate-err` | Linear | 74900 | 0.0483 | 0.0765 | 0.8690 | -0.0004 | 0.0366 | 1.97% | 1765 |
| B | `gate-err` | Poli | 74900 | 0.0430 | 0.0703 | 0.8894 | -0.0001 | 0.0321 | 1.56% | 862 |
|  |  |  |  |  |  |  |  |  |  |  |
| C | `mag` | Linear | 60315 | 0.0995 | 0.1322 | 0.4775 | 0.0343 | 0.0779 | 11.05% | **163** |
| C | `mag` | Poli | 60315 | 0.0975 | 0.1311 | 0.4862 | 0.0422 | 0.0732 | 10.97% | 202 |
| C | `mag_err` | Linear | 60315 | 0.0934 | 0.1289 | 0.5033 | **0.0200** | 0.0735 | 9.54% | 210 |
| C | `mag_err` | Poli | 60315 | **0.0842** | **0.1206** | **0.5652** | 0.0209 | **0.0636** | **7.62%** | 187 |
| C | `gate-err` | Linear | 60315 | 0.0991 | 0.1321 | 0.4786 | 0.0334 | 0.0778 | 11.02% | 175 |
| C | `gate-err` | Poli | 60315 | 0.0979 | 0.1315 | 0.4828 | 0.0441 | 0.0730 | 11.11% | 188 |
|  |  |  |  |  |  |  |  |  |  |  |
| D | `mag` | Linear | 74642 | 0.1363 | 0.1958 | 0.2129 | -0.0191 | 0.1059 | 18.11% | 485 |
| D | `mag` | Poli | 74642 | 0.1328 | **0.1880** | **0.2741** | 0.0254 | 0.0989 | 18.08% | 447 |
| D | `mag_err` | Linear | 74642 | 0.1306 | 0.1946 | 0.2224 | **0.0007** | 0.0985 | 17.07% | 428 |
| D | `mag_err` | Poli | 74642 | **0.1252** | 0.2157 | 0.0451 | -0.0010 | **0.0905** | **15.18%** | **301** |
| D | `gate-err` | Linear | 74642 | 0.1367 | 0.1969 | 0.2043 | -0.0225 | 0.1057 | 18.31% | 493 |
| D | `gate-err` | Poli | 74642 | 0.1328 | 0.1886 | 0.2699 | 0.0224 | 0.0992 | 18.07% | 418 |

## teddyT

| Teste | Baseline | Modelo | N | MAE | RMSE | R² | Bias | NMAD | Out Cat | Prev neg |
|:---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B | `mag` | Linear | 74559 | 0.0266 | 0.0377 | 0.8161 | -0.0001 | 0.0216 | **0.10%** | **0** |
| B | `mag` | Poli | 74559 | 0.0240 | 0.0350 | 0.8411 | -0.0000 | 0.0190 | 0.14% | **0** |
| B | `mag_err` | Linear | 74559 | 0.0257 | 0.0368 | 0.8244 | **-0.0000** | 0.0205 | 0.11% | **0** |
| B | `mag_err` | Poli | 74559 | **0.0233** | **0.0345** | **0.8456** | 0.0000 | **0.0181** | 0.15% | 1 |
| B | `gate-err` | Linear | 74559 | 0.0269 | 0.0379 | 0.8133 | -0.0001 | 0.0220 | 0.11% | **0** |
| B | `gate-err` | Poli | 74559 | 0.0240 | 0.0351 | 0.8400 | -0.0001 | 0.0188 | 0.14% | **0** |
|  |  |  |  |  |  |  |  |  |  |  |
| C | `mag` | Linear | 97980 | 0.0282 | 0.0406 | 0.8337 | -0.0006 | 0.0223 | 0.18% | **0** |
| C | `mag` | Poli | 97980 | 0.0254 | 0.0376 | 0.8571 | -0.0005 | 0.0195 | 0.23% | **0** |
| C | `mag_err` | Linear | 97980 | 0.0273 | 0.0398 | 0.8398 | -0.0006 | 0.0212 | 0.19% | **0** |
| C | `mag_err` | Poli | 97980 | **0.0246** | **0.0372** | **0.8604** | -0.0007 | **0.0185** | 0.24% | 1 |
| C | `gate-err` | Linear | 97980 | 0.0284 | 0.0408 | 0.8322 | -0.0011 | 0.0226 | **0.17%** | **0** |
| C | `gate-err` | Poli | 97980 | 0.0253 | 0.0377 | 0.8562 | **-0.0005** | 0.0193 | 0.24% | **0** |
|  |  |  |  |  |  |  |  |  |  |  |
| D | `mag` | Linear | 75925 | 0.0514 | 0.0955 | 0.8072 | **-0.0018** | 0.0372 | 2.01% | 1779 |
| D | `mag` | Poli | 75925 | 0.1076 | 0.5041 | -4.3740 | 0.0817 | 0.0663 | 11.55% | 1903 |
| D | `mag_err` | Linear | 75925 | 0.0493 | **0.0908** | **0.8256** | 0.0020 | 0.0354 | **1.75%** | 1784 |
| D | `mag_err` | Poli | 75925 | 0.1006 | 0.5427 | -5.2294 | 0.0050 | 0.0441 | 8.11% | 1187 |
| D | `gate-err` | Linear | 75925 | **0.0488** | 0.0919 | 0.8214 | -0.0103 | **0.0352** | 1.87% | 2251 |
| D | `gate-err` | Poli | 75925 | 0.0913 | 0.3574 | -1.7009 | 0.0697 | 0.0567 | 8.56% | **667** |
