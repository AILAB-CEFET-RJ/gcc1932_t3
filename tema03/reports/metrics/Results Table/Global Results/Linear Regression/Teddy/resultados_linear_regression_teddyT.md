# Resultados dos baselines - teddyT - Linear Regression

Resultados dos baselines `mag`, `mag_err` e `gate-err` no dataset `teddyT`, avaliados separadamente nos testes externos `B`, `C` e `D`.

Todos os valores foram extraídos diretamente dos JSONs em `reports/metrics/Tests/`. Em cada teste, o melhor valor entre os três baselines está em **negrito**.

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

## Comparação geral entre os baselines

| Teste | Baseline | MAE | RMSE | R² | Bias | NMAD | Outliers catastróficos | Predições negativas | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B | `mag` | 0.0266 | 0.0377 | 0.8161 | -0.0001 | 0.0216 | **0.10%** | **0** | 74559 |
| B | `mag_err` | **0.0257** | **0.0368** | **0.8244** | **-0.0000** | **0.0205** | 0.11% | **0** | 74559 |
| B | `gate-err` | 0.0269 | 0.0379 | 0.8133 | -0.0001 | 0.0220 | 0.11% | **0** | 74559 |
|  |  |  |  |  |  |  |  |  |  |
| C | `mag` | 0.0282 | 0.0406 | 0.8337 | -0.0006 | 0.0223 | 0.18% | **0** | 97980 |
| C | `mag_err` | **0.0273** | **0.0398** | **0.8398** | **-0.0006** | **0.0212** | 0.19% | **0** | 97980 |
| C | `gate-err` | 0.0284 | 0.0408 | 0.8322 | -0.0011 | 0.0226 | **0.17%** | **0** | 97980 |
|  |  |  |  |  |  |  |  |  |  |
| D | `mag` | 0.0514 | 0.0955 | 0.8072 | **-0.0018** | 0.0372 | 2.01% | **1779** | 75925 |
| D | `mag_err` | 0.0493 | **0.0908** | **0.8256** | 0.0020 | 0.0354 | **1.75%** | 1784 | 75925 |
| D | `gate-err` | **0.0488** | 0.0919 | 0.8214 | -0.0103 | **0.0352** | 1.87% | 2251 | 75925 |

## Resultados por baseline e teste

## Baseline 1 - Linear Regression com magnitudes (`mag`)

- **Feature set:** `mag`
- **Features finais:** `u`, `g`, `r`, `i`, `z`
- **Parâmetros selecionados:** fit_intercept=True

| Teste | N | MAE | RMSE | R² | Bias | NMAD | Outliers catastróficos | Predições negativas |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B | 74559 | 0.0266 | 0.0377 | 0.8161 | -0.0001 | 0.0216 | 0.10% | 0 |
| C | 97980 | 0.0282 | 0.0406 | 0.8337 | -0.0006 | 0.0223 | 0.18% | 0 |
| D | 75925 | 0.0514 | 0.0955 | 0.8072 | -0.0018 | 0.0372 | 2.01% | 1779 |

### Teste B

- **Experimento:** `linear_regression_mag`
- **Parâmetros:** fit_intercept=True

| Métrica | Valor |
|---|---:|
| N | 74559 |
| MAE | 0.026579 |
| RMSE | 0.037651 |
| R² | 0.816063 |
| Bias | -0.000113 |
| NMAD | 0.021644 |
| Erro-padrão do NMAD | 0.000096 |
| Fração de outliers catastróficos | 0.10% |
| Limiar de outlier catastrófico | 0.15 |
| Predições negativas de redshift | 0 |

### Teste C

- **Experimento:** `linear_regression_mag`
- **Parâmetros:** fit_intercept=True

| Métrica | Valor |
|---|---:|
| N | 97980 |
| MAE | 0.028162 |
| RMSE | 0.040598 |
| R² | 0.833717 |
| Bias | -0.000601 |
| NMAD | 0.022253 |
| Erro-padrão do NMAD | 0.000088 |
| Fração de outliers catastróficos | 0.18% |
| Limiar de outlier catastrófico | 0.15 |
| Predições negativas de redshift | 0 |

### Teste D

- **Experimento:** `linear_regression_mag`
- **Parâmetros:** fit_intercept=True

| Métrica | Valor |
|---|---:|
| N | 75925 |
| MAE | 0.051407 |
| RMSE | 0.095472 |
| R² | 0.807243 |
| Bias | -0.001801 |
| NMAD | 0.037154 |
| Erro-padrão do NMAD | 0.000155 |
| Fração de outliers catastróficos | 2.01% |
| Limiar de outlier catastrófico | 0.15 |
| Predições negativas de redshift | 1779 |

## Baseline 2 - Linear Regression com magnitudes + erros fotométricos (`mag_err`)

- **Feature set:** `mag_err`
- **Features finais:** `u`, `g`, `r`, `i`, `z`, `uErr`, `gErr`, `rErr`, `iErr`, `zErr`
- **Parâmetros selecionados:** fit_intercept=True

| Teste | N | MAE | RMSE | R² | Bias | NMAD | Outliers catastróficos | Predições negativas |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B | 74559 | 0.0257 | 0.0368 | 0.8244 | -0.0000 | 0.0205 | 0.11% | 0 |
| C | 97980 | 0.0273 | 0.0398 | 0.8398 | -0.0006 | 0.0212 | 0.19% | 0 |
| D | 75925 | 0.0493 | 0.0908 | 0.8256 | 0.0020 | 0.0354 | 1.75% | 1784 |

### Teste B

- **Experimento:** `linear_regression_mag_err`
- **Parâmetros:** fit_intercept=True

| Métrica | Valor |
|---|---:|
| N | 74559 |
| MAE | 0.025709 |
| RMSE | 0.036790 |
| R² | 0.824378 |
| Bias | -0.000015 |
| NMAD | 0.020548 |
| Erro-padrão do NMAD | 0.000089 |
| Fração de outliers catastróficos | 0.11% |
| Limiar de outlier catastrófico | 0.15 |
| Predições negativas de redshift | 0 |

### Teste C

- **Experimento:** `linear_regression_mag_err`
- **Parâmetros:** fit_intercept=True

| Métrica | Valor |
|---|---:|
| N | 97980 |
| MAE | 0.027349 |
| RMSE | 0.039844 |
| R² | 0.839833 |
| Bias | -0.000591 |
| NMAD | 0.021178 |
| Erro-padrão do NMAD | 0.000085 |
| Fração de outliers catastróficos | 0.19% |
| Limiar de outlier catastrófico | 0.15 |
| Predições negativas de redshift | 0 |

### Teste D

- **Experimento:** `linear_regression_mag_err`
- **Parâmetros:** fit_intercept=True

| Métrica | Valor |
|---|---:|
| N | 75925 |
| MAE | 0.049321 |
| RMSE | 0.090807 |
| R² | 0.825618 |
| Bias | 0.001960 |
| NMAD | 0.035395 |
| Erro-padrão do NMAD | 0.000159 |
| Fração de outliers catastróficos | 1.75% |
| Limiar de outlier catastrófico | 0.15 |
| Predições negativas de redshift | 1784 |

## Baseline 3 - Linear Regression com magnitudes ponderadas pelos erros (`gate-err`)

- **Feature set:** `gate_err_manual`
- **Features finais:** `u_gate`, `g_gate`, `r_gate`, `i_gate`, `z_gate`
- **Parâmetros selecionados:** gate_strength=0.1, fit_intercept=True

| Teste | N | MAE | RMSE | R² | Bias | NMAD | Outliers catastróficos | Predições negativas |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B | 74559 | 0.0269 | 0.0379 | 0.8133 | -0.0001 | 0.0220 | 0.11% | 0 |
| C | 97980 | 0.0284 | 0.0408 | 0.8322 | -0.0011 | 0.0226 | 0.17% | 0 |
| D | 75925 | 0.0488 | 0.0919 | 0.8214 | -0.0103 | 0.0352 | 1.87% | 2251 |

### Teste B

- **Experimento:** `linear_regression_gate_err_manual_strength0.1`
- **Parâmetros:** gate_strength=0.1, fit_intercept=True

| Métrica | Valor |
|---|---:|
| N | 74559 |
| MAE | 0.026887 |
| RMSE | 0.037931 |
| R² | 0.813313 |
| Bias | -0.000122 |
| NMAD | 0.021993 |
| Erro-padrão do NMAD | 0.000095 |
| Fração de outliers catastróficos | 0.11% |
| Limiar de outlier catastrófico | 0.15 |
| Predições negativas de redshift | 0 |

### Teste C

- **Experimento:** `linear_regression_gate_err_manual_strength0.1`
- **Parâmetros:** gate_strength=0.1, fit_intercept=True

| Métrica | Valor |
|---|---:|
| N | 97980 |
| MAE | 0.028391 |
| RMSE | 0.040788 |
| R² | 0.832156 |
| Bias | -0.001113 |
| NMAD | 0.022588 |
| Erro-padrão do NMAD | 0.000091 |
| Fração de outliers catastróficos | 0.17% |
| Limiar de outlier catastrófico | 0.15 |
| Predições negativas de redshift | 0 |

### Teste D

- **Experimento:** `linear_regression_gate_err_manual_strength0.1`
- **Parâmetros:** gate_strength=0.1, fit_intercept=True

| Métrica | Valor |
|---|---:|
| N | 75925 |
| MAE | 0.048751 |
| RMSE | 0.091908 |
| R² | 0.821364 |
| Bias | -0.010326 |
| NMAD | 0.035192 |
| Erro-padrão do NMAD | 0.000157 |
| Fração de outliers catastróficos | 1.87% |
| Limiar de outlier catastrófico | 0.15 |
| Predições negativas de redshift | 2251 |
