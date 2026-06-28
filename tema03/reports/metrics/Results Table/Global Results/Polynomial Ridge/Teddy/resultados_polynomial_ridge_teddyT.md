# Resultados dos baselines - teddyT - Polynomial Ridge

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
| B | `mag` | 0.0240 | 0.0350 | 0.8411 | -0.0000 | 0.0190 | 0.14% | **0** | 74559 |
| B | `mag_err` | **0.0233** | **0.0345** | **0.8456** | **0.0000** | **0.0181** | 0.15% | 1 | 74559 |
| B | `gate-err` | 0.0240 | 0.0351 | 0.8400 | -0.0001 | 0.0188 | **0.14%** | **0** | 74559 |
|  |  |  |  |  |  |  |  |  |  |
| C | `mag` | 0.0254 | 0.0376 | 0.8571 | -0.0005 | 0.0195 | **0.23%** | **0** | 97980 |
| C | `mag_err` | **0.0246** | **0.0372** | **0.8604** | -0.0007 | **0.0185** | 0.24% | 1 | 97980 |
| C | `gate-err` | 0.0253 | 0.0377 | 0.8562 | **-0.0005** | 0.0193 | 0.24% | **0** | 97980 |
|  |  |  |  |  |  |  |  |  |  |
| D | `mag` | 0.1076 | 0.5041 | -4.3740 | 0.0817 | 0.0663 | 11.55% | 1903 | 75925 |
| D | `mag_err` | 0.1006 | 0.5427 | -5.2294 | **0.0050** | **0.0441** | **8.11%** | 1187 | 75925 |
| D | `gate-err` | **0.0913** | **0.3574** | **-1.7009** | 0.0697 | 0.0567 | 8.56% | **667** | 75925 |

## Resultados por baseline e teste

## Baseline 1 - Polynomial Ridge com magnitudes (`mag`)

- **Feature set:** `mag`
- **Features finais:** `u`, `g`, `r`, `i`, `z`
- **Parâmetros selecionados:** grau=2, alpha Ridge=0.5

| Teste | N | MAE | RMSE | R² | Bias | NMAD | Outliers catastróficos | Predições negativas |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B | 74559 | 0.0240 | 0.0350 | 0.8411 | -0.0000 | 0.0190 | 0.14% | 0 |
| C | 97980 | 0.0254 | 0.0376 | 0.8571 | -0.0005 | 0.0195 | 0.23% | 0 |
| D | 75925 | 0.1076 | 0.5041 | -4.3740 | 0.0817 | 0.0663 | 11.55% | 1903 |

### Teste B

- **Experimento:** `polynomial_ridge_mag_degree2_alpha0.5`
- **Parâmetros:** grau=2, alpha Ridge=0.5

| Métrica | Valor |
|---|---:|
| N | 74559 |
| MAE | 0.024039 |
| RMSE | 0.034996 |
| R² | 0.841087 |
| Bias | -0.000039 |
| NMAD | 0.019048 |
| Erro-padrão do NMAD | 0.000088 |
| Fração de outliers catastróficos | 0.14% |
| Limiar de outlier catastrófico | 0.15 |
| Predições negativas de redshift | 0 |

### Teste C

- **Experimento:** `polynomial_ridge_mag_degree2_alpha0.5`
- **Parâmetros:** grau=2, alpha Ridge=0.5

| Métrica | Valor |
|---|---:|
| N | 97980 |
| MAE | 0.025375 |
| RMSE | 0.037640 |
| R² | 0.857064 |
| Bias | -0.000539 |
| NMAD | 0.019521 |
| Erro-padrão do NMAD | 0.000075 |
| Fração de outliers catastróficos | 0.23% |
| Limiar de outlier catastrófico | 0.15 |
| Predições negativas de redshift | 0 |

### Teste D

- **Experimento:** `polynomial_ridge_mag_degree2_alpha0.5`
- **Parâmetros:** grau=2, alpha Ridge=0.5

| Métrica | Valor |
|---|---:|
| N | 75925 |
| MAE | 0.107609 |
| RMSE | 0.504102 |
| R² | -4.373959 |
| Bias | 0.081716 |
| NMAD | 0.066268 |
| Erro-padrão do NMAD | 0.000339 |
| Fração de outliers catastróficos | 11.55% |
| Limiar de outlier catastrófico | 0.15 |
| Predições negativas de redshift | 1903 |

## Baseline 2 - Polynomial Ridge com magnitudes + erros fotométricos (`mag_err`)

- **Feature set:** `mag_err`
- **Features finais:** `u`, `g`, `r`, `i`, `z`, `uErr`, `gErr`, `rErr`, `iErr`, `zErr`
- **Parâmetros selecionados:** grau=2, alpha Ridge=5

| Teste | N | MAE | RMSE | R² | Bias | NMAD | Outliers catastróficos | Predições negativas |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B | 74559 | 0.0233 | 0.0345 | 0.8456 | 0.0000 | 0.0181 | 0.15% | 1 |
| C | 97980 | 0.0246 | 0.0372 | 0.8604 | -0.0007 | 0.0185 | 0.24% | 1 |
| D | 75925 | 0.1006 | 0.5427 | -5.2294 | 0.0050 | 0.0441 | 8.11% | 1187 |

### Teste B

- **Experimento:** `polynomial_ridge_mag_err_degree2_alpha5`
- **Parâmetros:** grau=2, alpha Ridge=5

| Métrica | Valor |
|---|---:|
| N | 74559 |
| MAE | 0.023330 |
| RMSE | 0.034491 |
| R² | 0.845646 |
| Bias | 0.000021 |
| NMAD | 0.018135 |
| Erro-padrão do NMAD | 0.000084 |
| Fração de outliers catastróficos | 0.15% |
| Limiar de outlier catastrófico | 0.15 |
| Predições negativas de redshift | 1 |

### Teste C

- **Experimento:** `polynomial_ridge_mag_err_degree2_alpha5`
- **Parâmetros:** grau=2, alpha Ridge=5

| Métrica | Valor |
|---|---:|
| N | 97980 |
| MAE | 0.024649 |
| RMSE | 0.037192 |
| R² | 0.860448 |
| Bias | -0.000655 |
| NMAD | 0.018469 |
| Erro-padrão do NMAD | 0.000081 |
| Fração de outliers catastróficos | 0.24% |
| Limiar de outlier catastrófico | 0.15 |
| Predições negativas de redshift | 1 |

### Teste D

- **Experimento:** `polynomial_ridge_mag_err_degree2_alpha5`
- **Parâmetros:** grau=2, alpha Ridge=5

| Métrica | Valor |
|---|---:|
| N | 75925 |
| MAE | 0.100568 |
| RMSE | 0.542741 |
| R² | -5.229365 |
| Bias | 0.004982 |
| NMAD | 0.044098 |
| Erro-padrão do NMAD | 0.000251 |
| Fração de outliers catastróficos | 8.11% |
| Limiar de outlier catastrófico | 0.15 |
| Predições negativas de redshift | 1187 |

## Baseline 3 - Polynomial Ridge com magnitudes ponderadas pelos erros (`gate-err`)

- **Feature set:** `gate_err_manual`
- **Features finais:** `u_gate`, `g_gate`, `r_gate`, `i_gate`, `z_gate`
- **Parâmetros selecionados:** grau=2, alpha Ridge=1, gate_strength=0.1

| Teste | N | MAE | RMSE | R² | Bias | NMAD | Outliers catastróficos | Predições negativas |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B | 74559 | 0.0240 | 0.0351 | 0.8400 | -0.0001 | 0.0188 | 0.14% | 0 |
| C | 97980 | 0.0253 | 0.0377 | 0.8562 | -0.0005 | 0.0193 | 0.24% | 0 |
| D | 75925 | 0.0913 | 0.3574 | -1.7009 | 0.0697 | 0.0567 | 8.56% | 667 |

### Teste B

- **Experimento:** `polynomial_ridge_gate_err_manual_degree2_alpha1_gatestrength0.1`
- **Parâmetros:** grau=2, alpha Ridge=1, gate_strength=0.1

| Métrica | Valor |
|---|---:|
| N | 74559 |
| MAE | 0.024008 |
| RMSE | 0.035113 |
| R² | 0.840022 |
| Bias | -0.000065 |
| NMAD | 0.018810 |
| Erro-padrão do NMAD | 0.000083 |
| Fração de outliers catastróficos | 0.14% |
| Limiar de outlier catastrófico | 0.15 |
| Predições negativas de redshift | 0 |

### Teste C

- **Experimento:** `polynomial_ridge_gate_err_manual_degree2_alpha1_gatestrength0.1`
- **Parâmetros:** grau=2, alpha Ridge=1, gate_strength=0.1

| Métrica | Valor |
|---|---:|
| N | 97980 |
| MAE | 0.025316 |
| RMSE | 0.037749 |
| R² | 0.856238 |
| Bias | -0.000505 |
| NMAD | 0.019279 |
| Erro-padrão do NMAD | 0.000080 |
| Fração de outliers catastróficos | 0.24% |
| Limiar de outlier catastrófico | 0.15 |
| Predições negativas de redshift | 0 |

### Teste D

- **Experimento:** `polynomial_ridge_gate_err_manual_degree2_alpha1_gatestrength0.1`
- **Parâmetros:** grau=2, alpha Ridge=1, gate_strength=0.1

| Métrica | Valor |
|---|---:|
| N | 75925 |
| MAE | 0.091305 |
| RMSE | 0.357379 |
| R² | -1.700949 |
| Bias | 0.069685 |
| NMAD | 0.056695 |
| Erro-padrão do NMAD | 0.000268 |
| Fração de outliers catastróficos | 8.56% |
| Limiar de outlier catastrófico | 0.15 |
| Predições negativas de redshift | 667 |
