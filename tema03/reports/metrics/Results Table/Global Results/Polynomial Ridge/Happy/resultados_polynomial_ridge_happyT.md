# Resultados dos baselines - happyT - Polynomial Ridge

Resultados dos baselines `mag`, `mag_err` e `gate-err` no dataset `happyT`, avaliados separadamente nos testes externos `B`, `C` e `D`.

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
| B | `mag` | 0.0429 | 0.0703 | 0.8892 | -0.0001 | 0.0321 | 1.57% | 805 | 74900 |
| B | `mag_err` | **0.0378** | **0.0660** | **0.9025** | **0.0000** | **0.0269** | **1.31%** | **267** | 74900 |
| B | `gate-err` | 0.0430 | 0.0703 | 0.8894 | -0.0001 | 0.0321 | 1.56% | 862 | 74900 |
|  |  |  |  |  |  |  |  |  |  |
| C | `mag` | 0.0975 | 0.1311 | 0.4862 | 0.0422 | 0.0732 | 10.97% | 202 | 60315 |
| C | `mag_err` | **0.0842** | **0.1206** | **0.5652** | **0.0209** | **0.0636** | **7.62%** | **187** | 60315 |
| C | `gate-err` | 0.0979 | 0.1315 | 0.4828 | 0.0441 | 0.0730 | 11.11% | 188 | 60315 |
|  |  |  |  |  |  |  |  |  |  |
| D | `mag` | 0.1328 | **0.1880** | **0.2741** | 0.0254 | 0.0989 | 18.08% | 447 | 74642 |
| D | `mag_err` | **0.1252** | 0.2157 | 0.0451 | **-0.0010** | **0.0905** | **15.18%** | **301** | 74642 |
| D | `gate-err` | 0.1328 | 0.1886 | 0.2699 | 0.0224 | 0.0992 | 18.07% | 418 | 74642 |

## Resultados por baseline e teste

## Baseline 1 - Polynomial Ridge com magnitudes (`mag`)

- **Feature set:** `mag`
- **Features finais:** `u`, `g`, `r`, `i`, `z`
- **Parâmetros selecionados:** grau=2, alpha Ridge=10

| Teste | N | MAE | RMSE | R² | Bias | NMAD | Outliers catastróficos | Predições negativas |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B | 74900 | 0.0429 | 0.0703 | 0.8892 | -0.0001 | 0.0321 | 1.57% | 805 |
| C | 60315 | 0.0975 | 0.1311 | 0.4862 | 0.0422 | 0.0732 | 10.97% | 202 |
| D | 74642 | 0.1328 | 0.1880 | 0.2741 | 0.0254 | 0.0989 | 18.08% | 447 |

### Teste B

- **Experimento:** `polynomial_ridge_mag_degree2_alpha10`
- **Parâmetros:** grau=2, alpha Ridge=10

| Métrica | Valor |
|---|---:|
| N | 74900 |
| MAE | 0.042944 |
| RMSE | 0.070340 |
| R² | 0.889212 |
| Bias | -0.000118 |
| NMAD | 0.032135 |
| Erro-padrão do NMAD | 0.000143 |
| Fração de outliers catastróficos | 1.57% |
| Limiar de outlier catastrófico | 0.15 |
| Predições negativas de redshift | 805 |

### Teste C

- **Experimento:** `polynomial_ridge_mag_degree2_alpha10`
- **Parâmetros:** grau=2, alpha Ridge=10

| Métrica | Valor |
|---|---:|
| N | 60315 |
| MAE | 0.097463 |
| RMSE | 0.131120 |
| R² | 0.486171 |
| Bias | 0.042194 |
| NMAD | 0.073189 |
| Erro-padrão do NMAD | 0.000367 |
| Fração de outliers catastróficos | 10.97% |
| Limiar de outlier catastrófico | 0.15 |
| Predições negativas de redshift | 202 |

### Teste D

- **Experimento:** `polynomial_ridge_mag_degree2_alpha10`
- **Parâmetros:** grau=2, alpha Ridge=10

| Métrica | Valor |
|---|---:|
| N | 74642 |
| MAE | 0.132776 |
| RMSE | 0.188046 |
| R² | 0.274136 |
| Bias | 0.025431 |
| NMAD | 0.098914 |
| Erro-padrão do NMAD | 0.000457 |
| Fração de outliers catastróficos | 18.08% |
| Limiar de outlier catastrófico | 0.15 |
| Predições negativas de redshift | 447 |

## Baseline 2 - Polynomial Ridge com magnitudes + erros fotométricos (`mag_err`)

- **Feature set:** `mag_err`
- **Features finais:** `u`, `g`, `r`, `i`, `z`, `uErr`, `gErr`, `rErr`, `iErr`, `zErr`
- **Parâmetros selecionados:** grau=2, alpha Ridge=10

| Teste | N | MAE | RMSE | R² | Bias | NMAD | Outliers catastróficos | Predições negativas |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B | 74900 | 0.0378 | 0.0660 | 0.9025 | 0.0000 | 0.0269 | 1.31% | 267 |
| C | 60315 | 0.0842 | 0.1206 | 0.5652 | 0.0209 | 0.0636 | 7.62% | 187 |
| D | 74642 | 0.1252 | 0.2157 | 0.0451 | -0.0010 | 0.0905 | 15.18% | 301 |

### Teste B

- **Experimento:** `polynomial_ridge_mag_err_degree2_alpha10`
- **Parâmetros:** grau=2, alpha Ridge=10

| Métrica | Valor |
|---|---:|
| N | 74900 |
| MAE | 0.037758 |
| RMSE | 0.065973 |
| R² | 0.902543 |
| Bias | 0.000049 |
| NMAD | 0.026896 |
| Erro-padrão do NMAD | 0.000118 |
| Fração de outliers catastróficos | 1.31% |
| Limiar de outlier catastrófico | 0.15 |
| Predições negativas de redshift | 267 |

### Teste C

- **Experimento:** `polynomial_ridge_mag_err_degree2_alpha10`
- **Parâmetros:** grau=2, alpha Ridge=10

| Métrica | Valor |
|---|---:|
| N | 60315 |
| MAE | 0.084209 |
| RMSE | 0.120613 |
| R² | 0.565221 |
| Bias | 0.020909 |
| NMAD | 0.063603 |
| Erro-padrão do NMAD | 0.000329 |
| Fração de outliers catastróficos | 7.62% |
| Limiar de outlier catastrófico | 0.15 |
| Predições negativas de redshift | 187 |

### Teste D

- **Experimento:** `polynomial_ridge_mag_err_degree2_alpha10`
- **Parâmetros:** grau=2, alpha Ridge=10

| Métrica | Valor |
|---|---:|
| N | 74642 |
| MAE | 0.125183 |
| RMSE | 0.215681 |
| R² | 0.045117 |
| Bias | -0.001023 |
| NMAD | 0.090485 |
| Erro-padrão do NMAD | 0.000437 |
| Fração de outliers catastróficos | 15.18% |
| Limiar de outlier catastrófico | 0.15 |
| Predições negativas de redshift | 301 |

## Baseline 3 - Polynomial Ridge com magnitudes ponderadas pelos erros (`gate-err`)

- **Feature set:** `gate_err_manual`
- **Features finais:** `u_gate`, `g_gate`, `r_gate`, `i_gate`, `z_gate`
- **Parâmetros selecionados:** grau=2, alpha Ridge=10, gate_strength=0.1

| Teste | N | MAE | RMSE | R² | Bias | NMAD | Outliers catastróficos | Predições negativas |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B | 74900 | 0.0430 | 0.0703 | 0.8894 | -0.0001 | 0.0321 | 1.56% | 862 |
| C | 60315 | 0.0979 | 0.1315 | 0.4828 | 0.0441 | 0.0730 | 11.11% | 188 |
| D | 74642 | 0.1328 | 0.1886 | 0.2699 | 0.0224 | 0.0992 | 18.07% | 418 |

### Teste B

- **Experimento:** `polynomial_ridge_gate_err_manual_degree2_alpha10_gatestrength0.1`
- **Parâmetros:** grau=2, alpha Ridge=10, gate_strength=0.1

| Métrica | Valor |
|---|---:|
| N | 74900 |
| MAE | 0.043023 |
| RMSE | 0.070281 |
| R² | 0.889400 |
| Bias | -0.000088 |
| NMAD | 0.032061 |
| Erro-padrão do NMAD | 0.000156 |
| Fração de outliers catastróficos | 1.56% |
| Limiar de outlier catastrófico | 0.15 |
| Predições negativas de redshift | 862 |

### Teste C

- **Experimento:** `polynomial_ridge_gate_err_manual_degree2_alpha10_gatestrength0.1`
- **Parâmetros:** grau=2, alpha Ridge=10, gate_strength=0.1

| Métrica | Valor |
|---|---:|
| N | 60315 |
| MAE | 0.097932 |
| RMSE | 0.131545 |
| R² | 0.482837 |
| Bias | 0.044066 |
| NMAD | 0.072952 |
| Erro-padrão do NMAD | 0.000378 |
| Fração de outliers catastróficos | 11.11% |
| Limiar de outlier catastrófico | 0.15 |
| Predições negativas de redshift | 188 |

### Teste D

- **Experimento:** `polynomial_ridge_gate_err_manual_degree2_alpha10_gatestrength0.1`
- **Parâmetros:** grau=2, alpha Ridge=10, gate_strength=0.1

| Métrica | Valor |
|---|---:|
| N | 74642 |
| MAE | 0.132824 |
| RMSE | 0.188588 |
| R² | 0.269946 |
| Bias | 0.022368 |
| NMAD | 0.099175 |
| Erro-padrão do NMAD | 0.000454 |
| Fração de outliers catastróficos | 18.07% |
| Limiar de outlier catastrófico | 0.15 |
| Predições negativas de redshift | 418 |
