# Análisis de Modelos — Mortalidad Prematura por ENT (NCDMORT3070)

> Variable objetivo: probabilidad (%) de morir entre los 30 y 70 años por una enfermedad no transmisible (cardiovascular, cáncer, diabetes, respiratoria crónica).
>
> Datos: WHO + World Bank, 185 países, 2000–2021.
> Split temporal: **entrenamiento 2000–2020 (3,885 filas)** | **prueba 2021 (185 filas, un país = una fila)**.

---

## 1. Contexto del dataset

| Métrica | Valor |
|---|---|
| Observaciones modelables | 4,070 |
| Países | 185 |
| Rango temporal | 2000 – 2021 |
| Target (media global) | 21.15% |
| Target (desviación estándar) | 7.46 pp |
| Target (mínimo / máximo) | 6.9% / 45.3% |

El rango del target va de casi 7% (países con sistemas de salud muy desarrollados como Japón, Suiza) hasta 45% (países subsaharianos con alta carga de enfermedades no transmisibles y bajo acceso a atención médica). Una dispersión de 38 puntos porcentuales indica que hay espacio real para que un modelo capture patrones, pero también que el mundo es muy heterogéneo.

**Nota sobre el split**: el dataset disponible cubre hasta 2021. Con corte en ese año, el conjunto de prueba queda con 185 observaciones — una por país — todas del mismo año. Esto mide qué tan bien el modelo generaliza a un nuevo período en todos los países simultáneamente, que es exactamente la pregunta de investigación (¿funciona en datos no vistos?).

---

## 2. Resultados de los modelos

### 2.1 Tabla de métricas

| Modelo | RMSE Train | R² Train | RMSE Test | MAE Test | R² Test |
|---|---|---|---|---|---|
| **Ridge Regression** | 5.17 pp | 0.519 | 5.26 pp | 4.13 pp | 0.463 |
| **Random Forest** | **0.66 pp** | **0.992** | **3.46 pp** | **2.58 pp** | **0.768** |
| **XGBoost** | 1.95 pp | 0.932 | 4.12 pp | 3.17 pp | 0.670 |

Las unidades son **puntos porcentuales** del riesgo de morir. Por ejemplo, un RMSE de 3.46 pp significa que el modelo se equivoca en promedio unas 3.5 unidades porcentuales en su predicción del riesgo nacional.

### 2.2 Explicación de las métricas

**RMSE (Root Mean Squared Error)**: penaliza errores grandes más que pequeños. Mejor cuando se necesita evitar predicciones muy desviadas. Unidades: puntos porcentuales del target.

**MAE (Mean Absolute Error)**: error promedio sin penalizar outliers. Más interpretable en el dominio: "el modelo se equivoca en promedio X pp". Un MAE de 2.58 pp (Random Forest) significa que si un país tiene 20% de riesgo, el modelo predice entre 17.4% y 22.6% la mayoría de las veces.

**R² (coeficiente de determinación)**: fracción de la varianza del target explicada por el modelo. Un R² de 0.77 significa que el modelo captura el 77% de las diferencias entre países. R²=1 sería perfección, R²=0 equivale a predecir siempre la media global.

---

## 3. Análisis modelo por modelo

### 3.1 Ridge Regression — El honesto que no exagera

Ridge es una regresión lineal con regularización L2. Asume que la relación entre predictores y mortalidad es **aditiva y lineal**: cada punto extra de esperanza de vida baja la mortalidad en una cantidad fija, sin importar el contexto.

**Resultados**: Train R²=0.52, Test R²=0.46.

El gap entre train y test es mínimo (0.06), lo que indica que **no hay overfitting**. El modelo es estable, lo que da confianza en que lo que aprendió en 2000–2020 no es ruido específico de ese período. Sin embargo, solo explica el 46% de la varianza en 2021 — prácticamente la mitad de la variación entre países queda sin explicar.

**¿Por qué falla?** La mortalidad por ENT no es una combinación lineal simple de los predictores. Un país puede tener alto PIB pero también alta prevalencia de tabaquismo (Europa del Este), o bajo gasto en salud pero muy buena salud preventiva (Cuba). Esas interacciones y no-linealidades son invisibles para Ridge.

**Bias de +2.16 pp y solo 29.2% de predicciones dentro de ±2 pp**: el modelo tiende a sobrestimar el riesgo. Esto tiene sentido porque Ridge regresa hacia la media — para los países con mortalidad muy baja (outliers positivos del sistema de salud), el modelo los arrastra hacia el promedio global.

**Veredicto**: útil como baseline estadístico y para entender qué coeficientes son relevantes, pero insuficiente para predicción seria.

---

### 3.2 Random Forest — El mejor predictor, con trampa

Random Forest construye 300 árboles de decisión, cada uno entrenado en una muestra aleatoria de datos y variables. La predicción final es el promedio de todos los árboles.

**Resultados**: Train R²=0.992, Test R²=0.768.

Aquí hay una señal de alerta enorme: **R² de entrenamiento de 0.992**. Los árboles aprendieron casi perfectamente los datos de entrenamiento — esto se llama **overfitting**. El modelo memorizó los patrones específicos de 2000–2020 hasta el último decimal.

Sin embargo, **en el conjunto de prueba sigue siendo el mejor** (RMSE=3.46, MAE=2.58, R²=0.77). Eso significa que el overfitting existe pero los patrones generalizables que captó son reales y útiles.

**Bias de +1.53 pp y 52.4% dentro de ±2 pp**: el mejor de los tres. Más de la mitad de las predicciones caen a menos de 2 pp del valor real.

**¿Por qué gana?** Random Forest puede capturar interacciones no lineales complejas: PIB alto + baja cobertura médica genera un patrón diferente a PIB alto + alta cobertura. Esas interacciones son invisibles para Ridge pero transparentes para los árboles.

**Riesgo**: si los datos de 2022+ muestran tendencias nuevas (post-pandemia, shocks económicos), el Random Forest podría degradarse más que XGBoost porque sus árboles están más ajustados al período histórico.

**Veredicto**: el mejor modelo en la métrica que importa (predicción en datos no vistos de 2021), pero hay que monitorear su degradación si el mundo cambia rápido.

---

### 3.3 XGBoost — El equilibrista

XGBoost es gradient boosting: construye árboles en secuencia, donde cada árbol corrige los errores del anterior. La tasa de aprendizaje baja (0.05) y la regularización (subsample=0.8) lo frenan para no memorizar.

**Resultados**: Train R²=0.932, Test R²=0.670.

El gap train-test es el mayor de los tres (0.26 en R²), lo que sugiere que, a pesar de la regularización, también está sobreajustando — aunque menos drásticamente que Random Forest en entrenamiento.

**¿Por qué queda tercero?** Con solo 185 países y un único año de prueba, el ajuste fino que hace XGBoost (aprender residuales iterativamente) captura patrones históricos muy específicos que no se transfieren igual de bien. Random Forest, al promediar árboles independientes, tiene naturalmente más varianza reducida en sus predicciones.

**Bias de +2.15 pp y 43.2% dentro de ±2 pp**: en el medio, mejor que Ridge pero peor que Random Forest.

**Veredicto**: el más robusto en teoría para extrapolación temporal, pero en este dataset específico queda superado por Random Forest. En un escenario con datos de 2022–2024, podría remontar.

---

## 4. Análisis de la gráfica: Predicciones vs Valores Reales

### ¿Qué muestra esta gráfica?

Cada punto es un país en 2021. El eje X es el valor real de NCDMORT3070; el eje Y es lo que predijo el modelo. La línea diagonal punteada representa predicción perfecta (real = predicho).

- **Puntos sobre la diagonal**: el modelo sobrestimó el riesgo (predijo más mortalidad de la que ocurrió).
- **Puntos bajo la diagonal**: el modelo subestimó.
- **Nube apretada alrededor de la diagonal**: buen modelo.

**Patrón claro en todos los modelos**: la nube se abre más en los países con mortalidad baja (eje X: 7–15%). El modelo predice valores en el rango 18–28% para países que en realidad están en 10–15%. Esto queda visible en los mayores errores:

| País | Real (%) | Predicho XGB (%) | Error (pp) |
|---|---|---|---|
| El Salvador | 12.6 | 26.4 | +13.8 |
| Nicaragua | 12.6 | 23.4 | +10.8 |
| Paraguay | 16.1 | 26.5 | +10.4 |
| Colombia | 10.3 | 20.3 | +10.0 |
| Líbano | 11.9 | 21.6 | +9.7 |
| Jordania | 11.6 | 20.8 | +9.2 |
| México | 16.3 | 25.3 | +9.0 |

**Interpretación**: estos países tienen indicadores socioeconómicos intermedios — PIB moderado, cobertura de salud razonable — que históricamente se asociaban con mortalidades más altas. Pero han logrado **reducciones más rápidas de lo que predicen sus variables estructurales**. El modelo no sabe de reformas de salud pública, campañas de prevención o cambios culturales que no están capturados en los indicadores cuantitativos.

El único error inverso notable: Eswatini (real 32.3%, predicho 23.4%), donde el modelo subestimó — una nación con indicadores aceptables en papel pero con problemas estructurales profundos.

---

## 5. Análisis de la gráfica: Importancia de Variables

### Qué mide cada modelo

- **Ridge**: magnitud del coeficiente (valor absoluto). Refleja cuánto cambia el target por cada desviación estándar del predictor.
- **Random Forest / XGBoost**: reducción de impureza promedio. Qué fracción de la varianza del target se explica al dividir los nodos por esa variable.

### Hallazgos consistentes entre los tres modelos

| Rango | Variable | Interpretación |
|---|---|---|
| 1° (siempre) | **Esperanza de Vida** | Proxy del nivel global de salud de un país. Alta correlación porque ambas métricas miden longevidad y calidad de vida. |
| 2°–3° | **PIB per cápita (log)** | Desarrollo económico general — capacidad de financiar sistemas de salud, condiciones de vida, acceso. |
| 3°–4° | **Mortalidad Infantil** | Indicador de condiciones de vida tempranas y resiliencia del sistema de salud. |
| Medio | **Médicos / 10k hab.** | Acceso a atención médica. Más relevante en Ridge (coeficiente lineal claro). |
| Bajo | **Gasto en Salud (% PIB)** | Sorprendentemente débil en todos los modelos. |
| Último | **Camas Hospitalarias** | El predictor menos informativo consistentemente. |

### El hallazgo más interesante: Gasto en Salud no importa

El gasto en salud como porcentaje del PIB aparece en posiciones 6–8 en todos los modelos. Esto contradice la intuición inicial de que "más gasto = menos mortalidad".

**¿Por qué?** Porque el gasto no captura la **eficiencia**. EE.UU. gasta ~17% del PIB en salud y tiene una mortalidad por ENT relativamente alta. Cuba gasta mucho menos pero tiene sistemas preventivos muy efectivos. El dinero per se no compra salud; lo que importa es cómo se estructura el sistema. Esta variable mide cantidad, no calidad.

**Implicación para política pública**: los modelos sugieren que priorizar esperanza de vida (como indicador proxy de condiciones estructurales) y desarrollo económico general tiene más poder predictivo que simplemente aumentar el presupuesto de salud.

---

## 6. Análisis de la gráfica: Residuales

### ¿Qué muestra esta gráfica?

Eje X: valor predicho por el modelo. Eje Y: error (predicho – real). La línea en y=0 es el ideal.

- **Puntos arriba de y=0**: el modelo sobreestimó.
- **Nube sin patrón** alrededor de y=0: errores aleatorios (modelo bien calibrado).
- **Forma de embudo (mayor dispersión a la derecha)**: el modelo comete errores más grandes para ciertos rangos de predicción.

**Patrón observado**: todos los modelos tienen residuales positivos dominantes en el rango de predicciones bajas (10–20%). El modelo sistemáticamente sobreestima cuando el target real es bajo. Esto es un **sesgo estructural** que ningún modelo logra eliminar completamente.

**Causa**: los países con mortalidad baja en 2021 llegaron ahí por trayectorias históricas complejas que el modelo no puede inferir solo con variables de punto en el tiempo o tendencias suavizadas.

---

## 7. Gráfica de radar: Comparativa normalizada

El radar muestra las tres métricas normalizadas (0–1, mayor es mejor en cada eje):
- **RMSE↓**: normalizado inversamente — menor RMSE da puntaje más alto.
- **MAE↓**: igual.
- **R²↑**: mayor es mejor directamente.

Random Forest domina los tres vértices del triángulo. XGBoost queda en posición intermedia. Ridge cubre la menor área en el triángulo, lo que confirma visualmente que es el modelo más débil en las tres métricas simultáneamente.

---

## 8. Veredicto final

### ¿Cuál modelo usar?

**Random Forest es el ganador práctico** en este dataset. Métricas de prueba: RMSE=3.46 pp, MAE=2.58 pp, R²=0.77. Más del 52% de sus predicciones caen a menos de 2 pp del valor real — sobre una escala de 40 pp, eso es bastante preciso.

Sin embargo, **la elección depende del objetivo**:

| Objetivo | Modelo recomendado | Razón |
|---|---|---|
| Mejor predicción en 2021 | Random Forest | Gana en todas las métricas de test |
| Robustez en datos futuros (2022+) | XGBoost | Menor overfitting relativo, más regularizado |
| Interpretabilidad de coeficientes | Ridge Regression | Coeficientes lineales directamente interpretables |
| Publicación académica que exige transparencia | Ridge + RF como comparación | El más honesto ante revisores |

### Lo más interesante del análisis

**El patrón de error no es aleatorio**: los países que el modelo más sobreestima son de América Latina y Medio Oriente — economías en transición con mortalidades por ENT que cayeron más rápido de lo que sus variables estructurales predecirían. Esto no es un fallo del modelo; es una señal de que **algo externo a los predictores disponibles** está impulsando la mejora (reformas específicas, transición epidemiológica acelerada, cambios en estilos de vida). Si hubiera datos de políticas públicas de salud preventiva o tasas de tabaquismo, el modelo mejoraría sustancialmente.

**Esperanza de Vida como predictor estelar** es consistente pero también preocupante: ambas variables (esperanza de vida y mortalidad por ENT) miden aspectos del mismo fenómeno de salud poblacional. No son independientes conceptualmente. En un modelo explicativo puro sería problemático (casi circularidad); para predicción pura es válido y útil.

---

## 9. Cómo reproducir este análisis

```bash
# 1. Generar el dataset procesado (si no existe)
.venv/Scripts/python src/eda.py

# 2. Abrir el dashboard interactivo
.venv/Scripts/python -m streamlit run src/app.py
```

En el dashboard:
- Ajustar el slider de corte al año deseado (2021 por defecto)
- Activar los tres modelos
- Navegar por las tabs: Métricas → Predicciones → Importancia de Variables
