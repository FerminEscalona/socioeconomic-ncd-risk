# Monitor Global de Salud y Resiliencia Sanitaria

### Predicción de la mortalidad prematura por enfermedades no transmisibles a partir de indicadores socioeconómicos y de infraestructura sanitaria

---

**Asignatura:** Herramientas de Big Data
**Programa:** Maestría en Analítica Aplicada (Asignatura Coterminal)
**Institución:** Universidad de la Sabana — Facultad de Ingeniería
**Profesor:** Hugo Franco, Ph.D.
**Fecha de entrega:** 3 de mayo de 2026

**Autores:**

| Nombre | Código | Carrera |
|---|---|---|
| Fermín Escalona | 300181 | Ingeniería Informática |
| Julián Romero | 325312 | Ingeniería Informática |
| Samuel Ramírez | 296748 | Ingeniería Industrial |

---

## 2. Resumen ejecutivo

Las enfermedades no transmisibles (ENT) — cardiovasculares, cáncer, diabetes y respiratorias crónicas — concentran la mayor parte de la mortalidad prematura en el mundo y su distribución no es homogénea entre países. Este proyecto construye una plataforma analítica reproducible que integra datos abiertos de la Organización Mundial de la Salud (OMS) y del Banco Mundial, los consolida en un panel histórico país–año, los carga en un esquema dimensional sobre PostgreSQL y los explota desde tres frentes: análisis exploratorio en Python, procesamiento masivo con Polars y un dashboard interactivo en Streamlit.

Sobre 4.070 observaciones modelables (185 países, 2000–2021) se compararon tres modelos — Ridge, Random Forest y XGBoost — para predecir el indicador `NCDMORT3070` (probabilidad de morir entre los 30 y 70 años por una ENT). Random Forest obtuvo el mejor desempeño en el conjunto de prueba (RMSE = 3,46 pp; MAE = 2,58 pp; R² = 0,768) frente a Ridge (R² = 0,463) y XGBoost (R² = 0,670). La esperanza de vida, el PIB per cápita y la mortalidad infantil emergieron como los predictores más informativos; el gasto en salud como porcentaje del PIB resultó sorprendentemente débil, sugiriendo que la cantidad de gasto importa menos que su asignación.

El proyecto entrega no solo un modelo predictivo, sino una arquitectura ETL → almacén dimensional → analítica → dashboard que cualquier evaluador puede reproducir desde cero ejecutando una secuencia documentada de comandos.

## 3. Abstract

Non-communicable diseases (NCDs) — cardiovascular, cancer, diabetes and chronic respiratory conditions — account for most premature mortality worldwide, with sharp inequalities across countries. This project builds a reproducible analytical platform that ingests open data from the World Health Organization (WHO) and the World Bank, consolidates them into a country–year historical panel, loads them into a dimensional schema on PostgreSQL, and exploits the resulting view through three layers: exploratory analysis in Python, in-memory distributed analysis with Polars, and an interactive Streamlit dashboard.

Over 4,070 modelable observations (185 countries, 2000–2021), three models — Ridge, Random Forest and XGBoost — were compared to predict the WHO indicator `NCDMORT3070` (probability of dying between ages 30 and 70 from an NCD). Random Forest produced the best test performance (RMSE = 3.46 pp; MAE = 2.58 pp; R² = 0.768), outperforming Ridge (R² = 0.463) and XGBoost (R² = 0.670). Life expectancy, GDP per capita and infant mortality emerged as the most informative predictors, while health expenditure as a share of GDP was surprisingly weak — suggesting that *how* health spending is structured matters more than *how much* is spent. The deliverable is not only a model but a fully documented ETL → dimensional warehouse → analytics → dashboard pipeline that any evaluator can reproduce end-to-end from documented commands.

**Keywords:** non-communicable diseases, premature mortality, social determinants of health, machine learning, Random Forest, XGBoost, Ridge regression, PostgreSQL, Polars, Streamlit, big data.

---

## 4. Introducción

### 4.a Formulación del problema y necesidades de información

Las ENT son responsables de aproximadamente **74 % de las muertes a nivel global** y de cerca de **17 millones de muertes prematuras** (antes de los 70 años) cada año, con una concentración desproporcionada en países de ingresos bajos y medios (World Health Organization, 2022). La probabilidad incondicional de morir entre los 30 y 70 años por alguna de las cuatro principales ENT — codificada por la OMS en el indicador `NCDMORT3070` — es uno de los principales objetivos de seguimiento de los Objetivos de Desarrollo Sostenible (Meta 3.4), que aspira a reducir esta probabilidad en un tercio para 2030 (United Nations, 2015).

A pesar de su relevancia, no existe una respuesta consolidada y operativa a una pregunta esencialmente práctica: **¿cuánto de la variación internacional en mortalidad prematura por ENT puede explicarse a partir de indicadores socioeconómicos y de infraestructura sanitaria disponibles públicamente, y hasta qué punto esos mismos indicadores pueden anticipar el riesgo en un período no observado?** Responder a esta pregunta requiere:

1. **Información histórica de salud pública**: tasas de mortalidad por ENT, esperanza de vida, mortalidad infantil, incidencia de enfermedades transmisibles y disponibilidad de infraestructura clínica (camas, médicos).
2. **Información socioeconómica comparable entre países**: PIB per cápita, gasto en salud como porcentaje del PIB y acceso a servicios básicos (agua potable).
3. **Una estructura común país–año** que permita unir indicadores de diferentes fuentes sin perder integridad referencial.
4. **Un entorno reproducible** que permita reentrenar y reevaluar los modelos cuando se actualicen las fuentes.

### 4.b Marco conceptual

**Enfermedades no transmisibles (ENT).** Conjunto de enfermedades no infecciosas y de larga duración que incluye, en la definición operativa de la OMS, las cardiovasculares, los cánceres, la diabetes y las respiratorias crónicas (World Health Organization, 2022). El indicador `NCDMORT3070` mide la probabilidad de morir entre los 30 y 70 años por alguna de estas cuatro causas, asumiendo las tasas específicas por edad y causa observadas durante un año dado.

**Determinantes sociales de la salud.** Conjunto de condiciones en las que las personas nacen, crecen, viven, trabajan y envejecen. La Comisión sobre Determinantes Sociales de la Salud de la OMS estableció que estos determinantes — ingreso, educación, acceso a agua y saneamiento, gasto público en salud — explican gran parte de las inequidades en salud entre y dentro de los países (Commission on Social Determinants of Health, 2008; Marmot, 2005).

**Panel longitudinal país–año.** Estructura de datos en la que cada observación está identificada por la pareja (país, año), lo que permite combinar variabilidad transversal (entre países) y temporal (dentro de un país a lo largo del tiempo).

**Esquema dimensional.** Modelo de datos compuesto por una tabla de hechos numéricos rodeada de tablas de dimensiones (Kimball & Ross, 2013). En este proyecto, la tabla de hechos `analytics.fact_indicator_value` se rodea de tres dimensiones — país, año e indicador — y se materializa una vista wide `analytics.v_model_panel` para consumo analítico.

**Imputación de datos faltantes en series país–año.** Estrategia jerárquica que prioriza la coherencia temporal por país (interpolación) sobre la coherencia transversal (mediana anual o global). Es preferida frente a la imputación por la media en presencia de outliers (Little & Rubin, 2019).

**Validación temporal (out-of-time).** Estrategia de evaluación en la que el conjunto de prueba contiene observaciones de un período posterior al del entrenamiento. Mide la capacidad real de generalización futura de un modelo, a diferencia de la validación cruzada estándar que mezcla períodos (Hyndman & Athanasopoulos, 2021).

**Modelos comparados.**
- **Ridge Regression** — regresión lineal con regularización L2 que penaliza coeficientes grandes para reducir varianza (Hoerl & Kennard, 1970).
- **Random Forest** — ensamble de árboles de decisión entrenados sobre muestras bootstrap y subconjuntos aleatorios de variables, cuya predicción es el promedio de los árboles individuales (Breiman, 2001).
- **XGBoost** — gradient boosting con regularización: construye árboles de forma secuencial, donde cada árbol corrige residuales del anterior (Chen & Guestrin, 2016).

### 4.c Antecedentes

La relación entre desarrollo socioeconómico y carga de enfermedad no es nueva. Preston (1975) documentó la relación curvilínea entre el ingreso per cápita y la esperanza de vida, mostrando rendimientos decrecientes del ingreso sobre la longevidad — la llamada "curva de Preston". Esta tradición se ha actualizado periódicamente: Deaton (2013) muestra que aunque el crecimiento económico es condición necesaria, no es suficiente para reducir la mortalidad sin políticas públicas explícitas de salud.

El estudio Global Burden of Disease (GBD) constituye el esfuerzo más amplio de cuantificación sistemática de la carga de enfermedad por país y factor de riesgo, e identifica los factores conductuales (tabaco, dieta, alcohol, actividad física) y metabólicos (presión arterial, glucemia, IMC) como los principales contribuyentes a la mortalidad por ENT (GBD 2019 Risk Factors Collaborators, 2020). Bloom et al. (2011) cuantifican el impacto económico acumulado de las ENT en 47 billones de dólares entre 2010 y 2030, arguyendo que la inversión preventiva tiene retornos altos pero subutilizados.

En cuanto a métodos predictivos, la literatura reciente ha aplicado *machine learning* a la mortalidad y a la carga de enfermedad. Se ha mostrado que los modelos basados en árboles (Random Forest, gradient boosting) suelen superar a regresiones lineales en problemas con interacciones no lineales y heterogeneidad entre subpoblaciones (Hastie, Tibshirani, & Friedman, 2009). En el dominio específico de salud poblacional, modelos como XGBoost se han empleado para estimar mortalidad infantil (Bitew et al., 2020) y predicción de riesgo cardiovascular individual a partir de electrocardiogramas y registros clínicos (Weng et al., 2017). Sin embargo, los enfoques que parten de **panel país–año a partir de indicadores macro** son menos frecuentes en la literatura revisada y suelen quedar restringidos a regresiones econométricas tradicionales con efectos fijos.

Como métodos posibles de solución se consideraron: regresión lineal/Ridge como baseline interpretable; modelos de árboles ensamblados para capturar no linealidades; redes neuronales tabulares (descartadas por el tamaño moderado del dataset y la pérdida de interpretabilidad); y modelos de panel con efectos fijos (descartados porque colapsarían el aprendizaje país–específico, contrario al objetivo de generalización a partir de variables estructurales). La combinación finalmente seleccionada (Ridge + Random Forest + XGBoost) es la práctica estándar recomendada por James et al. (2021) cuando se busca contrastar simplicidad interpretable con capacidad predictiva.

### 4.d Objetivo del proyecto

**Objetivo general.** Construir una plataforma reproducible de extracción, integración, almacenamiento y modelado de datos públicos de salud y socioeconomía que permita estimar la probabilidad de mortalidad prematura por ENT en un país a partir de sus indicadores estructurales.

**Objetivos específicos.**
1. Diseñar y operar un pipeline ETL que extraiga indicadores desde la WHO GHO API y la World Bank Open Data API, los normalice a un esquema común *long* y los persista en disco como insumos versionables.
2. Limpiar, depurar e imputar el panel país–año aplicando reglas explícitas (ISO3, rango temporal, dedup por dimensión sexo, no imputación del target) y producir un dataset modelable *wide*.
3. Cargar el dataset en PostgreSQL con un esquema dimensional y exponer una vista `analytics.v_model_panel` que aísle a los consumidores del esquema físico.
4. Comparar tres modelos de regresión (Ridge, Random Forest, XGBoost) con un *split* temporal estricto y reportar métricas RMSE, MAE y R² sobre datos no vistos.
5. Entregar un dashboard interactivo en Streamlit que permita a cualquier usuario explorar predicciones, residuales e importancia de variables sin necesidad de modificar código.

---

## 5. Datos empleados

### 5.1 Fuentes de datos

Se utilizaron exclusivamente fuentes abiertas, sin credenciales, de organismos internacionales de referencia:

- **OMS — Global Health Observatory (GHO) API** (`https://ghoapi.azureedge.net/api`). Proporciona indicadores globales de salud organizados por código de indicador, país, año y dimensiones complementarias (sexo, grupo de edad). La extracción se realiza mediante el módulo [src/extractors/who_extractor.py](src/extractors/who_extractor.py).
- **Banco Mundial — World Bank Open Data API** (`https://api.worldbank.org/v2`). Proporciona indicadores socioeconómicos y de desarrollo en estructura país–año. La extracción se realiza mediante el módulo [src/extractors/world_bank_extractor.py](src/extractors/world_bank_extractor.py).

El catálogo completo de indicadores descargados se centraliza en [src/config/indicators.py](src/config/indicators.py), donde cada indicador se registra con su nombre descriptivo y su rol conceptual (`target`, `descriptive`, `predictive_feature`, `health_feature`, `health_social_feature`, `socioeconomic_feature`).

### 5.2 Variable dependiente

**`NCDMORT3070` — Probabilidad de morir entre los 30 y 70 años por una ENT** (fuente: OMS). Expresada en porcentaje (rango observado 6,9 % – 45,3 %; media global 21,15 %; desviación estándar 7,46 pp). Esta es la única variable que **no se imputa en ningún caso**, para no introducir señal sintética en el target.

### 5.3 Variables independientes

La Tabla 1 resume las variables independientes utilizadas en el modelo final.

**Tabla 1.** Variables independientes del modelo `NCDMORT3070`.

| Código | Fuente | Descripción | Rol conceptual |
|---|---|---|---|
| `MDG_0000000001` | OMS | Tasa de mortalidad infantil (por 1.000 nacidos vivos) | Resiliencia del sistema de salud |
| `WHS4_100` | OMS | Camas hospitalarias por 10.000 habitantes | Infraestructura sanitaria |
| `TB_e_inc_num_log1p` | OMS | log1p de la incidencia absoluta de tuberculosis | Carga de enfermedad transmisible |
| `SP.DYN.LE00.IN` | Banco Mundial | Esperanza de vida al nacer (años) | Indicador agregado de salud |
| `SH.MED.PHYS.ZS` | Banco Mundial | Médicos por 1.000 habitantes | Cobertura de personal médico |
| `SH.XPD.CHEX.GD.ZS` | Banco Mundial | Gasto en salud (% del PIB) | Esfuerzo financiero en salud |
| `SH.H2O.BASW.ZS` | Banco Mundial | Acceso a agua potable básica (% de la población) | Determinante social |
| `NY.GDP.PCAP.CD_log1p` | Banco Mundial | log1p del PIB per cápita en USD corrientes | Desarrollo económico |

### 5.4 Esquema común y volumen

Todos los CSV crudos se normalizan al esquema *long*:

```text
country_code, year, indicator_code, indicator_name, source, value
```

El extractor de OMS conserva además las columnas dimensionales originales (`Dim1Type`, `Dim1`, `Dim2Type`, `Dim2`, `Low`, `High`, `ParentLocation`, fechas), porque son necesarias para resolver duplicados por dimensión sexo.

Tras la consolidación y filtrado, el panel modelable contiene **4.070 observaciones** correspondientes a **185 países** en el rango **2000–2021**. El año 2025 se excluye por ausencia generalizada de datos. Los códigos de país se restringen a un conjunto fijo de 250 códigos ISO3 documentado en [src/eda.py](src/eda.py); los agregados regionales y por nivel de ingreso del Banco Mundial se descartan por defecto (pueden incluirse con la bandera `--include-aggregates`).

### 5.5 Calidad y faltantes

Las series presentan distintos grados de completitud según indicador y país. La estrategia de imputación se aplica únicamente a predictores y se describe en la sección 6.3.

---

## 6. Materiales y Métodos

### 6.1 Arquitectura general

La Figura 1 describe la arquitectura del proyecto. Está organizada en cinco etapas estrictamente secuenciales: (1) extracción, (2) consolidación y EDA, (3) carga dimensional en PostgreSQL, (4) analítica con Polars y (5) dashboard de modelos.

**Figura 1.** Arquitectura de cinco etapas del pipeline.

```
┌──────────────────┐     ┌────────────────────┐     ┌────────────────┐
│  WHO GHO API     │     │ src/extractors/    │     │ data/raw/who/  │
│  World Bank API  │ ──► │  who_extractor.py  │ ──► │ data/raw/      │
│                  │     │  wb_extractor.py   │     │   world_bank/  │
└──────────────────┘     └────────────────────┘     └────────┬───────┘
                                                             │ src/main.py
                                                             ▼
                                                   ┌──────────────────┐
                                                   │   src/eda.py     │
                                                   │ limpieza,        │
                                                   │ imputación,      │
                                                   │ pivote wide      │
                                                   └────────┬─────────┘
                                                            ▼
                                          data/processed/eda/unified_wide.csv
                                                            │
                                                            ▼
                                            ┌─────────────────────────────┐
                                            │  src/database/              │
                                            │   load_postgres.py          │
                                            │  (esquema analytics.*)      │
                                            └────────┬────────────────────┘
                                                     ▼
                                       analytics.v_model_panel  (PostgreSQL)
                                            │                  │
                            ┌───────────────┘                  └──────────────┐
                            ▼                                                 ▼
                  ┌──────────────────┐                              ┌──────────────────┐
                  │ src/polars_      │                              │  src/app.py      │
                  │  analysis.py     │                              │  (Streamlit:     │
                  │  (Polars +       │                              │   Ridge / RF /   │
                  │   Plotly HTML)   │                              │   XGBoost)       │
                  └──────────────────┘                              └──────────────────┘
```

### 6.2 Extracción

Para cada indicador definido en [src/config/indicators.py](src/config/indicators.py), el orquestador [src/main.py](src/main.py) invoca al extractor correspondiente, normaliza el resultado al esquema *long* común y persiste un CSV por indicador en `data/raw/who/` o `data/raw/world_bank/`. Cualquier fallo de un indicador individual se registra y se contabiliza, pero no detiene la ejecución del resto del lote — el pipeline es tolerante a fallos parciales y reportable.

### 6.3 Consolidación, limpieza e imputación

El módulo [src/eda.py](src/eda.py) ejecuta las siguientes operaciones, en orden:

1. **Validación de esquema.** Cada CSV debe contener exactamente las columnas esperadas; los archivos no conformes se rechazan.
2. **Filtro temporal.** Se conservan únicamente los años `2000–2024`.
3. **Filtro geográfico.** Se conservan únicamente los códigos ISO3 de países y territorios soberanos (lista fija en `ISO3_COUNTRY_CODES`).
4. **Resolución de duplicados WHO por dimensión `SEX`.** El indicador `NCDMORT3070` se reporta por la OMS desagregado por sexo (`SEX_BTSX`, `SEX_FMLE`, `SEX_MLE`); se conserva exclusivamente `SEX_BTSX` (ambos sexos) como valor nacional total.
5. **Eliminación de redundancia.** Se elimina `WHOSIS_000001` por ser conceptualmente equivalente a `SP.DYN.LE00.IN` (ambos miden esperanza de vida al nacer).
6. **Pivote a formato *wide*** por la llave `(country_code, year)`.
7. **Imputación de predictores únicamente** (el target `NCDMORT3070` se mantiene sin imputar). La estrategia es jerárquica:
   - **Interpolación lineal por país** ordenando por `country_code` y `year`. Preserva la trayectoria histórica propia de cada país.
   - **Mediana por año** para los huecos no resueltos por interpolación. Se prefiere mediana sobre media por su robustez ante outliers.
   - **Mediana global** como respaldo final cuando un año no tiene suficientes observaciones.
8. **Transformaciones logarítmicas.** Se aplica `log1p` a `NY.GDP.PCAP.CD` y `TB_e_inc_num` para suavizar colas largas: el PIB per cápita varía entre cientos y más de 100.000 USD; la incidencia absoluta de tuberculosis varía en varios órdenes de magnitud.

El resultado es el archivo `data/processed/eda/unified_wide.csv` con las nueve columnas modelables más los identificadores `country_code` y `year`.

### 6.4 Almacenamiento dimensional en PostgreSQL

El módulo [src/database/load_postgres.py](src/database/load_postgres.py) reconstruye el almacén analítico desde cero en cada ejecución (estrategia *drop-and-reload*, válida para datasets pequeños y reproducibles). El esquema `analytics` contiene:

- `analytics.dim_country` — dimensión país (ISO3, nombre).
- `analytics.dim_year` — dimensión año.
- `analytics.dim_indicator` — dimensión indicador (código, nombre, fuente).
- `analytics.fact_indicator_value` — tabla de hechos en formato *long*.
- `analytics.v_model_panel` — vista *wide* materializada conceptualmente sobre los hechos, expuesta como única superficie de consumo analítico.

PostgreSQL 16 corre en un contenedor Docker definido en [docker-compose.yml](docker-compose.yml), parametrizado por variables de entorno (`POSTGRES_*`) cargadas desde un archivo `.env` local que **no se versiona**. El puerto local mapeado es `5433 → 5432`. Las credenciales se leen de `.env` en cada script para evitar credenciales *hard-coded*.

### 6.5 Análisis distribuido con Polars

El módulo [src/polars_analysis.py](src/polars_analysis.py) usa **Polars** (DataFrames columnares en Rust) y **ConnectorX** (lector multihilo de bases de datos relacionales) para extraer la vista `analytics.v_model_panel` directamente a memoria en formato Arrow, sin pasar por Pandas. Sobre ese DataFrame se realizan:
- Enriquecimiento geográfico ISO3 → continente vía `pycountry-convert`.
- Agregaciones por década y región.
- Exportación de estadísticas y matrices de correlación a `data/processed/polars_eda/*.parquet`.
- Generación de visualizaciones interactivas HTML con **Plotly**: scatter animado evolutivo, mapa coroplético mundial y boxplot por región (`data/processed/polars_eda/plots/`).

Se eligió Polars sobre Pandas por su mejor rendimiento en operaciones agrupadas y porque la combinación Polars + ConnectorX + Parquet representa un *stack* moderno de Big Data tabular en memoria coherente con los objetivos de la asignatura.

### 6.6 Modelos predictivos y protocolo experimental

**Variable objetivo:** `NCDMORT3070`.

**Predictores:** las ocho variables descritas en la Tabla 1, en sus formas finales (incluyendo las versiones `_log1p`).

**División temporal estricta.** Se aplica un *split out-of-time* parametrizable desde la interfaz, con corte por defecto en **2021**: las filas con `year < 2021` van a entrenamiento (3.885 observaciones) y las filas restantes a prueba (185 observaciones — una por país en 2021). Esta estrategia mide explícitamente la capacidad del modelo de generalizar a un período no observado, que es la pregunta de interés.

**Identificadores no usados como features.** `country_code` y `year` se separan como identificadores; no se incluyen como predictores numéricos directos para evitar que el modelo memorice países o tendencias temporales sin señal causal.

**Modelos comparados.**
- **Ridge Regression** (`sklearn.linear_model.Ridge`) con `StandardScaler` previo (los coeficientes lineales requieren features estandarizadas).
- **Random Forest** (`sklearn.ensemble.RandomForestRegressor`) con 300 árboles.
- **XGBoost** (`xgboost.XGBRegressor`) con learning rate 0,05 y `subsample = 0,8`, ambos para regularizar.

**Métricas reportadas:** RMSE, MAE y R² en *train* y *test*. Adicionalmente: análisis de residuales, scatter de predicciones vs reales, distribución del error, *top-10* países con mayor error absoluto e importancia de variables por modelo.

**Implementación interactiva.** Toda la comparación se materializa en el dashboard [src/app.py](src/app.py) (Streamlit), que permite:
- Mover el año de corte mediante un *slider*.
- Activar/desactivar modelos individualmente.
- Navegar tres tabs (Métricas, Predicciones, Importancia de Variables).
- Operar en dos modos de fuente de datos: PostgreSQL (preferido vía ConnectorX) y CSV procesado como respaldo *offline*.

### 6.7 Reproducibilidad

El proyecto está empaquetado para ejecutarse de cero a *dashboard* con la siguiente secuencia de comandos:

```bash
python -m venv .venv
.venv/Scripts/activate                       # Windows; en Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                         # completar credenciales locales
.venv/Scripts/python src/main.py             # 1. extraer
.venv/Scripts/python src/eda.py              # 2. limpiar y consolidar
docker compose up -d postgres                # 3. levantar PostgreSQL
.venv/Scripts/python src/database/load_postgres.py    # 4. cargar
.venv/Scripts/python src/polars_analysis.py  # 5. analítica + plots
.venv/Scripts/python -m streamlit run src/app.py      # 6. dashboard en http://localhost:8501
```

Las dependencias están listadas en [requirements.txt](requirements.txt). Las reglas operativas y guías para futuros mantenedores están documentadas en [CLAUDE.md](CLAUDE.md), [AGENTS.md](AGENTS.md) e [instructions.md](instructions.md).

---

## 7. Resultados

### 7.1 Caracterización del panel modelable

Tras aplicar todas las reglas de limpieza, el panel modelable contiene **4.070 observaciones** distribuidas en **185 países** y **22 años** (2000–2021). El target `NCDMORT3070` tiene una media global de **21,15 %** y desviación estándar de **7,46 pp**, con un mínimo de 6,9 % (típicamente Japón, Suiza) y un máximo de 45,3 % (países subsaharianos con alta carga combinada de ENT y baja capacidad sanitaria). La amplitud de **38 puntos porcentuales** entre extremos confirma que existe variación suficiente para que un modelo capture señal real, y a la vez que el problema es genuinamente heterogéneo.

### 7.2 Resultados de los modelos predictivos

La Tabla 2 resume el desempeño de los tres modelos en *train* (2000–2020) y *test* (2021).

**Tabla 2.** Métricas de desempeño de los tres modelos sobre `NCDMORT3070`. Unidades: puntos porcentuales (pp).

| Modelo | RMSE Train | R² Train | RMSE Test | MAE Test | R² Test |
|---|---|---|---|---|---|
| Ridge Regression | 5,17 | 0,519 | 5,26 | 4,13 | 0,463 |
| **Random Forest** | **0,66** | **0,992** | **3,46** | **2,58** | **0,768** |
| XGBoost | 1,95 | 0,932 | 4,12 | 3,17 | 0,670 |

**Random Forest obtuvo el mejor desempeño en todas las métricas de prueba**: RMSE 3,46 pp, MAE 2,58 pp, R² 0,768. Más del 52 % de sus predicciones cayeron a menos de 2 pp del valor real, sobre una escala observada de 38 pp. XGBoost quedó en posición intermedia (R² 0,670). Ridge produjo el peor desempeño (R² 0,463) pero con el menor *gap* train–test (0,06), confirmando que es el modelo más estable pero el menos expresivo.

El R² de entrenamiento de **0,992** de Random Forest indica un sobreajuste pronunciado a los datos históricos; sin embargo, la generalización a 2021 sigue siendo la mejor de los tres, lo que sugiere que el modelo, pese a memorizar, captura patrones generalizables reales. XGBoost presenta el mayor *gap* train–test en R² (0,26), señal de que su esquema de boosting con regularización no fue suficiente para evitar sobreajuste en este tamaño de dataset.

### 7.3 Análisis de errores y residuales

La Tabla 3 lista los países con mayor error absoluto bajo el modelo XGBoost. El patrón es geográficamente coherente: la mayoría son países latinoamericanos y de Medio Oriente con indicadores estructurales intermedios.

**Tabla 3.** Top-7 países con mayor error absoluto del modelo XGBoost en 2021.

| País | Real (%) | Predicho XGB (%) | Error (pp) |
|---|---|---|---|
| El Salvador | 12,6 | 26,4 | +13,8 |
| Nicaragua | 12,6 | 23,4 | +10,8 |
| Paraguay | 16,1 | 26,5 | +10,4 |
| Colombia | 10,3 | 20,3 | +10,0 |
| Líbano | 11,9 | 21,6 | +9,7 |
| Jordania | 11,6 | 20,8 | +9,2 |
| México | 16,3 | 25,3 | +9,0 |

En todos los casos el modelo **sobrestima** la mortalidad real. El único error inverso notable es Eswatini (real 32,3 %; predicho 23,4 %): el modelo subestima a un país con indicadores estructurales aceptables pero problemas sanitarios persistentes. El análisis de residuales (visualizado en el dashboard, tab "Predicciones") muestra que los residuales positivos dominan el rango de predicciones bajas (10–20 %), lo que corresponde a un sesgo estructural común a los tres modelos.

### 7.4 Importancia de variables

Los tres modelos coinciden en el orden general de importancia de variables (Tabla 4), pese a usar criterios distintos (magnitud absoluta del coeficiente para Ridge; reducción de impureza media para los modelos de árboles).

**Tabla 4.** Ordenamiento típico de importancia de variables consistente entre los tres modelos.

| Rango | Variable | Interpretación |
|---|---|---|
| 1° | Esperanza de vida (`SP.DYN.LE00.IN`) | Proxy del nivel global de salud poblacional |
| 2° – 3° | PIB per cápita (log) (`NY.GDP.PCAP.CD_log1p`) | Capacidad de financiar y operar sistemas de salud |
| 3° – 4° | Mortalidad infantil (`MDG_0000000001`) | Condiciones de vida tempranas y resiliencia sanitaria |
| Medio | Médicos / 1.000 hab. (`SH.MED.PHYS.ZS`) | Acceso a personal médico |
| Bajo | Gasto en salud (% PIB) (`SH.XPD.CHEX.GD.ZS`) | Esfuerzo financiero — sorprendentemente débil |
| Último | Camas hospitalarias (`WHS4_100`) | Predictor menos informativo de manera consistente |

El hallazgo más contraintuitivo es que el **gasto en salud como % del PIB** aparece sistemáticamente entre las posiciones 6° y 8° de importancia. La razón conceptual es que esta variable mide *cantidad* y no *eficiencia*: Estados Unidos gasta cerca del 17 % del PIB en salud y mantiene una mortalidad por ENT relativamente alta, mientras que países con menor gasto pero con sistemas preventivos eficientes obtienen mejores resultados. Para política pública, esto sugiere priorizar la **estructura** del sistema (cobertura, prevención) sobre el simple incremento del presupuesto.

### 7.5 Productos del proyecto

El proyecto entrega:

1. Pipeline ETL ejecutable de extremo a extremo (`src/main.py`, `src/eda.py`).
2. Almacén dimensional sobre PostgreSQL con vista única de consumo (`analytics.v_model_panel`).
3. Análisis distribuido con Polars y artefactos `.parquet` y plots interactivos HTML.
4. Dashboard interactivo Streamlit con comparación de modelos, *split* temporal ajustable y modo *offline* CSV.
5. Documentación operativa exhaustiva: [README.md](README.md), [instructions.md](instructions.md), [AGENTS.md](AGENTS.md), [CLAUDE.md](CLAUDE.md) y [model_analysis.md](model_analysis.md) (este último contiene la discusión técnica detallada modelo-por-modelo).

---

## 8. Discusión y conclusiones

### 8.1 Lecciones técnicas

**El target es predecible — pero hasta cierto punto.** Random Forest explica el 77 % de la varianza de `NCDMORT3070` en 2021 a partir de ocho indicadores macro. Es un hallazgo positivo: confirma que las condiciones estructurales de un país tienen poder predictivo significativo sobre la mortalidad prematura por ENT. Pero el 23 % restante no es ruido: contiene información sobre políticas públicas específicas, transición epidemiológica, hábitos culturales (tabaco, dieta) y reformas de salud que los indicadores agregados no capturan. Para llevar el modelo a un R² superior a 0,85 sería necesario incorporar al menos: prevalencia de tabaquismo, presión arterial poblacional, cobertura de vacunación, e indicadores de sistemas de salud preventiva.

**Los errores no son aleatorios — son geográficos.** El patrón de sobrestimación concentrado en América Latina y Medio Oriente sugiere que estos países han logrado **caídas en mortalidad por ENT más rápidas de lo que sus variables estructurales predicen**. Esto no es un fallo del modelo: es una señal de que las políticas específicas de prevención y modificación de estilos de vida en estas regiones están operando por encima de lo que el desarrollo socioeconómico solo explicaría.

**Random Forest gana, pero hay que vigilarlo.** El sobreajuste de entrenamiento (R² = 0,992) es real. Si los datos posteriores a 2021 muestran tendencias estructuralmente nuevas — por ejemplo, secuelas post-pandémicas o shocks económicos prolongados — el modelo podría degradarse antes que XGBoost, que está más regularizado. Por eso, el dashboard mantiene los tres modelos disponibles: permite reentrenar con cortes posteriores y verificar si las posiciones relativas cambian.

**Esperanza de vida es predictor estelar pero conceptualmente cercano al target.** La esperanza de vida y la mortalidad prematura por ENT no son variables independientes — ambas miden facetas del mismo fenómeno de salud poblacional. Para predicción pura, esto es válido y útil. Para un modelo *explicativo* (causal), habría que reformular el problema o excluir esta variable y aceptar peores métricas.

### 8.2 Lecciones metodológicas y de ingeniería

La separación estricta entre fuentes (esquema *long* común), almacén (esquema dimensional), capa analítica (vista única) y consumo (dashboard) demostró ser robusta: cualquier cambio en los modelos del dashboard no requirió tocar el ETL, y agregar un nuevo indicador requiere solo registrar una entrada en [src/config/indicators.py](src/config/indicators.py) y reejecutar la cadena. Esta arquitectura es coherente con el principio de Kimball & Ross (2013) de aislar a los consumidores del esquema físico mediante vistas.

La elección de **Polars + ConnectorX + Parquet** sobre Pandas + SQLAlchemy + CSV resultó en tiempos de extracción y agregación significativamente menores en pruebas locales y, sobre todo, en una huella de memoria menor — relevante para escalar a paneles más grandes (más indicadores, granularidad subnacional).

La estrategia de **imputación jerárquica predictores–solo** preserva la integridad del target y honra la naturaleza temporal de los datos. Imputar el target habría introducido información sintética que el modelo aprendería a reproducir, inflando artificialmente las métricas.

### 8.3 Limitaciones

1. **Una sola fila por país en el conjunto de prueba.** El corte 2021 deja 185 observaciones de prueba, todas del mismo año. El modelo evalúa generalización temporal pero no puede separar el efecto país del efecto año en el conjunto de prueba.
2. **Indicadores macro únicamente.** No se incluyeron variables conductuales (tabaquismo, dieta) ni indicadores de política sanitaria.
3. **Imputación necesaria en predictores.** Aunque la estrategia es jerárquica y robusta, los países con menor cobertura histórica reciben proporcionalmente más imputación, lo que puede sesgar las predicciones para esas naciones.
4. **No hay tests automatizados aún.** El proyecto no cuenta todavía con suite de tests, lo que es un siguiente paso natural antes de poner el pipeline en operación recurrente.

### 8.4 Trabajo futuro

- Incorporar indicadores conductuales (prevalencia de tabaquismo, índice de masa corporal poblacional) como features adicionales.
- Validar con un *split* por país (*country-holdout*) además del temporal, para distinguir generalización geográfica de generalización temporal.
- Probar modelos lineales mixtos con efectos fijos por país que actúen como baseline interpretable adicional.
- Añadir tests de cleaning rules y de parsing de respuestas de las APIs.
- Empaquetar el flujo en un orquestador (Airflow o Prefect) para reentrenamientos periódicos.

### 8.5 Conclusión

El proyecto demuestra que es posible construir, con datos abiertos y herramientas estándar, una plataforma reproducible que combina ingeniería de datos (extracción, esquema dimensional, vista única) y analítica predictiva (comparación de modelos con *split* temporal). El modelo final (Random Forest, R² = 0,77) responde afirmativamente a la pregunta de investigación: **sí, una proporción alta de la variación internacional de la mortalidad prematura por ENT puede explicarse y anticiparse a partir de indicadores socioeconómicos y de infraestructura sanitaria disponibles públicamente**. El principal hallazgo sustantivo es que el *cuánto* gasta un país en salud importa menos que su *desarrollo general* y su *capacidad sanitaria efectiva* — un mensaje relevante para política pública.

---

## 9. Bibliografía

Bitew, F. H., Sparks, C. S., & Nyarko, S. H. (2020). Machine learning algorithms for predicting under-five mortality in Ethiopia. *BMC Medical Informatics and Decision Making*, 20, 1–10.

Bloom, D. E., Cafiero, E. T., Jané-Llopis, E., Abrahams-Gessel, S., Bloom, L. R., Fathima, S., Feigl, A. B., Gaziano, T., Mowafi, M., Pandya, A., Prettner, K., Rosenberg, L., Seligman, B., Stein, A. Z., & Weinstein, C. (2011). *The Global Economic Burden of Non-communicable Diseases*. World Economic Forum and Harvard School of Public Health, Geneva.

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.

Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794).

Commission on Social Determinants of Health. (2008). *Closing the gap in a generation: health equity through action on the social determinants of health*. World Health Organization, Geneva.

Deaton, A. (2013). *The Great Escape: Health, Wealth, and the Origins of Inequality*. Princeton University Press.

GBD 2019 Risk Factors Collaborators. (2020). Global burden of 87 risk factors in 204 countries and territories, 1990–2019: a systematic analysis for the Global Burden of Disease Study 2019. *The Lancet*, 396(10258), 1223–1249.

Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.

Hoerl, A. E., & Kennard, R. W. (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems. *Technometrics*, 12(1), 55–67.

Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: principles and practice* (3rd ed.). OTexts.

James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning with Applications in R* (2nd ed.). Springer.

Kimball, R., & Ross, M. (2013). *The Data Warehouse Toolkit: The Definitive Guide to Dimensional Modeling* (3rd ed.). Wiley.

Little, R. J. A., & Rubin, D. B. (2019). *Statistical Analysis with Missing Data* (3rd ed.). Wiley.

Marmot, M. (2005). Social determinants of health inequalities. *The Lancet*, 365(9464), 1099–1104.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

Preston, S. H. (1975). The changing relation between mortality and level of economic development. *Population Studies*, 29(2), 231–248.

United Nations. (2015). *Transforming our world: the 2030 Agenda for Sustainable Development* (A/RES/70/1). United Nations General Assembly.

Weng, S. F., Reps, J., Kai, J., Garibaldi, J. M., & Qureshi, N. (2017). Can machine-learning improve cardiovascular risk prediction using routine clinical data? *PLoS ONE*, 12(4), e0174944.

World Bank. (2024). *World Development Indicators*. Washington, D.C.: The World Bank. https://databank.worldbank.org/source/world-development-indicators

World Health Organization. (2022). *Noncommunicable diseases: Progress monitor 2022*. World Health Organization, Geneva.

World Health Organization. (n.d.). *Global Health Observatory data repository*. Recuperado de https://www.who.int/data/gho

---

## 10. Anexos

### Anexo A — Estructura del repositorio

```
socioeconomic-ncd-risk/
├── README.md                       Contexto académico del proyecto (es)
├── instructions.md                 Guía operativa completa (es)
├── AGENTS.md                       Convenciones de código del repositorio
├── CLAUDE.md                       Guía para mantenedores (incl. invariantes de eda.py)
├── model_analysis.md               Análisis detallado modelo por modelo (es)
├── docker-compose.yml              Definición del contenedor PostgreSQL
├── .env.example                    Plantilla de credenciales (sin secretos reales)
├── requirements.txt                Dependencias Python
├── src/
│   ├── main.py                     Orquestador de extracción
│   ├── eda.py                      Limpieza, imputación, dataset modelable
│   ├── polars_analysis.py          Analítica con Polars + Plotly
│   ├── app.py                      Dashboard Streamlit
│   ├── config/
│   │   └── indicators.py           Catálogo de indicadores OMS y Banco Mundial
│   ├── extractors/
│   │   ├── who_extractor.py        Cliente WHO GHO API
│   │   └── world_bank_extractor.py Cliente World Bank API
│   ├── database/
│   │   └── load_postgres.py        Carga al esquema dimensional
│   └── utils/
│       ├── file_writer.py
│       └── logging.py
└── data/
    ├── raw/{who, world_bank}/      CSVs descargados (no versionados)
    └── processed/                  Salidas de eda.py y polars_analysis.py
```

### Anexo B — Comandos de evaluación

Para reproducir el proyecto completo desde cero (asumiendo Python 3.11+, Docker y `git`):

```bash
git clone <url-repositorio>
cd socioeconomic-ncd-risk
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env
# Editar .env con valores locales (POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD)

.venv/Scripts/python src/main.py
.venv/Scripts/python src/eda.py
docker compose up -d postgres
.venv/Scripts/python src/database/load_postgres.py
.venv/Scripts/python src/polars_analysis.py
.venv/Scripts/python -m streamlit run src/app.py
```

Para evaluación rápida sin levantar PostgreSQL, el dashboard opera en modo *offline* leyendo `data/processed/eda/unified_wide.csv`.

### Anexo C — Catálogo completo de indicadores

Definido en [src/config/indicators.py](src/config/indicators.py).

**OMS (`WHO_INDICATORS`):**
- `NCDMORT3070` — Probabilidad de morir entre 30 y 70 años por ENT (target).
- `WHOSIS_000001` — Esperanza de vida al nacer (descriptive; eliminado del modelo final).
- `MDG_0000000001` — Tasa de mortalidad infantil (descriptive).
- `WHS4_100` — Camas hospitalarias por 10.000 habitantes (health_feature).
- `TB_e_inc_num` — Incidencia de tuberculosis (descriptive; usado como `_log1p`).

**Banco Mundial (`WORLD_BANK_INDICATORS`):**
- `SH.XPD.CHEX.GD.ZS` — Gasto en salud como % del PIB (predictive_feature).
- `SH.MED.PHYS.ZS` — Médicos por 1.000 habitantes (health_feature).
- `SP.DYN.LE00.IN` — Esperanza de vida al nacer (descriptive).
- `SH.H2O.BASW.ZS` — Acceso a agua potable básica (health_social_feature).
- `NY.GDP.PCAP.CD` — PIB per cápita en USD corrientes (socioeconomic_feature; usado como `_log1p`).

### Anexo D — Esquema de la vista `analytics.v_model_panel`

| Columna | Tipo | Origen |
|---|---|---|
| `country_code` | TEXT | Identificador (ISO3) |
| `year` | INT | Identificador temporal |
| `MDG_0000000001` | NUMERIC | Mortalidad infantil (OMS) |
| `NCDMORT3070` | NUMERIC | **Target** — probabilidad de morir 30–70 por ENT (OMS) |
| `SH.H2O.BASW.ZS` | NUMERIC | Agua potable (Banco Mundial) |
| `SH.MED.PHYS.ZS` | NUMERIC | Médicos / 1.000 hab. (Banco Mundial) |
| `SH.XPD.CHEX.GD.ZS` | NUMERIC | Gasto salud % PIB (Banco Mundial) |
| `SP.DYN.LE00.IN` | NUMERIC | Esperanza de vida (Banco Mundial) |
| `WHS4_100` | NUMERIC | Camas hospitalarias (OMS) |
| `NY.GDP.PCAP.CD_log1p` | NUMERIC | log1p del PIB per cápita (Banco Mundial, transformado) |
| `TB_e_inc_num_log1p` | NUMERIC | log1p de incidencia TB (OMS, transformado) |

Consulta recomendada para entrenamiento de modelos:

```sql
SELECT *
FROM analytics.v_model_panel
WHERE "NCDMORT3070" IS NOT NULL;
```

### Anexo E — Detalle técnico del análisis modelo por modelo

El documento [model_analysis.md](model_analysis.md) contiene el análisis exhaustivo modelo por modelo, incluyendo: explicación de cada métrica con interpretación en el dominio (puntos porcentuales del riesgo de morir), análisis del patrón de errores con identificación de los siete países con mayor sobreestimación, hallazgos sobre importancia de variables (incluyendo el contraintuitivo bajo peso del gasto en salud) y veredicto operativo sobre cuál modelo elegir según el objetivo (mejor predicción puntual, robustez futura o interpretabilidad).

Los artefactos visuales (scatter real–predicho, residuales, importancia, radar comparativo) se generan dinámicamente en el dashboard [src/app.py](src/app.py); los mapas y boxplots por región se generan como HTML interactivo en [data/processed/polars_eda/plots/](data/processed/polars_eda/plots/) al ejecutar [src/polars_analysis.py](src/polars_analysis.py).
