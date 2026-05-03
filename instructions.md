# Instrucciones de uso del proyecto

Este proyecto construye un panel historico de indicadores de salud y variables socioeconomicas para analizar la relacion entre condiciones estructurales de los paises y mortalidad prematura por enfermedades no transmisibles.

El dataset final modelable queda disponible en PostgreSQL como una vista wide llamada `analytics.v_model_panel`.

## Archivos principales

`README.md`

Contiene el contexto academico del proyecto, la pregunta de investigacion, fuentes de datos y objetivo general.

`src/config/indicators.py`

Define los indicadores descargados desde WHO y World Bank. Incluye el nombre descriptivo de cada indicador y su rol conceptual, por ejemplo target, variable socioeconomica o variable de salud.

`src/extractors/who_extractor.py`

Extrae datos desde WHO GHO API. El extractor conserva el esquema ampliado de WHO, incluyendo columnas como `Dim1Type`, `Dim1`, `Dim2Type`, `Dim2`, `Low`, `High`, `ParentLocation` y fechas de la observacion. Esto se hizo para explicar duplicados por dimensiones, especialmente sexo.

`src/extractors/world_bank_extractor.py`

Extrae indicadores desde World Bank Open Data API y los normaliza al esquema base:

```text
country_code, year, indicator_code, indicator_name, source, value
```

`src/main.py`

Orquesta la extraccion completa de WHO y World Bank. Escribe los CSV crudos en:

```text
data/raw/who/
data/raw/world_bank/
```

`src/eda.py`

Construye el panel unificado, aplica limpieza, prepara el dataset modelable y genera EDA. Sus salidas principales quedan en:

```text
data/processed/eda/unified_wide.csv
data/processed/eda/unified_wide_before_imputation.csv
data/processed/eda/unified_long.csv
data/processed/eda/unified_long_canonical.csv
data/processed/eda/eda_report.md
data/processed/eda/tables/
data/processed/eda/plots/
```

`docker-compose.yml`

Define el contenedor PostgreSQL del proyecto. Usa variables desde `.env`, por lo que no contiene credenciales hard-coded.

`src/database/load_postgres.py`

Levanta PostgreSQL, limpia la base, crea el esquema dimensional, carga el dataset final y verifica lectura desde la base.

`.env.example`

Plantilla versionable para crear el archivo `.env` local. No debe contener credenciales reales.

`.env`

Archivo local con credenciales reales. Esta ignorado por Git y no debe versionarse.

## Metodologia de limpieza e imputacion

El panel final se construye desde los CSV crudos en formato long y se pivotea por la llave:

```text
country_code + year
```

La limpieza aplicada fue:

1. Filtrar solo anos `2000-2024`.
2. Excluir `2025` por falta generalizada de datos.
3. Filtrar solo codigos ISO3 de paises y territorios.
4. Resolver duplicados de WHO priorizando `SEX_BTSX`, que representa ambos sexos.
5. Eliminar `WHOSIS_000001`, porque es redundante con `SP.DYN.LE00.IN`.
6. Mantener `NCDMORT3070` sin imputar, porque es la variable objetivo.
7. Imputar solo predictores.
8. Aplicar transformaciones logaritmicas a variables con fuerte sesgo.

La duplicidad en WHO se explica principalmente por la dimension `SEX`:

```text
SEX_BTSX
SEX_FMLE
SEX_MLE
```

Para el dataset modelable se conserva `SEX_BTSX` como valor nacional total.

### Imputacion

La imputacion se aplica solo a predictores, nunca al target `NCDMORT3070`.

El flujo es:

1. Interpolacion lineal por pais, ordenando por `country_code` y `year`.
2. Mediana por `year` para huecos restantes.
3. Mediana global como respaldo final cuando no existe mediana anual.

La razon de esta metodologia es que los datos son series historicas por pais. La interpolacion conserva la evolucion temporal propia de cada pais. La mediana anual evita usar medias sensibles a outliers. La mediana global se usa solo como respaldo minimo para completar predictores cuando un ano no tiene suficientes datos.

### Transformaciones

Se reemplazaron estas variables por versiones `log1p`:

```text
NY.GDP.PCAP.CD -> NY.GDP.PCAP.CD_log1p
TB_e_inc_num -> TB_e_inc_num_log1p
```

Esto suaviza la escala de PIB per capita e incidencia absoluta de tuberculosis, que tienen colas largas y outliers.

## Dataset final para modelado

El archivo final wide es:

```text
data/processed/eda/unified_wide.csv
```

Columnas esperadas:

```text
country_code
year
MDG_0000000001
NCDMORT3070
SH.H2O.BASW.ZS
SH.MED.PHYS.ZS
SH.XPD.CHEX.GD.ZS
SP.DYN.LE00.IN
WHS4_100
NY.GDP.PCAP.CD_log1p
TB_e_inc_num_log1p
```

Para entrenar modelos predictivos, usar solo filas con target disponible:

```sql
SELECT *
FROM analytics.v_model_panel
WHERE "NCDMORT3070" IS NOT NULL;
```

## Configuracion de credenciales

Las credenciales deben ir en `.env`.

Crear el archivo a partir de la plantilla:

```bash
cp .env.example .env
```

Completar `.env` con los valores locales:

```text
POSTGRES_HOST=localhost
POSTGRES_HOST_PORT=5433
POSTGRES_CONTAINER_NAME=socioeconomic_ncd_postgres
POSTGRES_DB=<database>
POSTGRES_USER=<user>
POSTGRES_PASSWORD=<password>
```

No subir `.env` a Git. Ya esta ignorado en `.gitignore`.

## Uso de Docker y PostgreSQL

Levantar PostgreSQL:

```bash
docker compose up -d postgres
```

Verificar estado del contenedor:

```bash
docker compose ps
```

El puerto local configurado es:

```text
localhost:5433 -> postgres:5432
```

## Cargar datos en PostgreSQL

Primero asegurarse de tener el dataset final generado:

```bash
.venv/bin/python src/eda.py
```

Luego cargar PostgreSQL:

```bash
.venv/bin/python src/database/load_postgres.py
```

Este script hace lo siguiente:

1. Lee credenciales desde `.env`.
2. Levanta el contenedor PostgreSQL si no esta activo.
3. Elimina esquemas de usuario existentes para evitar datos antiguos.
4. Crea el esquema `analytics`.
5. Crea tablas dimensionales.
6. Carga el dataset limpio desde `unified_wide.csv`.
7. Crea la vista `analytics.v_model_panel`.
8. Verifica conteos y lectura de filas.

Tablas creadas:

```text
analytics.dim_country
analytics.dim_year
analytics.dim_indicator
analytics.fact_indicator_value
```

Vista principal para analisis y modelos:

```text
analytics.v_model_panel
```

## Lectura desde PostgreSQL

URL JDBC:

```text
jdbc:postgresql://${POSTGRES_HOST}:${POSTGRES_HOST_PORT}/${POSTGRES_DB}
```

Tabla o vista recomendada:

```text
analytics.v_model_panel
```

Consulta recomendada para ML:

```sql
SELECT *
FROM analytics.v_model_panel
WHERE "NCDMORT3070" IS NOT NULL;
```

Consulta recomendada para analisis descriptivo general:

```sql
SELECT *
FROM analytics.v_model_panel;
```

Consulta recomendada para analisis dimensional por indicador:

```sql
SELECT
    c.country_code,
    y.year,
    i.indicator_code,
    i.indicator_name,
    i.source,
    f.value
FROM analytics.fact_indicator_value f
JOIN analytics.dim_country c
    ON c.country_code = f.country_code
JOIN analytics.dim_year y
    ON y.year = f.year
JOIN analytics.dim_indicator i
    ON i.indicator_code = f.indicator_code;
```

## Siguiente etapa: Polars distribuido

Para analisis distribuido con Polars o cualquier motor compatible con JDBC, leer desde `analytics.v_model_panel`.

Analisis esperados:

1. Correlacion entre gasto en salud y mortalidad:

```text
SH.XPD.CHEX.GD.ZS vs NCDMORT3070
```

2. Estadisticas descriptivas por indicador.

Para estadisticas por indicador, usar la tabla dimensional long:

```text
analytics.fact_indicator_value
```

Para correlaciones y entrenamiento de modelos, usar la vista wide:

```text
analytics.v_model_panel
```

## Siguiente etapa: modelo predictivo

Variable objetivo:

```text
NCDMORT3070
```

Predictores disponibles:

```text
MDG_0000000001
SH.H2O.BASW.ZS
SH.MED.PHYS.ZS
SH.XPD.CHEX.GD.ZS
SP.DYN.LE00.IN
WHS4_100
NY.GDP.PCAP.CD_log1p
TB_e_inc_num_log1p
```

Antes de entrenar:

1. Leer `analytics.v_model_panel`.
2. Filtrar `NCDMORT3070 IS NOT NULL`.
3. Separar `country_code` y `year` como identificadores, no como predictores numericos directos.
4. Usar las demas columnas como features.
5. Validar el modelo con separacion temporal o por pais si el objetivo es medir generalizacion real.

## Reproducibilidad completa

Flujo completo desde cero:

```bash
cp .env.example .env
# Completar .env localmente.

.venv/bin/python src/main.py
.venv/bin/python src/eda.py
docker compose up -d postgres
.venv/bin/python src/database/load_postgres.py
```

Si solo se quiere recargar PostgreSQL con el dataset ya existente:

```bash
.venv/bin/python src/database/load_postgres.py
```
