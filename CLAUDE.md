# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Academic project that predicts premature mortality from non-communicable diseases (target: WHO indicator `NCDMORT3070`, ages 30–70) from country-level socioeconomic and health-infrastructure indicators. Data comes from the WHO GHO API and the World Bank Open Data API. End-to-end flow: extract raw indicators → unify/clean/impute → load into PostgreSQL (dimensional schema) → analyze with Polars → serve a Streamlit model-comparison dashboard.

Project-facing documentation is written in Spanish (`README.md`, `instructions.md`). Preserve Spanish in user-facing docs unless a file is explicitly aimed at English contributors.

## Pipeline (run order matters)

Each stage consumes the output of the previous one. Use the project venv: `.venv/Scripts/python` on Windows, `.venv/bin/python` on Linux/macOS.

```bash
# 0. Setup (once)
python -m venv .venv
.venv/Scripts/activate                # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env                  # then fill in POSTGRES_* values

# 1. Extract raw CSVs from WHO + World Bank → data/raw/{who,world_bank}/
.venv/bin/python src/main.py

# 2. Unify, clean, impute, transform → data/processed/eda/unified_wide.csv (+ EDA report/plots)
.venv/bin/python src/eda.py
#    Optional flags: --raw-dir, --output-dir, --start-year, --end-year, --include-aggregates

# 3. Start Postgres + load dimensional schema and v_model_panel view
docker compose up -d postgres
.venv/bin/python src/database/load_postgres.py

# 4. Polars/Plotly analysis → data/processed/polars_eda/{*.parquet, plots/*.html}
.venv/bin/python src/polars_analysis.py

# 5. Streamlit dashboard (Ridge / Random Forest / XGBoost comparison) on http://localhost:8501
.venv/bin/python -m streamlit run src/app.py
```

There are no tests, linters, or build system configured. Do not invent `make test` / `pytest` commands — none exist yet.

## Architecture

### Data contract (long format)

Every raw extractor must emit CSVs with this schema; `src/eda.py` rejects anything else:

```text
country_code, year, indicator_code, indicator_name, source, value
```

The WHO extractor additionally preserves the original WHO dimensional columns (`Dim1Type`, `Dim1`, `Dim2Type`, `Dim2`, `Low`, `High`, `ParentLocation`, dates). These extra columns must survive into the raw CSV — they are what `eda.py` uses to deduplicate by `SEX_BTSX`.

### Cleaning invariants enforced by `src/eda.py`

These are not arbitrary — they are the contract the downstream model and Postgres view depend on. Don't change them without updating the model panel and dashboard:

- Year range: `2000–2024` only. `2025` is excluded (sparse).
- Country filter: ISO3 codes from the hardcoded `ISO3_COUNTRY_CODES` set; aggregates (regions, income groups) are dropped unless `--include-aggregates` is passed.
- WHO duplicates by sex dimension are resolved by keeping `SEX_BTSX` (both sexes).
- `WHOSIS_000001` is dropped from the model panel — redundant with `SP.DYN.LE00.IN`.
- `NCDMORT3070` (target) is **never imputed**. Imputation is predictors-only.
- Imputation order: linear interpolation per country (sorted by year) → median by year → global median fallback.
- Log transforms: `NY.GDP.PCAP.CD` and `TB_e_inc_num` are replaced by `*_log1p` versions to tame long tails.

### Modelable dataset

`data/processed/eda/unified_wide.csv` is the single source of truth for the model. Expected columns:

```text
country_code, year,
MDG_0000000001, NCDMORT3070, SH.H2O.BASW.ZS, SH.MED.PHYS.ZS,
SH.XPD.CHEX.GD.ZS, SP.DYN.LE00.IN, WHS4_100,
NY.GDP.PCAP.CD_log1p, TB_e_inc_num_log1p
```

When training, filter `NCDMORT3070 IS NOT NULL` and treat `country_code` + `year` as identifiers, not numeric features. Validate with a temporal split (or country-holdout) — that's what the Streamlit dashboard does.

### PostgreSQL layer

`src/database/load_postgres.py` rebuilds the analytical schema from scratch every run (drops user schemas, recreates `analytics`, reloads from CSV). Tables:

- `analytics.dim_country`, `analytics.dim_year`, `analytics.dim_indicator`
- `analytics.fact_indicator_value` — long-format facts
- `analytics.v_model_panel` — wide view that downstream consumers (Polars, Streamlit) read from

Local Postgres listens on `localhost:5433` (mapped to the container's 5432). Credentials are read from `.env` via the `POSTGRES_*` variables; never hard-code them.

The script shells out to `docker compose exec ... psql` and uses a hardcoded staging path `/private/tmp/socioeconomic_ncd_postgres_load` — that path is macOS/Linux-style and may need adjustment on Windows.

### Adding a new indicator

1. Register it in `src/config/indicators.py` under `WHO_INDICATORS` or `WORLD_BANK_INDICATORS` with an `indicator_name` and `role`.
2. Re-run the pipeline from step 1. The extractor and EDA will pick it up automatically.
3. If it should be a model feature, also add it to the `FEATURES` list in [src/app.py](src/app.py) and the expected wide-column documentation.

### Streamlit dashboard data source

[src/app.py](src/app.py) tries PostgreSQL first (via `connectorx`) and falls back to `data/processed/eda/unified_wide.csv` if the DB is unreachable. Both paths must keep working — don't make the dashboard hard-depend on Postgres.

## Coding conventions (from AGENTS.md)

- Python: 4-space indentation, snake_case modules/functions, PascalCase classes, UPPER_CASE constants. Domain-tied names (`ncd_mortality_rate`, `world_bank_indicator`, `country_year_panel`).
- **Mandatory commenting style**: every logical block must have a short comment explaining *what* it does and *why*. This is a project-wide rule from AGENTS.md and overrides the usual "no comments" default — follow it when editing existing files and when adding new ones in this repo.
- Prioritize simplicity over abstraction; prefer slightly verbose, readable code.
- Commit messages: short and imperative (e.g. `Initialize README with project description and goals`).

## Files not to commit

`.gitignore` already excludes `.env`, `.venv/`, `data/raw/**/*.csv`, `data/processed/`, and `working/`. Do not add raw or processed datasets to git, and do not commit `.env`.
