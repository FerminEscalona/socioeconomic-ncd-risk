"""Carga el dataset modelable en PostgreSQL con un esquema dimensional."""

from __future__ import annotations

import csv
import os
import subprocess
import time
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = ROOT_DIR / "data" / "processed" / "eda" / "unified_wide.csv"
METADATA_PATH = ROOT_DIR / "data" / "processed" / "eda" / "tables" / "indicator_metadata.csv"
TEMP_DIR = Path("/private/tmp/socioeconomic_ncd_postgres_load")
TARGET_COLUMN = "NCDMORT3070"


def load_env() -> dict[str, str]:
    """Lee credenciales locales desde .env sin versionarlas en el repo."""
    env_path = ROOT_DIR / ".env"
    if not env_path.exists():
        raise FileNotFoundError("No existe .env. Crea uno usando .env.example.")

    values: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if not line or line.strip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()

    required_keys = [
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
        "POSTGRES_CONTAINER_NAME",
    ]
    missing_keys = [key for key in required_keys if not values.get(key)]
    if missing_keys:
        raise ValueError(f"Faltan variables requeridas en .env: {', '.join(missing_keys)}")
    return values


def run_command(command: list[str], env: dict[str, str] | None = None) -> str:
    """Ejecuta comandos de Docker/Postgres y falla si algo sale mal."""
    completed = subprocess.run(
        command,
        cwd=ROOT_DIR,
        env={**os.environ, **(env or {})},
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def run_psql(sql: str, env: dict[str, str]) -> str:
    """Ejecuta SQL dentro del contenedor de PostgreSQL."""
    return run_command(
        [
            "docker",
            "compose",
            "exec",
            "-T",
            "-e",
            "PGPASSWORD",
            "postgres",
            "psql",
            "-v",
            "ON_ERROR_STOP=1",
            "-U",
            env["POSTGRES_USER"],
            "-d",
            env["POSTGRES_DB"],
            "-c",
            sql,
        ],
        env={"PGPASSWORD": env["POSTGRES_PASSWORD"]},
    )


def start_postgres(env: dict[str, str]) -> None:
    """Levanta el contenedor y espera hasta que acepte conexiones."""
    run_command(["docker", "compose", "up", "-d", "postgres"], env=env)

    for _ in range(30):
        try:
            run_command(
                [
                    "docker",
                    "compose",
                    "exec",
                    "-T",
                    "postgres",
                    "pg_isready",
                    "-U",
                    env["POSTGRES_USER"],
                    "-d",
                    env["POSTGRES_DB"],
                ],
                env=env,
            )
            return
        except subprocess.CalledProcessError:
            time.sleep(2)
    raise TimeoutError("PostgreSQL no estuvo listo dentro del tiempo esperado.")


def reset_database(env: dict[str, str]) -> None:
    """Elimina esquemas de usuario para evitar mezclar cargas anteriores."""
    reset_sql = """
    DO $$
    DECLARE
        schema_record RECORD;
    BEGIN
        FOR schema_record IN
            SELECT schema_name
            FROM information_schema.schemata
            WHERE schema_name NOT IN ('information_schema', 'pg_catalog')
              AND schema_name NOT LIKE 'pg_toast%'
        LOOP
            EXECUTE format('DROP SCHEMA IF EXISTS %I CASCADE', schema_record.schema_name);
        END LOOP;
    END $$;

    CREATE SCHEMA public;
    CREATE SCHEMA analytics;
    GRANT ALL ON SCHEMA public TO public;
    """
    run_psql(reset_sql, env)


def create_schema(env: dict[str, str]) -> None:
    """Crea dimensiones, tabla de hechos y vista wide para lectura posterior."""
    schema_sql = """
    CREATE TABLE analytics.dim_country (
        country_code CHAR(3) PRIMARY KEY
    );

    CREATE TABLE analytics.dim_year (
        year INTEGER PRIMARY KEY
    );

    CREATE TABLE analytics.dim_indicator (
        indicator_code TEXT PRIMARY KEY,
        indicator_name TEXT NOT NULL,
        source TEXT NOT NULL,
        is_target BOOLEAN NOT NULL,
        is_transformed BOOLEAN NOT NULL,
        transformed_from TEXT
    );

    CREATE TABLE analytics.fact_indicator_value (
        country_code CHAR(3) NOT NULL REFERENCES analytics.dim_country(country_code),
        year INTEGER NOT NULL REFERENCES analytics.dim_year(year),
        indicator_code TEXT NOT NULL REFERENCES analytics.dim_indicator(indicator_code),
        value DOUBLE PRECISION,
        PRIMARY KEY (country_code, year, indicator_code)
    );
    """
    run_psql(schema_sql, env)


def build_indicator_metadata(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Construye metadatos de indicadores a partir del wide final."""
    source_metadata = pd.read_csv(METADATA_PATH) if METADATA_PATH.exists() else pd.DataFrame()
    metadata_by_code = {
        row["indicator_code"]: row
        for _, row in source_metadata.iterrows()
        if "indicator_code" in source_metadata.columns
    }
    rows: list[dict[str, object]] = []

    for column in wide_df.columns:
        if column in {"country_code", "year"}:
            continue
        base_code = column.replace("_log1p", "")
        base_metadata = metadata_by_code.get(base_code, {})
        is_transformed = column.endswith("_log1p")
        indicator_name = base_metadata.get("indicator_name", column)
        if is_transformed:
            indicator_name = f"{indicator_name} (log1p)"

        rows.append(
            {
                "indicator_code": column,
                "indicator_name": indicator_name,
                "source": base_metadata.get("source", "derived"),
                "is_target": column == TARGET_COLUMN,
                "is_transformed": is_transformed,
                "transformed_from": base_code if is_transformed else "",
            }
        )
    return pd.DataFrame(rows)


def write_load_files(wide_df: pd.DataFrame) -> dict[str, Path]:
    """Genera CSV temporales para COPY hacia PostgreSQL."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    indicator_columns = [column for column in wide_df.columns if column not in {"country_code", "year"}]

    paths = {
        "countries": TEMP_DIR / "dim_country.csv",
        "years": TEMP_DIR / "dim_year.csv",
        "indicators": TEMP_DIR / "dim_indicator.csv",
        "facts": TEMP_DIR / "fact_indicator_value.csv",
    }

    wide_df[["country_code"]].drop_duplicates().sort_values("country_code").to_csv(
        paths["countries"], index=False
    )
    wide_df[["year"]].drop_duplicates().sort_values("year").to_csv(paths["years"], index=False)
    build_indicator_metadata(wide_df).to_csv(paths["indicators"], index=False)

    fact_df = wide_df.melt(
        id_vars=["country_code", "year"],
        value_vars=indicator_columns,
        var_name="indicator_code",
        value_name="value",
    )
    fact_df.to_csv(paths["facts"], index=False, quoting=csv.QUOTE_MINIMAL)
    return paths


def copy_files_to_container(paths: dict[str, Path], env: dict[str, str]) -> None:
    """Copia los CSV temporales al contenedor para usar COPY server-side."""
    for path in paths.values():
        run_command(
            [
                "docker",
                "cp",
                str(path),
                f"{env['POSTGRES_CONTAINER_NAME']}:/tmp/{path.name}",
            ],
            env=env,
        )


def load_tables(env: dict[str, str]) -> None:
    """Carga dimensiones y hechos usando COPY."""
    copy_sql = """
    COPY analytics.dim_country(country_code)
    FROM '/tmp/dim_country.csv'
    WITH (FORMAT csv, HEADER true);

    COPY analytics.dim_year(year)
    FROM '/tmp/dim_year.csv'
    WITH (FORMAT csv, HEADER true);

    COPY analytics.dim_indicator(
        indicator_code,
        indicator_name,
        source,
        is_target,
        is_transformed,
        transformed_from
    )
    FROM '/tmp/dim_indicator.csv'
    WITH (FORMAT csv, HEADER true);

    COPY analytics.fact_indicator_value(country_code, year, indicator_code, value)
    FROM '/tmp/fact_indicator_value.csv'
    WITH (FORMAT csv, HEADER true, NULL '');
    """
    run_psql(copy_sql, env)


def create_model_view(wide_df: pd.DataFrame, env: dict[str, str]) -> None:
    """Crea una vista wide para lectura directa desde JDBC."""
    indicator_columns = [column for column in wide_df.columns if column not in {"country_code", "year"}]
    pivot_columns = [
        (
            f"MAX(value) FILTER (WHERE indicator_code = '{column}') "
            f'AS "{column}"'
        )
        for column in indicator_columns
    ]
    view_sql = f"""
    CREATE OR REPLACE VIEW analytics.v_model_panel AS
    SELECT
        country_code,
        year,
        {", ".join(pivot_columns)}
    FROM analytics.fact_indicator_value
    GROUP BY country_code, year
    ORDER BY country_code, year;
    """
    run_psql(view_sql, env)


def verify_load(env: dict[str, str]) -> None:
    """Verifica conexion y lectura desde las tablas cargadas."""
    checks_sql = """
    SELECT 'dim_country' AS object_name, COUNT(*) AS row_count FROM analytics.dim_country
    UNION ALL
    SELECT 'dim_year', COUNT(*) FROM analytics.dim_year
    UNION ALL
    SELECT 'dim_indicator', COUNT(*) FROM analytics.dim_indicator
    UNION ALL
    SELECT 'fact_indicator_value', COUNT(*) FROM analytics.fact_indicator_value
    UNION ALL
    SELECT 'v_model_panel', COUNT(*) FROM analytics.v_model_panel;

    SELECT * FROM analytics.v_model_panel ORDER BY country_code, year LIMIT 5;
    """
    print(run_psql(checks_sql, env))


def main() -> None:
    env = load_env()
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"No existe el dataset limpio: {DATASET_PATH}")

    wide_df = pd.read_csv(DATASET_PATH)
    start_postgres(env)
    reset_database(env)
    create_schema(env)
    paths = write_load_files(wide_df)
    copy_files_to_container(paths, env)
    load_tables(env)
    create_model_view(wide_df, env)
    verify_load(env)


if __name__ == "__main__":
    main()
