"""EDA unificado para datasets CSV de WHO y World Bank.

El script:
- carga todos los CSV con esquema long desde `data/raw`
- valida la estructura esperada
- unifica los datasets sin eliminar nulos
- pivotea a formato wide usando `country_code` y `year` como llave
- genera tablas de EDA y graficos en disco
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

EXPECTED_COLUMNS = [
    "country_code",
    "year",
    "indicator_code",
    "indicator_name",
    "source",
    "value",
]
PRIMARY_KEY = ["country_code", "year"]
RAW_SOURCE_FOLDERS = {"who", "world_bank"}
TARGET_COLUMN = "NCDMORT3070"
DROP_COLUMNS_FOR_MODEL = ["WHOSIS_000001"]
LOG1P_TRANSFORM_COLUMNS = ["NY.GDP.PCAP.CD", "TB_e_inc_num"]
ISO3_COUNTRY_CODES = {
    "ABW", "AFG", "AGO", "AIA", "ALA", "ALB", "AND", "ARE", "ARG", "ARM",
    "ASM", "ATA", "ATF", "ATG", "AUS", "AUT", "AZE", "BDI", "BEL", "BEN",
    "BES", "BFA", "BGD", "BGR", "BHR", "BHS", "BIH", "BLM", "BLR", "BLZ",
    "BMU", "BOL", "BRA", "BRB", "BRN", "BTN", "BVT", "BWA", "CAF", "CAN",
    "CCK", "CHE", "CHL", "CHN", "CIV", "CMR", "COD", "COG", "COK", "COL",
    "COM", "CPV", "CRI", "CUB", "CUW", "CXR", "CYM", "CYP", "CZE", "DEU",
    "DJI", "DMA", "DNK", "DOM", "DZA", "ECU", "EGY", "ERI", "ESH", "ESP",
    "EST", "ETH", "FIN", "FJI", "FLK", "FRA", "FRO", "FSM", "GAB", "GBR",
    "GEO", "GGY", "GHA", "GIB", "GIN", "GLP", "GMB", "GNB", "GNQ", "GRC",
    "GRD", "GRL", "GTM", "GUF", "GUM", "GUY", "HKG", "HMD", "HND", "HRV",
    "HTI", "HUN", "IDN", "IMN", "IND", "IOT", "IRL", "IRN", "IRQ", "ISL",
    "ISR", "ITA", "JAM", "JEY", "JOR", "JPN", "KAZ", "KEN", "KGZ", "KHM",
    "KIR", "KNA", "KOR", "KWT", "LAO", "LBN", "LBR", "LBY", "LCA", "LIE",
    "LKA", "LSO", "LTU", "LUX", "LVA", "MAC", "MAF", "MAR", "MCO", "MDA",
    "MDG", "MDV", "MEX", "MHL", "MKD", "MLI", "MLT", "MMR", "MNE", "MNG",
    "MNP", "MOZ", "MRT", "MSR", "MTQ", "MUS", "MWI", "MYS", "MYT", "NAM",
    "NCL", "NER", "NFK", "NGA", "NIC", "NIU", "NLD", "NOR", "NPL", "NRU",
    "NZL", "OMN", "PAK", "PAN", "PCN", "PER", "PHL", "PLW", "PNG", "POL",
    "PRI", "PRK", "PRT", "PRY", "PSE", "PYF", "QAT", "REU", "ROU", "RUS",
    "RWA", "SAU", "SDN", "SEN", "SGP", "SGS", "SHN", "SJM", "SLB", "SLE",
    "SLV", "SMR", "SOM", "SPM", "SRB", "SSD", "STP", "SUR", "SVK", "SVN",
    "SWE", "SWZ", "SXM", "SYC", "SYR", "TCA", "TCD", "TGO", "THA", "TJK",
    "TKL", "TKM", "TLS", "TON", "TTO", "TUN", "TUR", "TUV", "TWN", "TZA",
    "UGA", "UKR", "UMI", "URY", "USA", "UZB", "VAT", "VCT", "VEN", "VGB",
    "VIR", "VNM", "VUT", "WLF", "WSM", "YEM", "ZAF", "ZMB", "ZWE",
}


def configure_logger() -> logging.Logger:
    """Configura un logger simple para seguir la ejecucion del EDA."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("eda")


def parse_args() -> argparse.Namespace:
    """Define argumentos simples para reutilizar el script en distintas rutas."""
    parser = argparse.ArgumentParser(
        description="Analisis exploratorio de datos para WHO y World Bank.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directorio raiz donde viven los CSV crudos.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/eda"),
        help="Directorio donde se guardan tablas, dataset unificado y graficos.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Cantidad base de bins para histogramas.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2000,
        help="Primer ano incluido en el panel unificado.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2024,
        help="Ultimo ano incluido en el panel unificado.",
    )
    parser.add_argument(
        "--include-aggregates",
        action="store_true",
        help="Incluye agregados regionales o economicos ademas de paises ISO3.",
    )
    return parser.parse_args()


def discover_csv_files(raw_dir: Path) -> list[Path]:
    """Busca todos los CSV de WHO y World Bank para incluir ambas fuentes."""
    csv_files = sorted(
        file_path
        for file_path in raw_dir.rglob("*.csv")
        if file_path.parent.name in RAW_SOURCE_FOLDERS
    )
    if not csv_files:
        raise FileNotFoundError(
            f"No se encontraron CSV en '{raw_dir}'. "
            "Ejecuta primero la extraccion o apunta a otra ruta con --raw-dir."
        )
    return csv_files


def load_single_csv(file_path: Path) -> pd.DataFrame:
    """Carga un CSV y fuerza el esquema esperado sin eliminar registros."""
    df = pd.read_csv(file_path, low_memory=False)
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"El archivo '{file_path}' no contiene las columnas requeridas: "
            f"{', '.join(missing_columns)}"
        )

    extra_columns = [column for column in df.columns if column not in EXPECTED_COLUMNS]
    normalized = df[EXPECTED_COLUMNS + extra_columns].copy()

    # Normalizamos tipos porque el EDA depende de llaves y medidas consistentes.
    normalized["country_code"] = (
        normalized["country_code"].astype("string").str.strip().str.upper()
    )
    normalized["indicator_code"] = (
        normalized["indicator_code"].astype("string").str.strip()
    )
    normalized["indicator_name"] = (
        normalized["indicator_name"].astype("string").str.strip()
    )
    normalized["source"] = normalized["source"].astype("string").str.strip()
    normalized["year"] = pd.to_numeric(normalized["year"], errors="coerce")
    normalized["value"] = pd.to_numeric(normalized["value"], errors="coerce")
    return normalized


def filter_year_range(
    long_df: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Filtra el panel al rango temporal definido para el analisis reciente."""
    filtered = long_df.loc[
        long_df["year"].between(start_year, end_year, inclusive="both")
    ].copy()
    filtered["year"] = filtered["year"].astype(int)
    return filtered


def filter_iso3_countries(long_df: pd.DataFrame) -> pd.DataFrame:
    """Conserva solo codigos ISO3 de paises y territorios."""
    return long_df.loc[long_df["country_code"].isin(ISO3_COUNTRY_CODES)].copy()


def build_country_filter_summary(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compara cobertura antes y despues de excluir agregados no ISO3."""
    before_rows = len(before_df)
    after_rows = len(after_df)
    before_countries = before_df["country_code"].nunique(dropna=True)
    after_countries = after_df["country_code"].nunique(dropna=True)
    removed_country_codes = sorted(
        set(before_df["country_code"].dropna()) - set(after_df["country_code"].dropna())
    )

    return pd.DataFrame(
        [
            {
                "metric": "rows_before_iso3_filter",
                "value": before_rows,
            },
            {
                "metric": "rows_after_iso3_filter",
                "value": after_rows,
            },
            {
                "metric": "rows_removed",
                "value": before_rows - after_rows,
            },
            {
                "metric": "rows_removed_percentage",
                "value": round((before_rows - after_rows) / before_rows * 100, 2)
                if before_rows
                else 0,
            },
            {
                "metric": "country_codes_before_iso3_filter",
                "value": before_countries,
            },
            {
                "metric": "country_codes_after_iso3_filter",
                "value": after_countries,
            },
            {
                "metric": "country_codes_removed",
                "value": before_countries - after_countries,
            },
            {
                "metric": "removed_country_codes",
                "value": " | ".join(removed_country_codes),
            },
        ]
    )


def load_and_unify_datasets(csv_files: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatena todos los CSV en formato long y conserva metadatos por archivo."""
    dataframes: list[pd.DataFrame] = []
    file_summaries: list[dict[str, object]] = []

    for file_path in csv_files:
        df = load_single_csv(file_path)
        dataframes.append(df)
        file_summaries.append(
            {
                "file_name": file_path.name,
                "source_folder": file_path.parent.name,
                "rows": len(df),
                "null_values": int(df["value"].isna().sum()),
                "min_year": df["year"].min(),
                "max_year": df["year"].max(),
                "indicator_codes": int(df["indicator_code"].nunique()),
            }
        )

    long_df = pd.concat(dataframes, ignore_index=True).sort_values(
        ["source", "indicator_code", "country_code", "year"],
        na_position="last",
    )
    return long_df, pd.DataFrame(file_summaries)


def build_indicator_metadata(long_df: pd.DataFrame) -> pd.DataFrame:
    """Resume metadatos de indicadores para interpretar mejor las columnas wide."""
    metadata = (
        long_df[["indicator_code", "indicator_name", "source"]]
        .drop_duplicates()
        .sort_values(["source", "indicator_code"])
        .reset_index(drop=True)
    )
    return metadata


def build_general_summary(long_df: pd.DataFrame, csv_files: list[Path]) -> pd.DataFrame:
    """Resume tamano y cobertura global para una lectura rapida del dataset."""
    summary = pd.DataFrame(
        [
            {
                "metric": "total_files",
                "value": len(csv_files),
            },
            {
                "metric": "total_rows",
                "value": len(long_df),
            },
            {
                "metric": "total_columns",
                "value": long_df.shape[1],
            },
            {
                "metric": "unique_country_codes",
                "value": long_df["country_code"].nunique(dropna=True),
            },
            {
                "metric": "unique_years",
                "value": long_df["year"].nunique(dropna=True),
            },
            {
                "metric": "unique_indicator_codes",
                "value": long_df["indicator_code"].nunique(dropna=True),
            },
            {
                "metric": "unique_indicator_names",
                "value": long_df["indicator_name"].nunique(dropna=True),
            },
            {
                "metric": "unique_sources",
                "value": long_df["source"].nunique(dropna=True),
            },
            {
                "metric": "year_min",
                "value": long_df["year"].min(),
            },
            {
                "metric": "year_max",
                "value": long_df["year"].max(),
            },
        ]
    )
    return summary


def build_value_statistics(long_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula descriptivos basicos sobre la variable numerica de cada indicador."""
    stats_rows: list[dict[str, object]] = []

    for (indicator_code, indicator_name, source), group in long_df.groupby(
        ["indicator_code", "indicator_name", "source"],
        dropna=False,
    ):
        series = group["value"]
        stats_rows.append(
            {
                "indicator_code": indicator_code,
                "indicator_name": indicator_name,
                "source": source,
                "total_rows": len(group),
                "non_null_values": int(series.notna().sum()),
                "null_values": int(series.isna().sum()),
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "p25": series.quantile(0.25),
                "median": series.median(),
                "p75": series.quantile(0.75),
                "max": series.max(),
            }
        )

    return pd.DataFrame(stats_rows).sort_values(["source", "indicator_code"])


def build_nulls_by_indicator(long_df: pd.DataFrame) -> pd.DataFrame:
    """Mide faltantes por indicador antes del merge para ver la calidad base."""
    grouped = (
        long_df.groupby(["indicator_code", "indicator_name", "source"], dropna=False)
        .agg(
            total_rows=("value", "size"),
            non_null_values=("value", lambda series: int(series.notna().sum())),
            null_values=("value", lambda series: int(series.isna().sum())),
        )
        .reset_index()
    )
    grouped["null_percentage"] = (
        grouped["null_values"] / grouped["total_rows"] * 100
    ).round(2)
    return grouped.sort_values("null_percentage", ascending=False)


def build_coverage_by_country(long_df: pd.DataFrame) -> pd.DataFrame:
    """Cuenta cobertura por pais para identificar paneles mas y menos completos."""
    return (
        long_df.groupby("country_code", dropna=False)
        .agg(
            row_count=("country_code", "size"),
            indicator_count=("indicator_code", "nunique"),
            year_count=("year", "nunique"),
            source_count=("source", "nunique"),
        )
        .reset_index()
        .sort_values(["indicator_count", "row_count"], ascending=[False, False])
    )


def build_coverage_by_year(long_df: pd.DataFrame) -> pd.DataFrame:
    """Cuenta cobertura por ano para detectar huecos temporales entre fuentes."""
    return (
        long_df.groupby("year", dropna=False)
        .agg(
            row_count=("year", "size"),
            country_count=("country_code", "nunique"),
            indicator_count=("indicator_code", "nunique"),
        )
        .reset_index()
        .sort_values("year")
    )


def build_coverage_by_indicator(long_df: pd.DataFrame) -> pd.DataFrame:
    """Resume la cobertura de cada indicador antes de pivotear a formato wide."""
    coverage = (
        long_df.groupby(["indicator_code", "indicator_name", "source"], dropna=False)
        .agg(
            row_count=("indicator_code", "size"),
            country_count=("country_code", "nunique"),
            year_count=("year", "nunique"),
            null_values=("value", lambda series: int(series.isna().sum())),
        )
        .reset_index()
    )
    coverage["null_percentage"] = (
        coverage["null_values"] / coverage["row_count"] * 100
    ).round(2)
    return coverage.sort_values(["source", "indicator_code"])


def build_duplicate_key_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    """Detecta duplicados por llave ampliada porque afectan el pivot y el merge."""
    duplicate_counts = (
        long_df.groupby(PRIMARY_KEY + ["indicator_code"], dropna=False)
        .size()
        .reset_index(name="duplicate_count")
    )
    duplicates = duplicate_counts.loc[duplicate_counts["duplicate_count"] > 1].copy()
    if duplicates.empty:
        return pd.DataFrame(
            [
                {
                    "indicator_code": None,
                    "duplicate_keys": 0,
                    "max_duplicate_count": 0,
                }
            ]
        )

    return (
        duplicates.groupby("indicator_code", dropna=False)
        .agg(
            duplicate_keys=("duplicate_count", "size"),
            max_duplicate_count=("duplicate_count", "max"),
        )
        .reset_index()
        .sort_values("duplicate_keys", ascending=False)
    )


def build_duplicate_dimension_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    """Resume dimensiones WHO asociadas a duplicados por llave analitica."""
    dimension_columns = [
        column
        for column in [
            "Dim1Type",
            "Dim1",
            "Dim2Type",
            "Dim2",
            "Dim3Type",
            "Dim3",
            "DataSourceDimType",
            "DataSourceDim",
        ]
        if column in long_df.columns
    ]
    if not dimension_columns:
        return pd.DataFrame()

    duplicate_mask = long_df.duplicated(PRIMARY_KEY + ["indicator_code"], keep=False)
    duplicate_rows = long_df.loc[duplicate_mask].copy()
    if duplicate_rows.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    grouped = duplicate_rows.groupby(
        ["indicator_code", "indicator_name", "source"],
        dropna=False,
    )
    for (indicator_code, indicator_name, source), group in grouped:
        row: dict[str, object] = {
            "indicator_code": indicator_code,
            "indicator_name": indicator_name,
            "source": source,
            "duplicate_rows": len(group),
            "duplicate_keys": group[PRIMARY_KEY].drop_duplicates().shape[0],
        }
        for column in dimension_columns:
            values = sorted(str(value) for value in group[column].dropna().unique())
            row[f"{column}_values"] = " | ".join(values[:20])
        rows.append(row)

    return pd.DataFrame(rows).sort_values("duplicate_keys", ascending=False)


def build_canonical_long_dataset(long_df: pd.DataFrame) -> pd.DataFrame:
    """Selecciona una observacion por llave cuando WHO trae dimensiones multiples."""
    scored_df = long_df.copy()
    scored_df["_preference_score"] = 0

    # Para indicadores por sexo, priorizamos ambos sexos como serie nacional total.
    if "Dim1Type" in scored_df.columns and "Dim1" in scored_df.columns:
        has_sex_dimension = scored_df["Dim1Type"].astype("string").str.upper().eq("SEX")
        is_both_sexes = scored_df["Dim1"].astype("string").str.upper().isin(
            ["SEX_BTSX", "BTSX", "BOTHSEX", "BOTH_SEXES"]
        )
        scored_df.loc[has_sex_dimension & ~is_both_sexes, "_preference_score"] += 10

    # Preferimos observaciones con valor numerico porque alimentan el panel wide.
    scored_df.loc[scored_df["value"].isna(), "_preference_score"] += 100
    scored_df = scored_df.sort_values(
        PRIMARY_KEY + ["indicator_code", "_preference_score"],
        na_position="last",
    )
    canonical_df = scored_df.drop_duplicates(
        PRIMARY_KEY + ["indicator_code"],
        keep="first",
    ).drop(columns=["_preference_score"])
    return canonical_df


def build_wide_dataset(long_df: pd.DataFrame) -> pd.DataFrame:
    """Construye el panel wide por `country_code` y `year` manteniendo nulls."""
    base_keys = long_df[PRIMARY_KEY].drop_duplicates().sort_values(PRIMARY_KEY)
    wide_df = base_keys.copy()

    # Hacemos merge columna por columna para conservar llaves aunque el valor sea nulo.
    for indicator_code, group in long_df.groupby("indicator_code", dropna=False):
        indicator_values = (
            group.groupby(PRIMARY_KEY, dropna=False, as_index=False)["value"]
            .mean()
            .rename(columns={"value": indicator_code})
        )
        wide_df = wide_df.merge(indicator_values, on=PRIMARY_KEY, how="left")

    return wide_df.sort_values(PRIMARY_KEY).reset_index(drop=True)


def prepare_model_wide_dataset(wide_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica reglas simples de preparacion para el panel modelable."""
    prepared_df = wide_df.drop(columns=DROP_COLUMNS_FOR_MODEL, errors="ignore").copy()
    predictor_columns = [
        column
        for column in prepared_df.columns
        if column not in PRIMARY_KEY and column != TARGET_COLUMN
    ]
    before_missing = prepared_df[predictor_columns].isna().sum().sum()

    # Interpolamos dentro de cada pais para respetar la evolucion temporal nacional.
    prepared_df = prepared_df.sort_values(PRIMARY_KEY).reset_index(drop=True)
    prepared_df[predictor_columns] = prepared_df.groupby("country_code")[
        predictor_columns
    ].transform(lambda group: group.interpolate(method="linear", limit_area="inside"))

    # Los huecos restantes se cubren con la mediana anual del indicador.
    for column in predictor_columns:
        prepared_df[column] = prepared_df[column].fillna(
            prepared_df.groupby("year")[column].transform("median")
        )
    after_year_median_missing = prepared_df[predictor_columns].isna().sum().sum()

    # Si un ano completo no tiene datos, no existe mediana anual; usamos mediana global.
    for column in predictor_columns:
        prepared_df[column] = prepared_df[column].fillna(prepared_df[column].median())

    for column in LOG1P_TRANSFORM_COLUMNS:
        if column in prepared_df.columns:
            prepared_df[f"{column}_log1p"] = np.log1p(prepared_df[column])
            prepared_df = prepared_df.drop(columns=[column])

    after_columns = [
        column
        for column in prepared_df.columns
        if column not in PRIMARY_KEY and column != TARGET_COLUMN
    ]
    after_missing = prepared_df[after_columns].isna().sum().sum()
    target_missing = (
        int(prepared_df[TARGET_COLUMN].isna().sum())
        if TARGET_COLUMN in prepared_df.columns
        else 0
    )
    summary = pd.DataFrame(
        [
            {"metric": "year_max_after_filter", "value": int(prepared_df["year"].max())},
            {"metric": "dropped_columns", "value": " | ".join(DROP_COLUMNS_FOR_MODEL)},
            {"metric": "target_not_imputed", "value": TARGET_COLUMN},
            {"metric": "predictor_missing_before_imputation", "value": int(before_missing)},
            {
                "metric": "predictor_missing_after_year_median",
                "value": int(after_year_median_missing),
            },
            {"metric": "predictor_missing_after_final_fallback", "value": int(after_missing)},
            {"metric": "target_missing_after_preparation", "value": target_missing},
            {
                "metric": "log1p_transformed_columns",
                "value": " | ".join(LOG1P_TRANSFORM_COLUMNS),
            },
        ]
    )
    return prepared_df, summary


def build_missing_by_variable_wide(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Mide faltantes por variable tras el merge para ver la perdida de cobertura."""
    indicator_columns = [column for column in wide_df.columns if column not in PRIMARY_KEY]
    rows: list[dict[str, object]] = []

    for column in indicator_columns:
        null_count = int(wide_df[column].isna().sum())
        rows.append(
            {
                "indicator_code": column,
                "row_count": len(wide_df),
                "non_null_values": int(wide_df[column].notna().sum()),
                "null_values": null_count,
                "null_percentage": round(null_count / len(wide_df) * 100, 2),
            }
        )

    return pd.DataFrame(rows).sort_values("null_percentage", ascending=False)


def build_row_missingness_summary(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Resume cuantas variables faltan por fila para medir completitud del panel."""
    indicator_columns = [column for column in wide_df.columns if column not in PRIMARY_KEY]
    row_missing_count = wide_df[indicator_columns].isna().sum(axis=1)
    row_missing_percentage = row_missing_count / len(indicator_columns) * 100
    summary = pd.DataFrame(
        [
            {
                "metric": "rows_in_panel",
                "value": len(wide_df),
            },
            {
                "metric": "indicator_columns",
                "value": len(indicator_columns),
            },
            {
                "metric": "rows_without_missing_values",
                "value": int((row_missing_count == 0).sum()),
            },
            {
                "metric": "rows_with_any_missing_values",
                "value": int((row_missing_count > 0).sum()),
            },
            {
                "metric": "mean_missing_variables_per_row",
                "value": round(row_missing_count.mean(), 2),
            },
            {
                "metric": "median_missing_variables_per_row",
                "value": round(row_missing_count.median(), 2),
            },
            {
                "metric": "mean_missing_percentage_per_row",
                "value": round(row_missing_percentage.mean(), 2),
            },
            {
                "metric": "median_missing_percentage_per_row",
                "value": round(row_missing_percentage.median(), 2),
            },
        ]
    )
    return summary


def build_missingness_by_year_indicator(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula porcentaje de faltantes por ano e indicador para el heatmap."""
    indicator_columns = [column for column in wide_df.columns if column not in PRIMARY_KEY]
    missingness = (
        wide_df.groupby("year")[indicator_columns]
        .apply(lambda group: group.isna().mean() * 100)
        .sort_index()
    )
    return missingness


def build_outlier_tables(wide_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica IQR por variable porque es una regla robusta para un primer EDA."""
    indicator_columns = [column for column in wide_df.columns if column not in PRIMARY_KEY]
    summary_rows: list[dict[str, object]] = []
    flagged_rows: list[pd.DataFrame] = []

    for column in indicator_columns:
        series = wide_df[column].dropna()
        if series.empty:
            summary_rows.append(
                {
                    "indicator_code": column,
                    "non_null_values": 0,
                    "q1": np.nan,
                    "q3": np.nan,
                    "iqr": np.nan,
                    "lower_bound": np.nan,
                    "upper_bound": np.nan,
                    "outlier_count": 0,
                    "outlier_percentage": 0.0,
                }
            )
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = wide_df[column].notna() & (
            (wide_df[column] < lower_bound) | (wide_df[column] > upper_bound)
        )

        outlier_count = int(outlier_mask.sum())
        summary_rows.append(
            {
                "indicator_code": column,
                "non_null_values": int(series.shape[0]),
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": outlier_count,
                "outlier_percentage": round(outlier_count / series.shape[0] * 100, 2),
            }
        )

        if outlier_count > 0:
            flagged = wide_df.loc[outlier_mask, PRIMARY_KEY + [column]].copy()
            flagged["indicator_code"] = column
            flagged = flagged.rename(columns={column: "value"})
            flagged_rows.append(flagged)

    outliers_summary = pd.DataFrame(summary_rows).sort_values(
        "outlier_percentage",
        ascending=False,
    )
    outlier_rows = (
        pd.concat(flagged_rows, ignore_index=True)
        if flagged_rows
        else pd.DataFrame(columns=PRIMARY_KEY + ["value", "indicator_code"])
    )
    return outliers_summary, outlier_rows


def compute_correlation_matrix(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula correlaciones entre variables del panel wide usando Pearson."""
    indicator_columns = [column for column in wide_df.columns if column not in PRIMARY_KEY]
    return wide_df[indicator_columns].corr()


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    """Guarda tablas en CSV porque son faciles de inspeccionar y versionar."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def plot_distributions(
    wide_df: pd.DataFrame,
    indicator_metadata: pd.DataFrame,
    output_dir: Path,
    bins: int,
) -> None:
    """Genera histogramas y boxplots usando seaborn para revisar forma y dispersion."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_map = indicator_metadata.set_index("indicator_code")["indicator_name"].to_dict()

    for column in [value for value in wide_df.columns if value not in PRIMARY_KEY]:
        series = wide_df[column].dropna()
        if series.empty:
            continue
            
        indicator_name = metadata_map.get(column, column)

        fig, (ax_box, ax_hist) = plt.subplots(
            2, 1,
            sharex=True,
            gridspec_kw={"height_ratios": [0.15, 0.85]},
            figsize=(10, 6)
        )

        sns.boxplot(x=series, ax=ax_box, color="#A0CBE8", flierprops={"marker": "o", "markerfacecolor": "#E45756"})
        sns.histplot(x=series, ax=ax_hist, bins=bins, color="#4C78A8", kde=True)

        ax_box.set(xlabel="")
        ax_box.set_title(f"{column} | {indicator_name}\nn={len(series)} | media={series.mean():.2f} | mediana={series.median():.2f}", loc="left")
        ax_hist.set(xlabel="Valor", ylabel="Frecuencia")

        plt.tight_layout()
        plt.savefig(output_dir / f"{column}_distribution.png")
        plt.close()


def plot_correlation_heatmap(correlation_df: pd.DataFrame, output_path: Path) -> None:
    """Guarda un heatmap de correlacion usando seaborn para detectar relaciones lineales."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_df,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Correlación de Pearson"}
    )
    plt.title("Heatmap de correlación entre indicadores")
    plt.tight_layout()
    output_path_png = output_path.with_suffix(".png")
    plt.savefig(output_path_png)
    plt.close()


def plot_missingness_heatmap(missingness_df: pd.DataFrame, output_path: Path) -> None:
    """Visualiza faltantes por ano e indicador con un heatmap de seaborn."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        missingness_df.T,
        cmap="YlOrRd",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "% de Faltantes"}
    )
    plt.title("Porcentaje de faltantes por año e indicador")
    plt.xlabel("Año")
    plt.ylabel("Indicador")
    plt.tight_layout()
    output_path_png = output_path.with_suffix(".png")
    plt.savefig(output_path_png)
    plt.close()


def build_report(
    general_summary: pd.DataFrame,
    missing_by_variable_wide: pd.DataFrame,
    outliers_summary: pd.DataFrame,
    correlation_df: pd.DataFrame,
    country_filter_summary: pd.DataFrame,
    model_preparation_summary: pd.DataFrame,
) -> str:
    """Construye un resumen corto para leer hallazgos clave sin abrir todos los CSV."""
    top_missing = missing_by_variable_wide.head(3)
    top_outliers = outliers_summary.head(3)

    correlation_pairs: list[tuple[str, str, float]] = []
    for row_name in correlation_df.index:
        for column_name in correlation_df.columns:
            if row_name >= column_name:
                continue
            value = correlation_df.loc[row_name, column_name]
            if pd.notna(value):
                correlation_pairs.append((row_name, column_name, float(value)))

    correlation_pairs.sort(key=lambda item: abs(item[2]), reverse=True)
    top_correlations = correlation_pairs[:5]

    lines = [
        "# Resumen EDA",
        "",
        "## Cobertura general",
    ]
    for _, row in general_summary.iterrows():
        lines.append(f"- {row['metric']}: {row['value']}")

    lines.extend(
        [
            "",
            "## Impacto del filtro ISO3",
        ]
    )
    for _, row in country_filter_summary.iterrows():
        lines.append(f"- {row['metric']}: {row['value']}")

    lines.extend(
        [
            "",
            "## Preparacion para modelo",
        ]
    )
    for _, row in model_preparation_summary.iterrows():
        lines.append(f"- {row['metric']}: {row['value']}")

    lines.extend(
        [
            "",
            "## Variables con mayor porcentaje de faltantes tras el merge",
        ]
    )
    for _, row in top_missing.iterrows():
        lines.append(
            f"- {row['indicator_code']}: {row['null_percentage']}% "
            f"({row['null_values']} nulos de {row['row_count']})"
        )

    lines.extend(
        [
            "",
            "## Variables con mayor porcentaje de outliers por IQR",
        ]
    )
    for _, row in top_outliers.iterrows():
        lines.append(
            f"- {row['indicator_code']}: {row['outlier_percentage']}% "
            f"({row['outlier_count']} outliers)"
        )

    lines.extend(
        [
            "",
            "## Correlaciones mas altas en valor absoluto",
        ]
    )
    for left, right, value in top_correlations:
        lines.append(f"- {left} vs {right}: {value:.3f}")

    return "\n".join(lines) + "\n"


def write_text(content: str, output_path: Path) -> None:
    """Guarda un reporte corto en texto para consulta rapida desde el repo."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def main() -> None:
    """Ejecuta todo el flujo de EDA y persiste tablas y graficos en disco."""
    args = parse_args()
    logger = configure_logger()

    output_dir = args.output_dir
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"

    logger.info("Descubriendo archivos CSV en %s", args.raw_dir)
    csv_files = discover_csv_files(args.raw_dir)

    logger.info("Cargando y unificando %s archivos", len(csv_files))
    long_df, file_summary = load_and_unify_datasets(csv_files)
    long_df = filter_year_range(long_df, args.start_year, args.end_year)
    country_filter_summary = build_country_filter_summary(
        before_df=long_df,
        after_df=filter_iso3_countries(long_df),
    )
    if not args.include_aggregates:
        long_df = filter_iso3_countries(long_df)

    logger.info("Construyendo dataset wide por country_code y year")
    canonical_long_df = build_canonical_long_dataset(long_df)
    raw_wide_df = build_wide_dataset(canonical_long_df)
    wide_df, model_preparation_summary = prepare_model_wide_dataset(raw_wide_df)
    indicator_metadata = build_indicator_metadata(long_df)

    logger.info("Calculando tablas de EDA")
    general_summary = build_general_summary(long_df, csv_files)
    value_statistics = build_value_statistics(long_df)
    nulls_by_indicator = build_nulls_by_indicator(long_df)
    coverage_by_country = build_coverage_by_country(long_df)
    coverage_by_year = build_coverage_by_year(long_df)
    coverage_by_indicator = build_coverage_by_indicator(long_df)
    duplicate_key_summary = build_duplicate_key_summary(long_df)
    duplicate_dimension_summary = build_duplicate_dimension_summary(long_df)
    missing_by_variable_wide = build_missing_by_variable_wide(wide_df)
    row_missingness_summary = build_row_missingness_summary(wide_df)
    missingness_by_year_indicator = build_missingness_by_year_indicator(wide_df)
    outliers_summary, outlier_rows = build_outlier_tables(wide_df)
    correlation_df = compute_correlation_matrix(wide_df)

    logger.info("Guardando datasets principales y tablas de EDA")
    save_dataframe(long_df, output_dir / "unified_long.csv")
    save_dataframe(canonical_long_df, output_dir / "unified_long_canonical.csv")
    save_dataframe(raw_wide_df, output_dir / "unified_wide_before_imputation.csv")
    save_dataframe(wide_df, output_dir / "unified_wide.csv")
    save_dataframe(country_filter_summary, output_dir / "country_filter_summary.csv")
    save_dataframe(model_preparation_summary, output_dir / "model_preparation_summary.csv")
    save_dataframe(file_summary, tables_dir / "file_summary.csv")
    save_dataframe(indicator_metadata, tables_dir / "indicator_metadata.csv")
    save_dataframe(general_summary, tables_dir / "general_summary.csv")
    save_dataframe(value_statistics, tables_dir / "value_statistics_by_indicator.csv")
    save_dataframe(nulls_by_indicator, tables_dir / "nulls_by_indicator_long.csv")
    save_dataframe(coverage_by_country, tables_dir / "coverage_by_country.csv")
    save_dataframe(coverage_by_year, tables_dir / "coverage_by_year.csv")
    save_dataframe(coverage_by_indicator, tables_dir / "coverage_by_indicator.csv")
    save_dataframe(duplicate_key_summary, tables_dir / "duplicate_key_summary.csv")
    save_dataframe(duplicate_dimension_summary, tables_dir / "duplicate_dimension_summary.csv")
    save_dataframe(missing_by_variable_wide, tables_dir / "missing_by_variable_wide.csv")
    save_dataframe(row_missingness_summary, tables_dir / "row_missingness_summary.csv")
    save_dataframe(
        missingness_by_year_indicator.reset_index(),
        tables_dir / "missingness_by_year_indicator.csv",
    )
    save_dataframe(outliers_summary, tables_dir / "outliers_summary.csv")
    save_dataframe(outlier_rows, tables_dir / "outlier_rows.csv")
    save_dataframe(correlation_df.reset_index(), tables_dir / "correlation_matrix.csv")

    logger.info("Generando graficos")
    plot_distributions(
        wide_df=wide_df,
        indicator_metadata=indicator_metadata,
        output_dir=plots_dir / "distributions",
        bins=args.bins,
    )
    plot_correlation_heatmap(
        correlation_df=correlation_df,
        output_path=plots_dir / "correlation_heatmap.png",
    )
    plot_missingness_heatmap(
        missingness_df=missingness_by_year_indicator,
        output_path=plots_dir / "missingness_heatmap_by_year_indicator.png",
    )

    logger.info("Imprimiendo reporte resumido")
    report = build_report(
        general_summary=general_summary,
        missing_by_variable_wide=missing_by_variable_wide,
        outliers_summary=outliers_summary,
        correlation_df=correlation_df,
        country_filter_summary=country_filter_summary,
        model_preparation_summary=model_preparation_summary,
    )
    write_text(report, output_dir / "eda_report.md")
    print("\n" + "="*50)
    print(report)
    print("="*50 + "\n")

    logger.info("EDA completado. Resultados principales disponibles en %s y gráficos en %s", output_dir, plots_dir)


if __name__ == "__main__":
    main()
