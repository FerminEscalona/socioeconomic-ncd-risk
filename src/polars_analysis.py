"""Análisis Distribuido con Polars y Visualizaciones Interactivas con Plotly."""

from __future__ import annotations

import os
import logging
from pathlib import Path

import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import pycountry_convert as pc
from dotenv import load_dotenv

# Configuración de rutas
ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "polars_eda"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Crear directorios si no existen
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("polars_analysis")


def get_db_uri() -> str:
    """Construye la URI de conexión a PostgreSQL desde el archivo .env."""
    load_dotenv(ROOT_DIR / ".env")
    
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_HOST_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "postgres")
    
    # URL de conexión para ConnectorX (postgresql://user:password@host:port/db)
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def get_continent(country_code_iso3: str) -> str:
    """Convierte un código ISO3 a nombre de continente."""
    try:
        iso2 = pc.country_alpha3_to_country_alpha2(country_code_iso3)
        continent_code = pc.country_alpha2_to_continent_code(iso2)
        continent_name = pc.convert_continent_code_to_continent_name(continent_code)
        return continent_name
    except:
        return "Unknown"


def load_data(uri: str) -> pl.DataFrame:
    """Carga los datos desde la vista en PostgreSQL usando Polars y ConnectorX."""
    logger.info("Conectando a PostgreSQL y leyendo analytics.v_model_panel...")
    query = "SELECT * FROM analytics.v_model_panel"
    
    # pl.read_database_uri es extremadamente rápido y optimizado para grandes volúmenes.
    df = pl.read_database_uri(query=query, uri=uri, engine="connectorx")
    logger.info(f"Datos cargados exitosamente: {df.height} filas, {df.width} columnas.")
    return df


def prepare_data(df: pl.DataFrame) -> pl.DataFrame:
    """Añade columnas derivadas como Región y Década usando operaciones vectorizadas."""
    logger.info("Enriqueciendo datos (Región y Década)...")
    
    # 1. Obtener los continentes y añadirlos al DataFrame de forma rápida
    unique_countries = df.select("country_code").unique().to_series().to_list()
    continent_map = {c: get_continent(c) for c in unique_countries}
    
    # Convertir el mapa a DataFrame para un join rápido
    df_continents = pl.DataFrame({
        "country_code": list(continent_map.keys()),
        "region": list(continent_map.values())
    })
    
    df_enriched = (
        df.join(df_continents, on="country_code", how="left")
        .with_columns([
            # Calcular la década
            ((pl.col("year") // 10) * 10).alias("decade")
        ])
    )
    
    return df_enriched


def calculate_descriptive_stats(df: pl.DataFrame) -> None:
    """Calcula y exporta estadísticas descriptivas agrupadas por Región y Década."""
    logger.info("Calculando estadísticas descriptivas...")
    
    # Seleccionar las columnas clave a analizar
    target_col = "NCDMORT3070"
    spending_col = "SH.XPD.CHEX.GD.ZS"
    
    stats_df = (
        df.group_by(["region", "decade"])
        .agg([
            # Mortalidad
            pl.col(target_col).mean().alias(f"{target_col}_mean"),
            pl.col(target_col).median().alias(f"{target_col}_median"),
            pl.col(target_col).std().alias(f"{target_col}_std"),
            # Gasto en Salud
            pl.col(spending_col).mean().alias(f"{spending_col}_mean"),
            pl.col(spending_col).median().alias(f"{spending_col}_median"),
            pl.col(spending_col).std().alias(f"{spending_col}_std"),
            # Conteo de observaciones
            pl.len().alias("count")
        ])
        .sort(["region", "decade"])
    )
    
    output_file = OUTPUT_DIR / "descriptive_stats_by_region_decade.parquet"
    stats_df.write_parquet(output_file)
    logger.info(f"Estadísticas exportadas a {output_file}")


def calculate_correlations(df: pl.DataFrame) -> None:
    """Calcula la correlación entre Gasto en Salud y Mortalidad por Región y Década."""
    logger.info("Calculando correlaciones...")
    
    # Asegurarnos de que no haya nulos para el cálculo de correlación
    df_clean = df.drop_nulls(subset=["NCDMORT3070", "SH.XPD.CHEX.GD.ZS"])
    
    # Correlación Global
    global_corr = df_clean.select(pl.corr("NCDMORT3070", "SH.XPD.CHEX.GD.ZS")).item()
    logger.info(f"Correlación de Pearson global (Gasto vs Mortalidad): {global_corr:.4f}")
    
    # Correlación por Región usando Polars groupby y agg
    # Polars < 0.20 permitía pearson_corr, ahora usamos corr
    corr_by_region = (
        df_clean.group_by("region")
        .agg(
            pl.corr("NCDMORT3070", "SH.XPD.CHEX.GD.ZS").alias("correlation")
        )
        .sort("correlation")
    )
    
    output_file = OUTPUT_DIR / "correlation_by_region.parquet"
    corr_by_region.write_parquet(output_file)
    logger.info(f"Correlaciones por región exportadas a {output_file}")


def generate_interactive_plots(df: pl.DataFrame) -> None:
    """Genera gráficas HTML interactivas espectaculares usando Plotly."""
    logger.info("Generando gráficas interactivas con Plotly...")
    pd_df = df.to_pandas()  # Plotly Express consume DataFrames de Pandas
    
    # Asegurarnos de que el DataFrame esté ordenado por año para animaciones fluidas
    pd_df = pd_df.sort_values(by=["year", "country_code"])
    
    # 1. Animación Scatter: Evolución de Gasto en Salud vs Mortalidad a través del tiempo
    fig_scatter = px.scatter(
        pd_df,
        x="SH.XPD.CHEX.GD.ZS",
        y="NCDMORT3070",
        animation_frame="year",
        animation_group="country_code",
        size="SP.DYN.LE00.IN", # Usar expectativa de vida como tamaño (si existe)
        color="region",
        hover_name="country_code",
        title="Evolución Interanual: Gasto en Salud (% PIB) vs Mortalidad (NCD)",
        labels={
            "SH.XPD.CHEX.GD.ZS": "Gasto en Salud (% del PIB)",
            "NCDMORT3070": "Mortalidad por Enf. No Transmisibles (%)",
            "region": "Continente"
        },
        range_x=[0, pd_df["SH.XPD.CHEX.GD.ZS"].max() * 1.1],
        range_y=[0, pd_df["NCDMORT3070"].max() * 1.1],
        template="plotly_dark"
    )
    fig_scatter.write_html(PLOTS_DIR / "animated_scatter_spending_vs_mortality.html")
    
    # 2. Mapa Choropleth: Distribución Mundial de la Mortalidad en el último año disponible
    max_year = pd_df["year"].max()
    pd_df_latest = pd_df[pd_df["year"] == max_year]
    
    fig_map = px.choropleth(
        pd_df_latest,
        locations="country_code",
        color="NCDMORT3070",
        hover_name="country_code",
        color_continuous_scale=px.colors.sequential.YlOrRd,
        title=f"Mortalidad por Enfermedades No Transmisibles ({max_year})",
        labels={"NCDMORT3070": "Mortalidad (%)"},
        template="plotly_dark"
    )
    fig_map.write_html(PLOTS_DIR / "choropleth_map_mortality_latest_year.html")
    
    # 3. Boxplot: Distribución de la Mortalidad por Región
    fig_box = px.box(
        pd_df,
        x="region",
        y="NCDMORT3070",
        color="region",
        points="all",
        title="Distribución de la Mortalidad (NCD) por Región Histórica",
        labels={
            "NCDMORT3070": "Mortalidad por Enf. No Transmisibles (%)",
            "region": "Región"
        },
        template="plotly_dark"
    )
    fig_box.write_html(PLOTS_DIR / "boxplot_mortality_by_region.html")
    
    logger.info("¡Gráficas HTML generadas con éxito!")


def main() -> None:
    uri = get_db_uri()
    
    try:
        df_raw = load_data(uri)
    except Exception as e:
        logger.error(f"Error al conectar o leer de PostgreSQL: {e}")
        logger.error("Asegúrate de que 'docker compose up -d postgres' esté corriendo.")
        return
        
    df_prepared = prepare_data(df_raw)
    
    # Ejecutar análisis en paralelo
    calculate_descriptive_stats(df_prepared)
    calculate_correlations(df_prepared)
    
    # Generar Visualizaciones
    generate_interactive_plots(df_prepared)
    
    logger.info("Análisis con Polars finalizado exitosamente. Revisa la carpeta data/processed/polars_eda/")


if __name__ == "__main__":
    main()
