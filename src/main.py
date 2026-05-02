"""Orquestador de extraccion de datos WHO y World Bank."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Permite ejecutar "python src/main.py" sin ajustar PYTHONPATH manualmente.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config.indicators import WORLD_BANK_INDICATORS, WHO_INDICATORS
from src.extractors.who_extractor import extract_who_indicator
from src.extractors.world_bank_extractor import extract_world_bank_indicator
from src.utils.file_writer import write_csv
from src.utils.logging import get_logger

WHO_OUTPUT_DIR = ROOT_DIR / "data" / "raw" / "who"
WORLD_BANK_OUTPUT_DIR = ROOT_DIR / "data" / "raw" / "world_bank"


def _run_who_pipeline(logger) -> None:
    for indicator_code, metadata in WHO_INDICATORS.items():
        indicator_name = metadata["indicator_name"]
        logger.info("WHO | Iniciando extraccion de %s", indicator_code)
        try:
            df = extract_who_indicator(
                indicator_code=indicator_code,
                indicator_name=indicator_name,
            )
            _persist_indicator_df(df, WHO_OUTPUT_DIR, indicator_code)
            logger.info(
                "WHO | Exito %s | registros=%s",
                indicator_code,
                len(df),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("WHO | Error %s | %s", indicator_code, exc)


def _run_world_bank_pipeline(logger) -> None:
    for indicator_code, metadata in WORLD_BANK_INDICATORS.items():
        indicator_name = metadata["indicator_name"]
        logger.info("WORLD_BANK | Iniciando extraccion de %s", indicator_code)
        try:
            df = extract_world_bank_indicator(
                indicator_code=indicator_code,
                indicator_name=indicator_name,
            )
            _persist_indicator_df(df, WORLD_BANK_OUTPUT_DIR, indicator_code)
            logger.info(
                "WORLD_BANK | Exito %s | registros=%s",
                indicator_code,
                len(df),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("WORLD_BANK | Error %s | %s", indicator_code, exc)


def _persist_indicator_df(
    df: pd.DataFrame,
    output_dir: Path,
    indicator_code: str,
) -> None:
    output_path = output_dir / f"{indicator_code}.csv"
    write_csv(df, output_path)


def main() -> None:
    load_dotenv()
    logger = get_logger()
    logger.info("Inicio de pipeline de extraccion")
    _run_who_pipeline(logger)
    _run_world_bank_pipeline(logger)
    logger.info("Fin de pipeline de extraccion")


if __name__ == "__main__":
    main()
