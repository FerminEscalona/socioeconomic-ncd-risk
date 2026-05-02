"""Extractor para World Bank Open Data API."""

from __future__ import annotations

from typing import Any

import pandas as pd
import requests

WORLD_BANK_BASE_URL = "https://api.worldbank.org/v2"


def extract_world_bank_indicator(
    indicator_code: str,
    indicator_name: str,
    timeout_seconds: int = 30,
) -> pd.DataFrame:
    url = (
        f"{WORLD_BANK_BASE_URL}/country/all/indicator/{indicator_code}"
        "?format=json&per_page=20000"
    )
    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()

    payload: list[Any] = response.json()
    if not isinstance(payload, list) or len(payload) < 2:
        return _empty_dataframe()

    records = payload[1] or []
    raw_df = pd.DataFrame(records)
    if raw_df.empty:
        return _empty_dataframe()

    normalized = pd.DataFrame(
        {
            "country_code": raw_df.get("countryiso3code"),
            "year": pd.to_numeric(raw_df.get("date"), errors="coerce"),
            "indicator_code": indicator_code,
            "indicator_name": indicator_name,
            "source": "world_bank",
            "value": pd.to_numeric(raw_df.get("value"), errors="coerce"),
        }
    )

    normalized["country_code"] = normalized["country_code"].astype(str).str.upper()
    normalized = normalized[
        normalized["country_code"].notna()
        & (normalized["country_code"] != "NAN")
        & (normalized["country_code"].str.len() == 3)
    ]
    normalized = normalized.dropna(subset=["year"])
    normalized["year"] = normalized["year"].astype(int)
    return normalized.reset_index(drop=True)


def _empty_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "country_code",
            "year",
            "indicator_code",
            "indicator_name",
            "source",
            "value",
        ]
    )
