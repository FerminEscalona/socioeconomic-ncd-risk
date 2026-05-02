"""Extractor para WHO GHO API."""

from __future__ import annotations

from typing import Any

import pandas as pd
import requests

WHO_BASE_URL = "https://ghoapi.azureedge.net/api"


def extract_who_indicator(
    indicator_code: str,
    indicator_name: str,
    timeout_seconds: int = 30,
) -> pd.DataFrame:
    url = f"{WHO_BASE_URL}/{indicator_code}"
    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()

    payload: dict[str, Any] = response.json()
    records = payload.get("value", [])
    if not records:
        return _empty_dataframe()

    raw_df = pd.DataFrame(records)
    if raw_df.empty:
        return _empty_dataframe()

    normalized = pd.DataFrame(
        {
            "country_code": raw_df.get("SpatialDim"),
            "year": pd.to_numeric(raw_df.get("TimeDim"), errors="coerce"),
            "indicator_code": indicator_code,
            "indicator_name": indicator_name,
            "source": "who",
            "value": pd.to_numeric(raw_df.get("NumericValue"), errors="coerce"),
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
