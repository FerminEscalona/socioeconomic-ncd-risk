"""Extractor para WHO GHO API."""

from __future__ import annotations

import json
from typing import Any
from urllib.request import urlopen

import pandas as pd

WHO_BASE_URL = "https://ghoapi.azureedge.net/api"
WHO_RAW_COLUMNS = [
    "Id",
    "IndicatorCode",
    "SpatialDimType",
    "SpatialDim",
    "ParentLocationCode",
    "ParentLocation",
    "TimeDimType",
    "TimeDim",
    "Dim1Type",
    "Dim1",
    "Dim2Type",
    "Dim2",
    "Dim3Type",
    "Dim3",
    "DataSourceDimType",
    "DataSourceDim",
    "Value",
    "NumericValue",
    "Low",
    "High",
    "Comments",
    "Date",
    "TimeDimensionValue",
    "TimeDimensionBegin",
    "TimeDimensionEnd",
]


def extract_who_indicator(
    indicator_code: str,
    indicator_name: str,
    timeout_seconds: int = 30,
) -> pd.DataFrame:
    url = f"{WHO_BASE_URL}/{indicator_code}"
    with urlopen(url, timeout=timeout_seconds) as response:
        payload: dict[str, Any] = json.load(response)

    records = payload.get("value", [])
    if not records:
        return _empty_dataframe()

    raw_df = pd.DataFrame(records)
    if raw_df.empty:
        return _empty_dataframe()

    # Conservamos las dimensiones originales de WHO para explicar duplicados.
    for column in WHO_RAW_COLUMNS:
        if column not in raw_df.columns:
            raw_df[column] = pd.NA

    normalized = raw_df[WHO_RAW_COLUMNS].copy()
    normalized.insert(0, "country_code", raw_df.get("SpatialDim"))
    normalized.insert(1, "year", pd.to_numeric(raw_df.get("TimeDim"), errors="coerce"))
    normalized.insert(2, "indicator_code", indicator_code)
    normalized.insert(3, "indicator_name", indicator_name)
    normalized.insert(4, "source", "who")
    normalized.insert(5, "value", pd.to_numeric(raw_df.get("NumericValue"), errors="coerce"))

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
            *WHO_RAW_COLUMNS,
        ]
    )
