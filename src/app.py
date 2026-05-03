"""Dashboard Streamlit: Comparación de modelos para mortalidad por ENT (NCDMORT3070)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

TARGET = "NCDMORT3070"

FEATURES = [
    "MDG_0000000001",
    "SH.H2O.BASW.ZS",
    "SH.MED.PHYS.ZS",
    "SH.XPD.CHEX.GD.ZS",
    "SP.DYN.LE00.IN",
    "WHS4_100",
    "NY.GDP.PCAP.CD_log1p",
    "TB_e_inc_num_log1p",
]

FEATURE_LABELS = {
    "MDG_0000000001": "Mortalidad Infantil",
    "SH.H2O.BASW.ZS": "Agua Potable (%)",
    "SH.MED.PHYS.ZS": "Médicos / 10k hab.",
    "SH.XPD.CHEX.GD.ZS": "Gasto en Salud (% PIB)",
    "SP.DYN.LE00.IN": "Esperanza de Vida",
    "WHS4_100": "Camas Hospitalarias",
    "NY.GDP.PCAP.CD_log1p": "PIB per cápita (log)",
    "TB_e_inc_num_log1p": "Incidencia TB (log)",
}

MODEL_COLORS = {
    "Ridge Regression": "#636EFA",
    "Random Forest": "#00CC96",
    "XGBoost": "#EF553B",
}


# ─── Carga de datos ────────────────────────────────────────────────────────────

def _build_uri() -> str:
    user = os.getenv("POSTGRES_USER", "postgres")
    pw = os.getenv("POSTGRES_PASSWORD", "")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_HOST_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "postgres")
    return f"postgresql://{user}:{pw}@{host}:{port}/{db}"


@st.cache_data(show_spinner=False, ttl=300)
def load_data() -> tuple[pd.DataFrame, str]:
    """Carga desde PostgreSQL; si falla, usa el CSV procesado."""
    query = f'SELECT * FROM analytics.v_model_panel WHERE "{TARGET}" IS NOT NULL'
    try:
        import connectorx as cx
        df = cx.read_sql(_build_uri(), query, return_type="pandas")
        return df, "PostgreSQL"
    except Exception:
        pass

    csv_path = ROOT_DIR / "data" / "processed" / "eda" / "unified_wide.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        return df[df[TARGET].notna()].copy(), "CSV (offline)"

    raise FileNotFoundError(
        "No se encontró PostgreSQL ni el CSV procesado. "
        "Ejecuta primero src/eda.py y src/database/load_postgres.py"
    )


# ─── Modelado ─────────────────────────────────────────────────────────────────

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R²": r2_score(y_true, y_pred),
    }


@st.cache_data(show_spinner=False)
def run_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    selected: list[str],
) -> dict:
    catalog = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_features="sqrt", random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
        ),
    }

    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(X_train)
    Xte_sc = scaler.transform(X_test)

    out = {}
    for name in selected:
        m = catalog[name]
        if name == "Ridge Regression":
            m.fit(Xtr_sc, y_train)
            tr_pred = m.predict(Xtr_sc)
            te_pred = m.predict(Xte_sc)
            imp = pd.Series(np.abs(m.coef_), index=X_train.columns)
        else:
            m.fit(X_train, y_train)
            tr_pred = m.predict(X_train)
            te_pred = m.predict(X_test)
            imp = pd.Series(m.feature_importances_, index=X_train.columns)

        out[name] = {
            "train": _metrics(y_train, tr_pred),
            "test": _metrics(y_test, te_pred),
            "test_pred": te_pred,
            "importance": imp.rename(index=FEATURE_LABELS).sort_values(ascending=False),
        }
    return out


# ─── Helpers de visualización ──────────────────────────────────────────────────

def _metric_bar(results: dict, metric: str, split_year: int) -> go.Figure:
    names = list(results.keys())
    values = [results[n]["test"][metric] for n in names]
    colors = [MODEL_COLORS[n] for n in names]
    fig = go.Figure(go.Bar(
        x=names, y=values, marker_color=colors, text=[f"{v:.3f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"{metric} en prueba (año ≥ {split_year})",
        yaxis_title=metric,
        template="plotly_dark",
        showlegend=False,
    )
    return fig


def _scatter_pred(pred_df: pd.DataFrame, model_name: str) -> go.Figure:
    fig = px.scatter(
        pred_df,
        x="Real",
        y="Predicción",
        color="Año",
        hover_data=["País"],
        title=f"{model_name}: Predicciones vs Valores Reales (prueba)",
        template="plotly_dark",
        trendline="ols",
    )
    lo = min(pred_df["Real"].min(), pred_df["Predicción"].min()) * 0.95
    hi = max(pred_df["Real"].max(), pred_df["Predicción"].max()) * 1.05
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines", name="Perfecta",
        line=dict(dash="dash", color="white", width=1),
    ))
    return fig


def _residual_plot(pred_df: pd.DataFrame, model_name: str) -> go.Figure:
    fig = px.scatter(
        pred_df,
        x="Predicción",
        y="Error",
        color="Año",
        hover_data=["País"],
        title=f"{model_name}: Residuales",
        template="plotly_dark",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.5)")
    return fig


def _importance_bar(imp: pd.Series, model_name: str) -> go.Figure:
    color = MODEL_COLORS.get(model_name, "#636EFA")
    fig = go.Figure(go.Bar(
        x=imp.values[::-1],
        y=imp.index[::-1],
        orientation="h",
        marker_color=color,
    ))
    fig.update_layout(
        title=f"{model_name} — Importancia de Variables",
        xaxis_title="Importancia",
        template="plotly_dark",
    )
    return fig


# ─── App principal ─────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="NCD Risk — Modelos Predictivos",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🌍 Mortalidad Prematura por Enfermedades No Transmisibles")
    st.caption(
        "Comparación de modelos predictivos usando indicadores socioeconómicos "
        "y de salud (WHO + World Bank, 2000–2024)."
    )

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuración")

        split_year = st.slider(
            "Año de corte train / prueba",
            min_value=2005,
            max_value=2022,
            value=2021,
            help="Filas con año < corte → entrenamiento | año ≥ corte → prueba",
        )
        st.caption(f"Train: 2000 – {split_year - 1}   |   Test: {split_year} – 2024")

        st.divider()
        st.subheader("Modelos")
        use_ridge = st.checkbox("Ridge Regression", value=True)
        use_rf = st.checkbox("Random Forest", value=True)
        use_xgb = st.checkbox("XGBoost", value=True)

        selected_models = [
            n for n, flag in [
                ("Ridge Regression", use_ridge),
                ("Random Forest", use_rf),
                ("XGBoost", use_xgb),
            ] if flag
        ]

        if not selected_models:
            st.warning("Selecciona al menos un modelo.")
            st.stop()

    # ── Carga ──────────────────────────────────────────────────────────────────
    with st.spinner("Conectando a la base de datos..."):
        try:
            df_raw, source = load_data()
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

    st.caption(f"Fuente activa: **{source}**")

    df = df_raw[["country_code", "year"] + FEATURES + [TARGET]].dropna(
        subset=FEATURES + [TARGET]
    )

    train_df = df[df["year"] < split_year]
    test_df = df[df["year"] >= split_year]

    if len(test_df) == 0:
        st.error(f"No hay datos de prueba para año ≥ {split_year}. Mueve el corte.")
        st.stop()

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    # ── KPIs superiores ────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Observaciones totales", f"{len(df):,}")
    c2.metric("Países", df["country_code"].nunique())
    c3.metric("Años cubiertos", f"{int(df['year'].min())}–{int(df['year'].max())}")
    c4.metric("Filas de entrenamiento", f"{len(train_df):,}")
    c5.metric("Filas de prueba", f"{len(test_df):,}")

    # ── Entrenar ───────────────────────────────────────────────────────────────
    with st.spinner("Entrenando modelos..."):
        results = run_models(X_train, y_train, X_test, y_test, tuple(selected_models))

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab_metrics, tab_pred, tab_imp = st.tabs([
        "📊 Métricas", "🔍 Predicciones", "📌 Importancia de Variables"
    ])

    # ── Tab 1: Métricas ────────────────────────────────────────────────────────
    with tab_metrics:
        st.subheader("Comparación en conjunto de prueba")

        rows = []
        for name, res in results.items():
            rows.append({
                "Modelo": name,
                "RMSE train": f"{res['train']['RMSE']:.3f}",
                "R² train": f"{res['train']['R²']:.3f}",
                "RMSE test": f"{res['test']['RMSE']:.3f}",
                "MAE test": f"{res['test']['MAE']:.3f}",
                "R² test": f"{res['test']['R²']:.3f}",
            })
        st.dataframe(pd.DataFrame(rows).set_index("Modelo"), use_container_width=True)

        col_r2, col_rmse = st.columns(2)
        with col_r2:
            st.plotly_chart(_metric_bar(results, "R²", split_year), use_container_width=True)
        with col_rmse:
            st.plotly_chart(_metric_bar(results, "RMSE", split_year), use_container_width=True)

        # Radar de métricas normalizadas
        st.subheader("Radar comparativo (métricas normalizadas)")
        metrics_list = ["RMSE", "MAE", "R²"]
        raw = {n: [results[n]["test"][m] for m in metrics_list] for n in results}
        raw_df = pd.DataFrame(raw, index=metrics_list)

        # Normalizar 0–1, invertir RMSE y MAE (menor es mejor)
        norm = raw_df.copy()
        for m in ["RMSE", "MAE"]:
            col = norm.loc[m]
            norm.loc[m] = 1 - (col - col.min()) / (col.max() - col.min() + 1e-9)
        for m in ["R²"]:
            col = norm.loc[m]
            norm.loc[m] = (col - col.min()) / (col.max() - col.min() + 1e-9)

        fig_radar = go.Figure()
        categories = ["RMSE↓", "MAE↓", "R²↑"]
        for name in results:
            vals = norm[name].tolist()
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=name,
                line_color=MODEL_COLORS[name],
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(range=[0, 1])),
            template="plotly_dark",
            title="Comparativa normalizada (más alto = mejor en esa métrica)",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Tab 2: Predicciones ────────────────────────────────────────────────────
    with tab_pred:
        model_choice = st.selectbox("Selecciona modelo", list(results.keys()))

        pred_df = test_df[["country_code", "year", TARGET]].copy()
        pred_df["Predicción"] = results[model_choice]["test_pred"]
        pred_df["Error"] = pred_df["Predicción"] - pred_df[TARGET]
        pred_df = pred_df.rename(columns={"country_code": "País", "year": "Año", TARGET: "Real"})

        col_sc, col_res = st.columns(2)
        with col_sc:
            st.plotly_chart(_scatter_pred(pred_df, model_choice), use_container_width=True)
        with col_res:
            st.plotly_chart(_residual_plot(pred_df, model_choice), use_container_width=True)

        # Distribución del error
        fig_hist = px.histogram(
            pred_df,
            x="Error",
            nbins=40,
            title=f"{model_choice}: Distribución de Errores",
            template="plotly_dark",
            color_discrete_sequence=[MODEL_COLORS[model_choice]],
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="white")
        st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Top 10 países con mayor error absoluto")
        top_err = (
            pred_df.assign(ErrorAbs=pred_df["Error"].abs())
            .sort_values("ErrorAbs", ascending=False)
            .head(10)
            .drop(columns="ErrorAbs")
        )
        st.dataframe(top_err.reset_index(drop=True), use_container_width=True)

    # ── Tab 3: Importancia ─────────────────────────────────────────────────────
    with tab_imp:
        st.subheader("¿Qué variables explican la mortalidad por ENT?")

        for name, res in results.items():
            st.plotly_chart(_importance_bar(res["importance"], name), use_container_width=True)

        # Heatmap de importancias combinadas
        st.subheader("Heatmap comparativo de importancias")
        imp_matrix = pd.DataFrame(
            {n: res["importance"] for n, res in results.items()}
        )
        fig_heat = px.imshow(
            imp_matrix.T,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="Blues",
            title="Importancia por modelo y variable",
            template="plotly_dark",
        )
        st.plotly_chart(fig_heat, use_container_width=True)


if __name__ == "__main__":
    main()
