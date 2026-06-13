"""
Streamlit dashboard: Causes and Effects of Slow GDP Growth in Cambodia
Mirrors the analysis in mini-project.ipynb
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import skew
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).parent
DATA_FILES = ["economic_dataset_completed.csv", "economic_dataset.csv"]

FEATURE_LABELS = {
    "exports_pct": "Exports (% of GDP)",
    "fdi_pct_gdp": "FDI (% of GDP)",
    "inflation_gd": "Inflation (GDP Deflator %)",
    "net_exports": "Net Exports",
    "gross_investment": "Gross Investment (% of GDP)",
    "real_premium": "Real Premium",
    "unemployment rate": "Unemployment Rate (%)",
    "population growth": "Population Growth (%)",
}


@st.cache_data
def load_and_preprocess() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load CSV, clean column names, and impute missing exports with median."""
    df_raw = None
    for name in DATA_FILES:
        path = DATA_DIR / name
        if path.exists():
            df_raw = pd.read_csv(path)
            break
    if df_raw is None:
        raise FileNotFoundError("No dataset found. Expected economic_dataset.csv in project folder.")

    df_raw.columns = df_raw.columns.str.strip()
    missing_before = df_raw.isnull().sum().to_dict()

    export_skew = skew(df_raw["exports_pct"].dropna())
    export_mean = df_raw["exports_pct"].mean()
    export_median = df_raw["exports_pct"].median()

    df = df_raw.copy()
    df["exports_pct"] = df["exports_pct"].fillna(export_median)

    imputation_info = {
        "missing_before": missing_before,
        "export_skew": export_skew,
        "export_mean": export_mean,
        "export_median": export_median,
        "imputation_method": "median (|skew| > 0.5 → left-skewed distribution)",
    }
    return df_raw, df, imputation_info


@st.cache_data
def train_models(df: pd.DataFrame) -> dict:
    """Train all regression models used in the notebook."""
    features = [col for col in df.columns if col not in ["year", "gdp_growth"]]
    X = df[features]
    y = df["gdp_growth"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    linear = LinearRegression()
    linear.fit(X_train_scaled, y_train)

    ridge = Ridge(alpha=0.01)
    ridge.fit(X_train_scaled, y_train)

    lasso = Lasso(alpha=0.01)
    lasso.fit(X_train_scaled, y_train)

    rf = RandomForestRegressor(n_estimators=100, max_depth=2, random_state=42)
    rf.fit(X_train, y_train)

    def evaluate(model, X_tr, X_te, scaled: bool):
        if scaled:
            tr_pred = model.predict(X_tr)
            te_pred = model.predict(X_te)
        else:
            tr_pred = model.predict(X_tr)
            te_pred = model.predict(X_te)
        return {
            "train_r2": r2_score(y_train, tr_pred),
            "test_r2": r2_score(y_test, te_pred),
            "mse": mean_squared_error(y_test, te_pred),
            "mae": mean_absolute_error(y_test, te_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, te_pred)),
            "y_test_pred": te_pred,
        }

    linear_metrics = evaluate(linear, X_train_scaled, X_test_scaled, True)
    ridge_metrics = evaluate(ridge, X_train_scaled, X_test_scaled, True)
    lasso_metrics = evaluate(lasso, X_train_scaled, X_test_scaled, True)
    rf_metrics = evaluate(rf, X_train, X_test, False)

    comparison = pd.DataFrame(
        [
            {
                "Model": "Linear Regression",
                "Train R²": linear_metrics["train_r2"],
                "Test R²": linear_metrics["test_r2"],
                "MSE": linear_metrics["mse"],
                "MAE": linear_metrics["mae"],
                "RMSE": linear_metrics["rmse"],
            },
            {
                "Model": "Ridge (α=0.01)",
                "Train R²": ridge_metrics["train_r2"],
                "Test R²": ridge_metrics["test_r2"],
                "MSE": ridge_metrics["mse"],
                "MAE": ridge_metrics["mae"],
                "RMSE": ridge_metrics["rmse"],
            },
            {
                "Model": "Lasso (α=0.01)",
                "Train R²": lasso_metrics["train_r2"],
                "Test R²": lasso_metrics["test_r2"],
                "MSE": lasso_metrics["mse"],
                "MAE": lasso_metrics["mae"],
                "RMSE": lasso_metrics["rmse"],
            },
            {
                "Model": "Random Forest",
                "Train R²": rf_metrics["train_r2"],
                "Test R²": rf_metrics["test_r2"],
                "MSE": rf_metrics["mse"],
                "MAE": rf_metrics["mae"],
                "RMSE": rf_metrics["rmse"],
            },
        ]
    )

    linear_coefs = pd.Series(linear.coef_, index=features).sort_values(ascending=False)
    lasso_coefs = pd.DataFrame({"Feature": features, "Coefficient": lasso.coef_})
    rf_importance = pd.Series(rf.feature_importances_, index=features).sort_values(
        ascending=False
    )

    return {
        "features": features,
        "X_test": X_test,
        "y_test": y_test,
        "comparison": comparison,
        "linear_coefs": linear_coefs,
        "lasso_coefs": lasso_coefs,
        "rf_importance": rf_importance,
        "linear_metrics": linear_metrics,
        "predictions": {
            "Linear Regression": linear_metrics["y_test_pred"],
            "Ridge (α=0.01)": ridge_metrics["y_test_pred"],
            "Lasso (α=0.01)": lasso_metrics["y_test_pred"],
            "Random Forest": rf_metrics["y_test_pred"],
        },
    }


def styled_metrics(df: pd.DataFrame) -> None:
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest GDP Growth", f"{latest['gdp_growth']:.2f}%", f"{latest['gdp_growth'] - prev['gdp_growth']:.2f}% vs prior year")
    c2.metric("Avg GDP Growth", f"{df['gdp_growth'].mean():.2f}%")
    c3.metric("Years Covered", f"{int(df['year'].min())}–{int(df['year'].max())}")
    c4.metric("Observations", len(df))


def page_overview(df: pd.DataFrame, imputation_info: dict) -> None:
    st.header("Overview")
    st.markdown(
        "Interactive dashboard exploring **Cambodia's GDP growth (1990–2024)** and its "
        "relationship with exports, FDI, inflation, investment, and labor-market indicators."
    )
    styled_metrics(df)

    st.subheader("GDP Growth Over Time")
    fig = px.line(
        df,
        x="year",
        y="gdp_growth",
        markers=True,
        title="GDP Growth Trend (1990–2024)",
        labels={"year": "Year", "gdp_growth": "GDP Growth (%)"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.6)
    fig.update_layout(hovermode="x unified", height=420)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Data preprocessing notes"):
        st.write(
            f"**Missing exports (1990–1992):** imputed with median "
            f"({imputation_info['export_median']:.2f}%) because skewness = "
            f"{imputation_info['export_skew']:.3f} (|skew| > 0.5 → use median over mean)."
        )
        st.write(
            f"Mean exports: {imputation_info['export_mean']:.2f}% · "
            f"Median exports: {imputation_info['export_median']:.2f}%"
        )


def page_data_explorer(df_raw: pd.DataFrame, df: pd.DataFrame, imputation_info: dict) -> None:
    st.header("Data Explorer")

    tab_raw, tab_clean, tab_stats = st.tabs(["Raw Data", "Cleaned Data", "Summary Statistics"])

    with tab_raw:
        st.subheader("Original dataset")
        st.dataframe(df_raw, use_container_width=True, height=400)
        missing = pd.DataFrame(
            {"Column": list(imputation_info["missing_before"].keys()), "Missing Count": list(imputation_info["missing_before"].values())}
        )
        st.subheader("Missing values (before imputation)")
        st.dataframe(missing, use_container_width=True, hide_index=True)

    with tab_clean:
        st.subheader("Dataset after median imputation")
        st.dataframe(df, use_container_width=True, height=400)
        st.success("No missing values remain after preprocessing.")

    with tab_stats:
        st.subheader("Descriptive statistics")
        numeric = df.drop(columns=["year"])
        st.dataframe(numeric.describe().T.round(3), use_container_width=True)


def page_eda(df: pd.DataFrame) -> None:
    st.header("Exploratory Data Analysis")

    year_range = st.slider(
        "Year range",
        int(df["year"].min()),
        int(df["year"].max()),
        (int(df["year"].min()), int(df["year"].max())),
    )
    filtered = df[(df["year"] >= year_range[0]) & (df["year"] <= year_range[1])]

    st.subheader("Correlation Matrix")
    corr = filtered.drop(columns=["year"]).corr()
    corr_labels = [FEATURE_LABELS.get(c, c) for c in corr.columns]
    corr_display = corr.copy()
    corr_display.index = corr_labels
    corr_display.columns = corr_labels

    fig_corr = px.imshow(
        corr_display,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Correlation Matrix of Economic Indicators",
    )
    fig_corr.update_layout(height=520)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Time Series Trends")
    indicators = {
        "gdp_growth": ("GDP Growth (%)", "#2563eb"),
        "inflation_gd": ("Inflation – GDP Deflator (%)", "#dc2626"),
        "gross_investment": ("Gross Investment (% of GDP)", "#0d9488"),
        "exports_pct": ("Exports (% of GDP)", "#7c3aed"),
        "fdi_pct_gdp": ("FDI (% of GDP)", "#ea580c"),
    }
    selected = st.multiselect(
        "Select indicators to plot",
        list(indicators.keys()),
        default=["gdp_growth", "inflation_gd", "gross_investment"],
        format_func=lambda k: indicators[k][0],
    )

    if selected:
        fig_ts = go.Figure()
        for col in selected:
            label, color = indicators[col]
            fig_ts.add_trace(
                go.Scatter(
                    x=filtered["year"],
                    y=filtered[col],
                    mode="lines+markers",
                    name=label,
                    line=dict(color=color),
                )
            )
        fig_ts.update_layout(
            title="Selected Economic Indicators Over Time",
            xaxis_title="Year",
            yaxis_title="Value",
            hovermode="x unified",
            height=440,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("GDP Growth Distribution")
        fig_hist = px.histogram(
            filtered,
            x="gdp_growth",
            nbins=12,
            title="Distribution of GDP Growth",
            labels={"gdp_growth": "GDP Growth (%)", "count": "Frequency"},
            color_discrete_sequence=["#2563eb"],
        )
        fig_hist.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        st.subheader("GDP Growth vs Exports")
        fig_scatter = px.scatter(
            filtered,
            x="exports_pct",
            y="gdp_growth",
            hover_data=["year"],
            trendline="ols",
            title="Exports vs GDP Growth",
            labels={
                "exports_pct": "Exports (% of GDP)",
                "gdp_growth": "GDP Growth (%)",
            },
            color_discrete_sequence=["#7c3aed"],
        )
        fig_scatter.update_layout(height=360)
        st.plotly_chart(fig_scatter, use_container_width=True)


def page_models(model_results: dict) -> None:
    st.header("Predictive Models")
    st.markdown(
        "Regression models predict **GDP growth** from economic features "
        "(80/20 train–test split, `random_state=42`). Linear, Ridge, and Lasso use standardized features."
    )

    comparison = model_results["comparison"].copy()
    for col in ["Train R²", "Test R²", "MSE", "MAE", "RMSE"]:
        comparison[col] = comparison[col].map(lambda x: f"{x:.4f}")

    st.subheader("Model Comparison")
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    best_idx = model_results["comparison"]["Test R²"].idxmax()
    best_model = model_results["comparison"].loc[best_idx, "Model"]
    best_r2 = model_results["comparison"].loc[best_idx, "Test R²"]
    st.info(f"Best test R²: **{best_model}** ({best_r2:.4f})")

    st.subheader("Actual vs Predicted (Test Set)")
    model_choice = st.selectbox(
        "Model for prediction plot",
        list(model_results["predictions"].keys()),
    )
    y_test = model_results["y_test"]
    y_pred = model_results["predictions"][model_choice]

    fig_pred = go.Figure()
    fig_pred.add_trace(
        go.Scatter(
            x=y_test.values,
            y=y_pred,
            mode="markers",
            name="Predictions",
            marker=dict(size=10, color="#2563eb"),
            text=[f"Year index {i}" for i in range(len(y_test))],
        )
    )
    lims = [min(y_test.min(), min(y_pred)) - 1, max(y_test.max(), max(y_pred)) + 1]
    fig_pred.add_trace(
        go.Scatter(x=lims, y=lims, mode="lines", name="Perfect fit", line=dict(dash="dash", color="gray"))
    )
    fig_pred.update_layout(
        title=f"{model_choice}: Actual vs Predicted GDP Growth",
        xaxis_title="Actual GDP Growth (%)",
        yaxis_title="Predicted GDP Growth (%)",
        height=420,
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    st.subheader("Feature Importance & Coefficients")
    tab_linear, tab_lasso, tab_rf = st.tabs(
        ["Linear Regression", "Lasso", "Random Forest"]
    )

    with tab_linear:
        coefs = model_results["linear_coefs"]
        coef_df = pd.DataFrame(
            {
                "Feature": [FEATURE_LABELS.get(f, f) for f in coefs.index],
                "Coefficient": coefs.values,
            }
        )
        fig_coef = px.bar(
            coef_df,
            x="Coefficient",
            y="Feature",
            orientation="h",
            title="Linear Regression Coefficients (standardized features)",
            color="Coefficient",
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0,
        )
        fig_coef.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_coef, use_container_width=True)
        st.caption(f"Test R² = {model_results['linear_metrics']['test_r2']:.4f} · MSE = {model_results['linear_metrics']['mse']:.4f}")

    with tab_lasso:
        lasso_df = model_results["lasso_coefs"].copy()
        lasso_df["Feature"] = lasso_df["Feature"].map(lambda f: FEATURE_LABELS.get(f, f))
        st.dataframe(lasso_df.round(4), use_container_width=True, hide_index=True)
        active = lasso_df[lasso_df["Coefficient"] != 0]
        st.write(f"Lasso retained **{len(active)}** of **{len(lasso_df)}** features (α = 0.01).")

    with tab_rf:
        imp = model_results["rf_importance"]
        imp_df = pd.DataFrame(
            {
                "Feature": [FEATURE_LABELS.get(f, f) for f in imp.index],
                "Importance": imp.values,
            }
        )
        fig_rf = px.bar(
            imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Random Forest Feature Importance",
            color_discrete_sequence=["#0d9488"],
        )
        fig_rf.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_rf, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="Cambodia GDP Growth Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("🇰🇭 Cambodia GDP")
    st.sidebar.caption("Causes & effects of GDP growth · 1990–2024")

    try:
        df_raw, df, imputation_info = load_and_preprocess()
        model_results = train_models(df)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "Data Explorer", "EDA", "Predictive Models"],
        label_visibility="collapsed",
    )

    st.title("Causes & Effects of GDP Growth in Cambodia")
    st.markdown("---")

    if page == "Overview":
        page_overview(df, imputation_info)
    elif page == "Data Explorer":
        page_data_explorer(df_raw, df, imputation_info)
    elif page == "EDA":
        page_eda(df)
    elif page == "Predictive Models":
        page_models(model_results)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Source:** `mini-project.ipynb`")
    st.sidebar.markdown("**Data:** World Bank / national economic indicators")


if __name__ == "__main__":
    main()
