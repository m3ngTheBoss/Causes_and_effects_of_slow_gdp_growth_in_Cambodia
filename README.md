# Causes and Effects of Slow GDP Growth in Cambodia

An analysis of Cambodia's economic growth from **1990–2024** using machine learning regression models to identify key drivers and predictors of GDP growth.

**Institute of Technology of Cambodia** — Department of Applied Mathematics and Statistics  
**Course:** AMSI32_MIP - Mini Project | Academic Year 2025-2026  
**SDG Alignment:** SDG 8 — Decent Work and Economic Growth (Target 8.1, 8.2)

## Team — Group 1

| Name | ID |
|---|---|
| CHAN Serey | e20230409 |
| HACH Lyheng | e20231175 |
| HAY Pechmanea | e20230391 |
| HORT Kimmeng | e20231111 |

**Instructors:** Dr. HAS Sothea, Dr. PHAUK Sokkhey, Mr. NHIM Malai

## Project Overview

This project explores how various macroeconomic indicators — exports, foreign direct investment (FDI), inflation, gross investment, unemployment, population growth, and others — relate to and predict Cambodia's GDP growth. It applies **Linear**, **Ridge**, **Lasso**, and **Random Forest** regression models, compares their performance, and surfaces the most influential factors.

**Key findings:**
- **Lasso Regression** (α=0.01) performs best on test data: R² = 0.50, RMSE = 2.64
- Top positive predictors: `real_premium`, `population growth`, `fdi_pct_gdp`
- Top negative predictors: `gross_investment`, `net_exports`, `unemployment rate`
- Model performance is modest (R² ~0.50), suggesting GDP growth depends on factors beyond the 8 features in this dataset.

## Files

| File | Description |
|---|---|
| `mini-project.ipynb` | Jupyter Notebook with the full analysis pipeline: data loading, imputation, EDA, modeling, and evaluation |
| `app.py` | Interactive [Streamlit](https://streamlit.io/) dashboard replicating the analysis with dynamic charts and model comparison |
| `economic_dataset.csv` | Cambodia economic indicators dataset (35 annual observations, 10 variables) sourced from World Bank / national indicators |
| `requirements.txt` | Python dependencies |
| `MINI-project-report.pdf` | Final report in PDF format |
| `MINI-project-report.docx` | Final report in Word format |

## Variables

| Column | Description |
|---|---|
| `year` | Calendar year (1990–2024) |
| `exports_pct` | Exports (% of GDP) |
| `fdi_pct_gdp` | Foreign Direct Investment (% of GDP) |
| `inflation_gd` | Inflation (GDP Deflator %) |
| `net_exports` | Net Exports |
| `gross_investment` | Gross Investment (% of GDP) |
| `real_premium` | Real Premium |
| `gdp_growth` | GDP Growth (%) — **target variable** |
| `unemployment rate` | Unemployment Rate (%) |
| `population growth` | Population Growth (%) |

## Setup

```bash
pip install -r requirements.txt
```

### Run the dashboard

```bash
streamlit run app.py
```

### Run the notebook

```bash
jupyter notebook mini-project.ipynb
```

## Requirements

- Python ≥ 3.8
- streamlit ≥ 1.28.0
- pandas ≥ 2.0.0
- numpy ≥ 1.24.0
- scipy ≥ 1.11.0
- scikit-learn ≥ 1.3.0
- plotly ≥ 5.18.0
- matplotlib ≥ 3.7.0
- seaborn ≥ 0.13.0

## Results Summary

| Model | Train R² | Test R² | Test RMSE | Test MAE |
|---|---|---|---|---|
| Linear Regression | 0.6256 | 0.4868 | 2.67 | 1.96 |
| Ridge (α=0.01) | 0.6249 | 0.4912 | 2.66 | 1.95 |
| **Lasso (α=0.01)** | **0.6161** | **0.4974** | **2.64** | **1.95** |
| Random Forest | 0.6652 | 0.3268 | 3.06 | 2.32 |
