# Retail Store Weekly Sales Forecasting

A complete end-to-end data science and machine learning pipeline predicting weekly retail store sales using macroeconomic factors, store-level metadata, temporal patterns, and autoregressive lag features.

Accurate retail sales forecasting is critical for inventory management, staffing, and financial planning. This project builds and rigorously validates a multi-variable regression model using historical weekly sales data.

### Key Highlights:

- Exploratory Data Analysis & Cleaning: Handled multi-store structures and diagnosed extreme multicollinearity using Variance Inflation Factor (VIF).
- Rigorous Out-of-Sample Validation: Avoided data leakage by utilizing a strict chronological 80/20 time-based split (training on past data, testing strictly on unseen future weeks).
- Advanced Feature Engineering: Incorporated a log-transformed target variable to stabilize variance and introduced a lagged feature (sales_last_week_log) to capture historical sales momentum.
- Top-Tier Performance: Achieved an out-of-sample R^2 of 0.976 and significantly reduced error metrics compared to baseline models.

### Project Workflow & Methodology
1. Initial OLS Regression & Multicollinearity Correction
- Initial multi-variable Ordinary Least Squares (OLS) regression yielded a strong in-sample $R^2$ of 0.937, but flagged a massive condition number and high VIF values.
- Diagnosis: The cpi (Consumer Price Index) feature exhibited a VIF exceeding 446 and was statistically insignificant ($p = 0.340$).
- Resolution: Dropped cpi, which dramatically improved numerical stability (lowering the condition number by an order of magnitude) without sacrificing any predictive power.

2. Time-Based Out-of-Sample Validation
- Rather than using a random train/test split (which would allow the model to "peek" into future weeks), the dataset was split chronologically:
   - Train: Earliest 80% of weeks.
   - Test: Latest 20% of unseen future weeks.
   - Benchmark: Compared against a DummyRegressor (mean baseline).

 
| Model | RMSE | MAE | Out-of-Sample R² |
| :--- | :---: | :---: | :---: |
| Mean Baseline | $533,710.71 | $454,692.95 | -0.0005 |
| Initial Linear Regression | $102,086.34 | $69,544.75 | 0.9634 |
| Optimized Log-Linear + Lag Model | $82,850.88 | $54,972.24 | 0.9759 |

### Model Diagnostics & Visualizations
1. Actual vs. Predicted Performance
  The model tracks reality remarkably well across both low-volume and high-volume sales weeks, hugging the 45-degree ideal line closely.
2. Residual Analysis (Curing Heteroscedasticity)
   By applying a natural logarithm to weekly sales ($\ln(y)$), the model successfully neutralized the "megaphone" heteroscedasticity pattern seen in baseline dollar-scale models, bounding errors symmetrically around zero.
3. Feature Impact & AutoregressionIntroducing sales_last_week_log established sales momentum as the dominant predictor, while controlling for pre-holiday spending shifts.

### Tech Stack & Libraries

- Python `Python 3.14.4`

- Pandas & NumPy (Data manipulation and transformation)

- Statsmodels (OLS regression summaries, VIF diagnostics)

- Scikit-Learn (`LinearRegression`, `DummyRegressor`, evaluation metrics)

-  Matplotlib & Seaborn (Data visualization and residual plotting)

### How to Run the Code

1. Clone the repository:
```bash
git clone https://github.com/emmanuelakinmokun/predicting-weekly-retail-sales.git
```

2. Install dependencies
```bash
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn
```
