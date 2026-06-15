# Fraud Detection for E-commerce and Bank Transactions

## Project Overview
This project builds fraud detection models for two transaction streams:
- **E-commerce transactions** (Fraud_Data.csv) — rich behavioral and device context
- **Bank credit card transactions** (creditcard.csv) — PCA-anonymized features

Built as part of the Kifiya AI Training Program (10 Academy) — Week 5 Challenge.

## Project Structure
fraud-detection/

├── data/

│   ├── raw/                    # Original datasets (gitignored)

│   └── processed/              # Cleaned data and visualizations

├── notebooks/

│   ├── eda-fraud-data.ipynb    # Task 1: EDA for e-commerce data

│   ├── eda-creditcard.ipynb    # Task 1: EDA for credit card data

│   ├── modeling.ipynb          # Task 2: Model building and evaluation

│   └── shap-explainability.ipynb # Task 3: SHAP analysis

├── models/                     # Saved model artifacts

├── requirements.txt

└── README.md
## Setup Instructions
```bash
# Clone the repository
git clone https://github.com/reb-ika/fraud-detection.git
cd fraud-detection

# Install dependencies
pip install -r requirements.txt

# Add datasets to data/raw/
# - Fraud_Data.csv
# - IpAddress_to_Country.csv
# - creditcard.csv

# Launch Jupyter
jupyter notebook
```

## Task 1 — Data Analysis and Preprocessing
### E-commerce Data (Fraud_Data.csv)
- 151,112 transactions, 9.36% fraud rate
- No missing values or duplicates found
- Merged with IpAddress_to_Country.csv using range-based IP lookup
- Engineered features: `time_since_signup`, `hour_of_day`, `day_of_week`, `user_txn_count`
- Applied SMOTE on training set to handle class imbalance

### Credit Card Data (creditcard.csv)
- 284,807 transactions, 0.17% fraud rate — severely imbalanced
- Features V1-V28 already PCA-transformed for privacy
- Scaled `Amount` and `Time` using StandardScaler
- Applied SMOTE on training set only

## Task 2 — Model Building and Evaluation

### E-commerce Fraud Data Results
| Model | F1 Score | AUC-PR |
|-------|----------|--------|
| Random Forest | 0.6825 | 0.6228 |
| XGBoost | 0.6775 | 0.6029 |
| Logistic Regression | 0.3035 | 0.3600 |

✅ **Best Model: Random Forest** — highest F1 and AUC-PR

### Credit Card Data Results
| Model | F1 Score | AUC-PR |
|-------|----------|--------|
| XGBoost | 0.4942 | 0.8270 |
| Random Forest | 0.5608 | 0.7964 |
| Logistic Regression | 0.1094 | 0.7249 |

✅ **Best Model: XGBoost** — highest AUC-PR (best at ranking fraud risk)

### Why AUC-PR over Accuracy?
Overall accuracy is misleading on imbalanced data. A model predicting
all transactions as legitimate would achieve 90% accuracy on fraud data
but catch zero fraud cases. AUC-PR focuses on the minority (fraud) class
and measures the tradeoff between precision and recall directly.

## Task 3 — Model Explainability (SHAP)

### Top Fraud Drivers — E-commerce (Random Forest)
1. **time_since_signup** — Transactions very soon after signup are strong fraud signals
2. **purchase_value** — Unusually high purchase amounts correlate with fraud
3. **hour_of_day** — Fraud clusters at certain hours (late night)
4. **user_txn_count** — Multiple rapid transactions signal automated fraud
5. **age** — Certain age groups show higher fraud rates

### Top Fraud Drivers — Credit Card (XGBoost)
- V14, V17, V12 — PCA components most predictive of fraud
- Amount_scaled — High transaction amounts elevate fraud risk

## Business Recommendations
1. **Flag transactions within 1 hour of signup** for additional verification — 
   time_since_signup is the strongest fraud predictor in e-commerce data
2. **Trigger step-up authentication for high-value purchases** — 
   purchase_value consistently drives fraud predictions
3. **Monitor late-night transaction spikes** — hour_of_day shows fraud 
   clusters at off-peak hours suggesting automated attacks
4. **Rate-limit rapid sequential transactions** — high user_txn_count 
   indicates bot-driven fraud attempts
5. **Prioritize AUC-PR over accuracy** in all fraud model evaluations —
   the cost of missing fraud far exceeds the cost of false positives

## Tools and Technologies
- Python, Pandas, NumPy
- Scikit-learn, XGBoost, LightGBM
- Imbalanced-learn (SMOTE)
- SHAP for model explainability
- Matplotlib, Seaborn
- Jupyter Notebooks
- Git/GitHub

## Author
Rebika Woldeyesus — Kifiya AI Training Program (10 Academy), Week 5