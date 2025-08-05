# Fraud Detection Using Machine Learning

A comprehensive project demonstrating the end-to-end process of building a **real-time fraud detection system** using Python, advanced analytics, and machine learning. This project is designed to showcase my ability to solve complex business problems with data
---

## Dataset Overview

- **Number of rows:** 6,362,620 transactions
- **Source:** [Kaggle - Fraud Detection Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)  
  *(The `Fraud.csv` file used in this project is sourced from the above Kaggle dataset.)*

### Feature Descriptions

| Feature                     | Description                                                                                              |
|-----------------------------|----------------------------------------------------------------------------------------------------------|
| `step`                      | Maps a unit of time in the real world (1 step = 1 hour; 744 steps = 30 days simulation)                 |
| `type`                      | Transaction type: CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER                                           |
| `amount`                    | Amount of the transaction in local currency                                                             |
| `nameOrig`                  | Customer who started the transaction                                                                    |
| `oldbalanceOrg`             | Initial balance before the transaction                                                                  |
| `newbalanceOrig`            | New balance after the transaction                                                                       |
| `nameDest`                  | Customer who is the recipient of the transaction                                                        |
| `oldbalanceDest`            | Initial balance of recipient before the transaction (missing for merchants)                             |
| `newbalanceDest`            | New balance of recipient after the transaction (missing for merchants)                                  |
| `isFraud`                   | 1 if the transaction is fraudulent, 0 otherwise                                                         |
| `isFlaggedFraud`            | 1 if the transaction was flagged as illegal (attempt to transfer >200,000 in a single transaction), 0 otherwise |

These features allow for detailed analysis of transaction patterns, account behaviors, and identification of fraudulent activities.



## Project Overview

This project addresses the challenge of detecting fraudulent financial transactions using a real-world dataset. The solution covers the full journey from **data → insight → action**:

- **Exploratory Data Analysis (EDA):**  
  - Assessed data quality, missing values, and duplicates.
  - Visualized distributions and outliers for key features.
  - Analyzed fraud distribution by transaction type and value.
  - Investigated feature correlations and multicollinearity using correlation matrices and VIF.
- **Data Cleaning & Feature Engineering:**  
  - Handled missing values and outliers (preserved as fraud signals).
  - Engineered business-interpretable features: balance changes, transaction patterns, merchant/customer flags, and ratios.
  - Removed multicollinearity by creating derived features.
- **Statistical Feature Selection:**  
  - Used ANOVA, Chi-square, and Mutual Information to identify the most predictive variables.
- **Model Training & Evaluation:**  
  - Compared Random Forest, Gradient Boosting, and XGBoost.
  - Focused on precision-recall tradeoff for business impact.
  - Used stratified cross-validation and business-relevant metrics.
- **Deployment:**  
  - Built a self-serve, interactive Streamlit app for real-time fraud scoring.

---

## Key Business Impact

- **Identified actionable fraud signals** (e.g., account draining, high amount-to-balance ratios, suspicious transaction patterns).
- **Deployed a real-time scoring tool** for business and product teams to flag and investigate risky transactions.
- **Demonstrated ability to move beyond reporting** by proposing prevention strategies and measurable KPIs for ongoing monitoring.

---

## Exploratory Data Analysis (EDA) Highlights

- **Data Quality:** No explicit missing values; structural missingness handled as business logic.
- **Outlier Analysis:** Outliers in amount and balance fields preserved as they are strong fraud signals.
- **Fraud Distribution:** Analyzed by transaction type and value; high-value transactions have higher fraud rates.
- **Correlation & Multicollinearity:** Detected and addressed using feature engineering (balance changes, ratios).
- **Visualization:** Used boxplots, histograms, and heatmaps to understand data patterns and relationships.

---

## Q&A: Business and Technical Insights

**Q1: Data Cleaning Including Missing Values, Outliers, and Multicollinearity**  
- No explicit missing values; handled structural missingness for merchants.
- Outliers (e.g., high-value transactions, account draining) preserved as fraud signals.
- Severe multicollinearity detected and addressed by creating derived features (balance changes, ratios).

**Q2: Fraud Detection Model Description**  
- Multi-model ensemble: Random Forest, Gradient Boosting, XGBoost.
- Used 8 statistically significant features (union of ANOVA, Chi-square, Mutual Information).
- Stratified cross-validation, business-focused metrics (ROC-AUC, precision, recall, F1-score).
- XGBoost selected for best precision-recall balance and real-time suitability.

**Q3: Variable Selection**  
- Three-method statistical approach: ANOVA (numerical), Chi-square (categorical), Mutual Information (non-linear).
- Final model uses 8 core features with strong business meaning.

**Q4: Model Performance**  
- Random Forest: ROC AUC 0.9982, Precision 0.0214, Recall 0.7089, F1 0.0416
- Gradient Boosting: ROC AUC 0.9984, Precision 0.1435, Recall 0.9961, F1 0.2509
- XGBoost: ROC AUC 0.9427, Precision 0.3521, Recall 0.7100, F1 0.4708 (chosen for best business tradeoff)

**Why XGBoost Was Chosen as the Final Model**  
- **Superior Precision-Recall Balance:** XGBoost achieved the highest precision (0.3521) while maintaining strong recall (0.7100), resulting in the best F1-score (0.4708). This balance is crucial for fraud detection, as it minimizes false positives while still catching most fraudulent transactions.
- **Faster Training and Prediction:** XGBoost demonstrated significantly faster training and prediction times compared to Gradient Boosting, making it more suitable for real-time fraud detection systems where quick response is essential.
- **Business Impact:** The higher precision of XGBoost means fewer legitimate transactions are incorrectly flagged as fraud, reducing customer friction and operational costs.
- **Scalability:** XGBoost's computational efficiency makes it ideal for processing large-scale transaction data in production environments.
- **Interpretability:** XGBoost provides clear feature importance rankings, supporting business transparency and actionable insights.

**Q5: Key Fraud Prediction Factors**  
- `balance_change_orig`: Account draining
- `amount_to_oldbalance_orig_ratio`: High ratio signals fraud
- `type`: Transaction type (TRANSFER, CASH_OUT)
- `transaction_pattern_encoded`: Customer-to-customer patterns
- `dest_is_merchant`: Merchant vs customer destination

**Q6: Do These Factors Make Sense?**  
- Yes; they align with known fraud mechanisms (account takeover, draining, suspicious patterns).

**Q7: Prevention Recommendations**  
- Real-time fraud scoring system with automated blocking and manual review.
- Critical transaction monitoring and enhanced security controls (MFA, velocity checks).

**Q8: Measuring Success**  
- KPIs: 90% fraud detection, <2% false positives, 70-90% fraud loss reduction.
- Continuous monitoring via dashboards and regular model/business reviews.

---

## How to Use

### 1. Setup

- Clone the repository.
- Install dependencies:
  ```
  pip install -r requirements.txt
  ```
- Place the dataset (`Fraud.csv`) in the project folder.

### 2. Model Training

- Run `Fraud Detection.ipynb` to:
  - Clean and analyze the data.
  - Engineer features and select the most important ones.
  - Train and evaluate multiple models.
  - Save the final XGBoost model and encoders for deployment.

### 3. Real-Time Fraud Detection App

- Launch the Streamlit app:
  ```
  streamlit run main.py
  ```
- Enter transaction details in the web interface to get instant fraud predictions.

---

## Project Structure

```
Fraud-Detection-Using-ML/
│
├── Fraud Detection.ipynb        # Full EDA, feature engineering, model training, Q&A
├── main.py                     # Streamlit app for real-time fraud scoring
├── model_features.pkl          # Selected features for the model
├── type_label_encoder.pkl      # Label encoder for transaction type
├── xgb_fraud_model.pkl         # Trained XGBoost model
├── Data Dictionary.txt         # Dataset schema and business context
├── README.md                   # Project documentation (this file)
├── .gitignore                  # Ignore data, models, and temp files
└── Fraud.csv                   # (Not included in repo; see .gitignore)
```

---

## Sample Use Case

**Business Problem:**  
Fraudulent agents attempt to drain customer accounts by transferring funds and cashing out.

**Solution:**  
- The app predicts whether a transaction is likely to be fraudulent based on key features such as transaction type, amount, balance changes, and customer/merchant patterns.
- Business teams can use the app to flag high-risk transactions for review or automated blocking.

---


## Technical Highlights

- **Python (Pandas, NumPy, scikit-learn, XGBoost, Streamlit)**
- **Advanced feature engineering and statistical analysis**
- **Model evaluation using ROC-AUC, precision, recall, and F1-score**
- **Deployment-ready code for real-time analytics**


