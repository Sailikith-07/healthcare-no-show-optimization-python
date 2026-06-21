# Optimizing Healthcare Delivery: Predicting Patient No-Shows with Random Forest

## 📌 Executive Summary
Medical appointment no-shows disrupt clinical workflows, delay patient care, and present a massive financial burden to healthcare systems. This project leverages an optimized Random Forest classifier built on **110,000+ clinical appointment records** to proactively identify high-risk patients. 

By shifting the operational decision boundary to a strategic **50% threshold**, the final model successfully intercepts **78% of all no-shows**, unlocking an estimated **$277,000+ in recoverable revenue risk** while isolating critical operational bottlenecks.

---

## 🛠️ Repository Architecture & Workflow
The project is split into three modular stages to ensure clean code maintenance and reproducibility. Please execute them in the following order:

```
├── 1_data_cleaning.py       # Script 1: Raw data preprocessing and engineering
├── 2_eda_analysis.ipynb     # Notebook 1: Deep-dive exploratory visual analysis
└── 3_model_training.ipynb   # Notebook 2: Model tuning, thresholding & auditing
```

---

## 📈 Core Project Results & Metrics
Through hyperparameter tuning via `GridSearchCV` and custom business thresholding, the Random Forest model was optimized to balance clinical sensitivity with operational efficiency:

| Metric | Optimized Random Forest (50% Threshold) | High-Sensitivity Setting (40% Threshold) | Conservative Setting (60% Threshold) |
| :--- | :---: | :---: | :---: |
| **No-Show Catch Rate (Recall)** | **78.0%** | **91.0%** | **53.0%** |
| **Class 1 Precision** | 31.0% | 29.0% | 35.0% |
| **Overall Model Accuracy** | 60.0% | 53.0% | 71.0% |
| **Operational Priority** | **Balanced Sweet Spot** | Maximize Risk Capture | Minimize False Alarms |

---

## 💰 Operational & Financial Impact Analysis
Assuming a standard industry cost of **$200 per missed appointment slot**:
* **Total Untreated Revenue Risk:** The total cost of unmanaged missed appointments in the validation set equals the full volume of actual no-shows multiplied by $200.
* **Proactive Interception:** At the 50% threshold, the model successfully flags **78%** of these missing patients before they skip.
* **Recovered Revenue:** Assuming a conservative **50% operational success rate** (e.g., saving half of the flagged patients through automated follow-up calls or priority re-booking), deploying this model yields a projected revenue recovery of **$277,200.00**.

---

## 🔍 Key Data Insights & The "SMS Paradox"

### 1. Top Driving Factors (Feature Importance)
* **`WaitingDays` (The Primary Driver):** The chronological gap between the booking date and the physical appointment date is the single most powerful predictor of patient attrition.
* **`Age`:** Patient age was identified as the secondary driver, highlighting distinct behavioral reliability patterns across different generational cohorts.

### 2. Operational Audit: The SMS Paradox
During model auditing, a massive discrepancy in False Alarm Rates (False Positive Rates) was uncovered:
* **No SMS Population False Alarm Rate:** 31.48%
* **Received SMS Population False Alarm Rate:** **73.87%**

**Root-Cause Business Interpretation:** 
While it initially appears that text reminders cause higher model confusion, deeper pipeline tracking revealed a major **confounding variable**: the hospital's operational policy dictates that *only* patients with high lead times (`WaitingDays`) receive text alerts. The text reminders are functionally valuable, but because the AI over-indexes on the high waiting time, it creates an artificial behavioral penalty on text recipients. 

---

## 🛠️ Data Engineering & Modeling Workflow

1. **Chronological Cleaning:** Standardized mixed datetime strings, extracted explicit `WaitingDays`, and removed operational edge-cases (such as negative wait times).
2. **Feature Optimization:** Engineered binary structural values for multi-level categories and applied One-Hot Encoding to categorical variables (`AppointmentDayName`) to eradicate algorithm tracking bias.
3. **Hyperparameter Tuning:** Conducted a multi-core `GridSearchCV` optimizing `max_depth`, `n_estimators`, and `min_samples_split` targeting the Area Under the ROC Curve (`roc_auc`) to handle extreme class imbalance.
4. **Demographic Fairness Audit:** Evaluated demographic selection rates across protected classes (**Gender** and **Scholarship status**) to ensure equitable deployment boundaries across vulnerable patient populations.

---

## 🚀 How to Run the Project
1. Clone this repository.
2. Download the `KaggleV2-May-2016.csv` dataset into the project root directory.
3. Install dependencies: `pip install pandas numpy scikit-learn matplotlib seaborn`
4. Execute the Jupyter notebook or python script to reproduce the optimization grids and data visualizations.
