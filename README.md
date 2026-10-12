# Customer Churn Prediction & Retention Intelligence System

An end-to-end machine learning project that predicts customer churn probability and helps businesses identify customers who may need retention support.

The system combines data analysis, feature engineering, machine learning, model explainability, FastAPI, and Streamlit to deliver customer churn predictions through a web application.

## 🚀 Live Demo

### Streamlit Application
Access the deployed application and predict customer churn using customer information.

**[Open Customer Churn Prediction App](https://customer-churn-prediction-system-h2ze6yuyub2dcpkpbsmqfg.streamlit.app)**

### FastAPI Documentation
Explore and test the prediction API through the interactive Swagger interface.

**[Open FastAPI Swagger Documentation](https://customer-churn-prediction-system-39vu.onrender.com/docs)**

### API Health Check
Check whether the deployed prediction API is available.

**[Check API Health](https://customer-churn-prediction-system-39vu.onrender.com/health)**

---

## 📌 1. Business Problem

Customer churn occurs when customers stop using a company's products or services. High churn can reduce recurring revenue and increase customer acquisition costs.

Businesses need a way to identify customers who may leave so that retention teams can investigate their concerns and take appropriate action.

### Business Objective

Identify customers at risk of churn and support data-driven customer retention decisions.

### Machine Learning Objective

Predict the probability that a customer will churn based on their demographic information, tenure, contract details, billing information, and subscribed services.

### Target Variable

`Churn Value`

- `0` — Customer did not churn.
- `1` — Customer churned.

### Unit of Prediction

One customer per prediction request.

---

## 📊 2. Dataset

**Dataset:** IBM Telco Customer Churn

**Number of records:** 7,043 customers

The dataset includes information about customer demographics, tenure, monthly charges, total charges, contracts, payment methods, and subscribed services.

The raw and processed datasets are excluded from this GitHub repository.

Place the downloaded raw dataset at:

```text
data/raw/IBM_Telco_customer_churn_IBM_dataset.csv
```

The dataset source should be cited according to the original download location and its usage terms.

---

## 🛠️ 3. Technologies Used

| Category | Technologies |
|---|---|
| Programming | Python |
| Data manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine learning | Scikit-learn, XGBoost |
| Feature selection | Mutual Information |
| Model explainability | SHAP |
| Preprocessing | StandardScaler, OneHotEncoder, ColumnTransformer |
| ML workflow | Scikit-learn Pipeline |
| API development | FastAPI, Pydantic |
| Frontend | Streamlit |
| Model serialization | Joblib |
| Version control | Git, GitHub |
| API hosting | Render |
| Frontend hosting | Streamlit Community Cloud |

---

## 🔄 4. Project Workflow

```text
Business Problem Definition
          |
          v
Dataset Understanding
          |
          v
Exploratory Data Analysis
          |
          v
Data Preprocessing
          |
          v
Feature Engineering
          |
          v
Train / Validation / Test Split
          |
          v
Baseline Model
          |
          v
Model Training and Comparison
          |
          v
Cross-Validation
          |
          v
Hyperparameter Tuning
          |
          v
Class Imbalance Handling
          |
          v
Threshold Optimization
          |
          v
Final Model Selection
          |
          v
Model Explainability
          |
          v
Model Serialization
          |
          v
FastAPI Prediction API
          |
          v
Streamlit Application
          |
          v
Deployment
```

---

## 🔍 5. Exploratory Data Analysis

EDA was performed to identify patterns and relationships between customer characteristics and churn.

### Key Findings

- Customers with shorter tenure had higher churn rates.
- Month-to-month contract customers had the highest churn rate.
- Customers with higher monthly charges showed a higher churn tendency.
- Electronic Check users had the highest churn rate among payment methods.
- Fiber Optic customers had the highest churn rate among internet service types.
- Customers without Online Security showed a higher churn rate.
- Customers without Tech Support showed a higher churn rate.
- Customers with month-to-month contracts and low tenure represented an important high-risk segment.

These findings describe patterns in the historical dataset. They do not establish that a particular customer characteristic causes churn.

---

## ⚙️ 6. Feature Engineering

Two additional features were created and evaluated.

### Service Count

Represents the number of selected services associated with a customer.

### Tenure Group

Groups customers into four tenure ranges:

- 0–12 months
- 13–24 months
- 25–48 months
- 49–72 months

### Feature Selection

Mutual Information was used to rank candidate features according to their relationship with the churn target.

The highest-ranked features included:

- Contract
- Tenure Months
- Online Security
- Tech Support
- Tenure Group
- Internet Service

Experiments compared the top 10 features, top 15 features, and all 22 candidate features.

Feature selection did not produce a meaningful improvement over using all 22 features in the evaluated experiments, so all 22 were retained for the final model.

---

## 🤖 7. Models Evaluated

The following models were evaluated during development:

1. Dummy Classifier
2. Logistic Regression
3. Balanced Logistic Regression
4. Decision Tree
5. Random Forest
6. Gradient Boosting
7. Support Vector Machine (SVM)
8. XGBoost

The workflow included stratified data splitting, five-fold cross-validation, hyperparameter tuning, and decision-threshold experiments.

### Evaluation Metrics

The experiments considered:

- Accuracy
- Precision
- Recall
- F1 Score
- F2 Score
- ROC-AUC
- PR-AUC

Accuracy was not used as the sole selection criterion because the project's business objective prioritizes identifying customers who may churn.

---

## 🏆 8. Final Model Selection

**Selected model:** Balanced Logistic Regression

**Decision threshold:** `0.25`

The model was selected because identifying actual churners is the primary business priority.

Class weighting was used during training to give greater importance to the minority churn class.

### Why Recall Matters

A false negative occurs when the model predicts that a customer will not churn even though the customer actually churns.

Missing these customers can mean losing opportunities for timely retention intervention.

The selected threshold increases churn detection, but it also creates more false positives. The retention team must therefore consider its capacity and the costs of unnecessary outreach.

### Threshold Decision

A customer is classified as a churn candidate when:

```text
Predicted churn probability >= 0.25
```

The threshold was selected through experiments using validation data, with F2 used to give more weight to recall than precision.

The threshold is a decision rule, not a consequence of the target distribution alone.

---

## 📈 9. Final Test Results

The selected model was evaluated using a held-out test set at the decision threshold of `0.25`.

| Metric | Test Result |
|---|---:|
| Accuracy | 64.71% |
| Precision | 42.58% |
| Recall | 93.95% |
| F1 Score | 58.60% |
| ROC-AUC | 85.62% |
| PR-AUC | 68.17% |

### Confusion Matrix

```text
                 Predicted
                 No Churn   Churn

Actual No Churn     420      356
Actual Churn         17      264
```

### Interpretation

- **264 True Positives:** Actual churners correctly identified.
- **17 False Negatives:** Actual churners missed by the model.
- **420 True Negatives:** Customers who did not churn and were correctly classified.
- **356 False Positives:** Customers flagged as churn risks who did not churn.

The model detected approximately 93.95% of actual churners in this test set.

However, the 356 false positives show that high recall comes at a cost. The model should therefore be viewed as a churn-screening tool, not as a guarantee that every flagged customer will leave.

---

## 🔎 10. Model Explainability with SHAP

SHAP was used to investigate how input features influence model predictions.

The explainability analysis included:

- Global feature importance
- Feature contribution plots
- Analysis of individual predictions

The objective is to make model behavior more interpretable and support investigation of potential churn drivers.

SHAP contributions explain the model's output; they do not prove that a feature causes churn.

---

## 🏗️ 11. Application Architecture

The deployed application consists of a Streamlit frontend and a FastAPI backend.

```text
             User
              |
              v
       Streamlit Frontend
              |
              | Customer information
              v
         FastAPI Backend
              |
              v
       Input Validation
              |
              v
       Feature Engineering
              |
              v
    Saved ML Pipeline
              |
              v
      Churn Probability
              |
              v
       Decision Threshold
              |
              v
 Prediction + Risk Category
              |
              v
       Streamlit Results
```

### FastAPI Backend

The API handles customer input, validates the request, generates the engineered features, and returns the model's prediction.

Available endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check API availability |
| POST | `/predict` | Predict customer churn |

Interactive API documentation is available through the deployed Swagger interface.

### Streamlit Frontend

The frontend allows users to enter customer information through an interactive form.

It displays:

- Estimated churn probability
- Predicted outcome
- Risk level
- Decision threshold
- Suggested retention follow-up for flagged customers

The application-defined risk categories are:

| Estimated Churn Probability | Risk Category |
|---|---|
| Below 25% | Low |
| 25% to below 50% | Medium |
| 50% or above | High |

The classification threshold is 25%, so customers in the Medium category can still be classified as churn candidates.

**Important:** Churn probability is the model's estimate of the likelihood of churn. It is not the probability that the prediction is correct. Probability calibration has not yet been established.

---

## 📁 12. Project Structure

```text
customer_churn_ml/
│
├── api/
│   └── main.py
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── final_churn_model.joblib
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```

The raw and processed datasets are excluded from Git. The saved model artifact is included because the API requires it for prediction.

---

## 💻 13. Run the Project Locally

### Prerequisites

- Python
- Git
- A compatible Python environment

### Step 1: Clone the Repository

```bash
git clone https://github.com/reddyvaru/customer-churn-prediction-system.git
cd customer-churn-prediction-system
```

### Step 2: Create a Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 4: Start FastAPI

From the project root, run:

```bash
python -m uvicorn api.main:app --reload
```

The local API is available at:

```text
http://127.0.0.1:8000
```

Open the local Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

### Step 5: Start Streamlit

Open a second terminal, activate the same virtual environment, and run:

```bash
python -m streamlit run app/streamlit_app.py
```

The local frontend is normally available at:

```text
http://localhost:8501
```

Keep the FastAPI server running while using the Streamlit application.

---

## 📡 14. Example API Request

Send a `POST` request to:

```text
/predict
```

Example customer input:

```json
{
  "tenure_months": 6,
  "monthly_charges": 90.0,
  "total_charges": 540.0,
  "cltv": 4000,
  "partner": "No",
  "dependents": "No",
  "senior_citizen": "No",
  "gender": "Female",
  "phone_service": "Yes",
  "multiple_lines": "Yes",
  "internet_service": "Fiber optic",
  "online_security": "No",
  "online_backup": "No",
  "device_protection": "No",
  "tech_support": "No",
  "streaming_tv": "Yes",
  "streaming_movies": "Yes",
  "contract": "Month-to-month",
  "paperless_billing": "Yes",
  "payment_method": "Electronic check"
}
```

Example response structure:

```json
{
  "churn_probability": 0.8779,
  "churn_probability_percent": 87.79,
  "predicted_value": 1,
  "predicted_class": "Churn",
  "risk_level": "High",
  "threshold_used": 0.25
}
```

The response values above illustrate the API response format. Actual predictions depend on the customer input and deployed model artifact.

---

## ✅ 15. Testing Status

The application has been manually smoke-tested through:

- The FastAPI health endpoint
- The FastAPI Swagger interface
- The Streamlit prediction interface

Additional automated testing is still needed for:

- Input validation
- Invalid and missing inputs
- Feature engineering
- Model loading failures
- Prediction sanity checks
- Unexpected categories
- API error handling

---

## 💡 16. Business Recommendations

The model's predictions can help retention teams prioritize customers for investigation.

Potential actions include:

- Reviewing the needs of new customers with high churn scores.
- Investigating customer concerns related to contracts and billing.
- Reviewing service quality and technical support experiences.
- Prioritizing retention outreach according to churn risk and available capacity.
- Measuring whether retention interventions actually reduce churn.

Predictions should support business decisions rather than replace human judgment.

---

## ⚠️ 17. Limitations

- The model uses a static historical dataset.
- Observed relationships do not establish causation.
- High recall comes with a substantial number of false positives at the selected threshold.
- Probability calibration still requires evaluation.
- The business impact of retention interventions has not yet been measured.
- Automated tests and production monitoring require further development.
- Docker is not currently included in the project workflow.

---

## 🔮 18. Future Improvements

- Evaluate probability calibration.
- Investigate cost-sensitive threshold selection.
- Improve precision while maintaining acceptable recall.
- Extend SHAP explanations within the application.
- Implement automated tests for the API and model pipeline.
- Pin and manage compatible dependency versions.
- Improve model and data version management.
- Add production monitoring for prediction quality and service health.
- Evaluate the business impact of retention interventions.
- Revisit containerization when needed.

---

## 👨‍💻 19. Author

**GitHub:** [reddyvaru](https://github.com/reddyvaru)

**Repository:** [Customer Churn Prediction & Retention Intelligence System](https://github.com/reddyvaru/customer-churn-prediction-system)

**Live Application:** [Streamlit Demo](https://customer-churn-prediction-system-h2ze6yuyub2dcpkpbsmqfg.streamlit.app)