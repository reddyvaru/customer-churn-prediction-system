# Customer Churn Prediction & Retention Intelligence System

An end-to-end machine learning project that predicts customer churn probability and helps businesses identify customers who may need retention support.

The project combines Python, Scikit-learn, feature engineering, model evaluation, FastAPI, and Streamlit to build a customer churn prediction application.

## 1. Business Problem

Customer churn causes businesses to lose recurring revenue and increases the cost of acquiring replacement customers.

The objective of this project is to identify customers who are at risk of leaving so that a retention team can investigate their needs and take appropriate action.

### Business Objective
Identify high-risk customers early and support data-driven customer retention decisions.

### Machine Learning Objective
Predict the probability that a customer will churn using available customer information.

### Target Variable
`Churn Value`

- `0`: Customer did not churn
- `1`: Customer churned

### Unit of Prediction
One customer per prediction request.

## 2. Dataset

**Dataset:** IBM Telco Customer Churn

**Rows:** 7,043 customers

**Source:** [Mendeley Data — IBM Telco Customer Churn Dataset](https://data.mendeley.com/datasets/phsxg9ssrf/1)

The dataset contains customer demographics, tenure, billing information, contract details, and subscribed services.

The raw dataset is not committed to this repository. Download it from the source and place it at:

`data/raw/IBM_Telco_customer_churn_IBM_dataset.csv`

## 3. Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Mutual Information for feature ranking
- SHAP for model explainability
- XGBoost
- FastAPI
- Pydantic
- Streamlit
- Joblib
- Git and GitHub

## 4. Project Workflow

```text
Business Problem Definition
          |
          v
Dataset Inspection
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
Cross-Validation and Hyperparameter Tuning
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
```

## 5. Exploratory Data Analysis

EDA was performed to understand customer characteristics and their relationship with churn.

Key findings included:

- Customers with shorter tenure had higher churn rates.
- Month-to-month contract customers had the highest churn rate.
- Customers with higher monthly charges showed a higher churn tendency.
- Electronic Check users had the highest churn rate among payment methods.
- Fiber Optic customers had the highest churn rate among internet service types.
- Customers without Online Security or Tech Support showed higher churn rates.
- Customers with month-to-month contracts and low tenure formed an important high-risk segment.

These are observed relationships in the dataset and should not be interpreted as proof of causation.

## 6. Feature Engineering

Two additional features were created:

### Service Count
Represents the number of selected customer services.

### Tenure Group
Groups customers into four tenure ranges:

- 0–12 months
- 13–24 months
- 25–48 months
- 49–72 months

Mutual Information was also used to rank candidate features. The usefulness of the engineered and selected features was evaluated using validation performance.

## 7. Models Evaluated

The following models were evaluated:

- Dummy Classifier
- Logistic Regression
- Class-weighted Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- XGBoost

Model comparison included stratified cross-validation, hyperparameter tuning, and prediction-threshold experiments.

The models were compared using accuracy, precision, recall, F1 score, ROC-AUC, and PR-AUC.

## 8. Final Model

**Selected model:** Balanced Logistic Regression

**Decision threshold:** 0.25

The model was selected primarily because the business objective prioritizes identifying actual churners.

Class weighting was used to place greater emphasis on the minority churn class during training.

### Why Recall Matters

Missing an actual churner can mean losing a customer whom the business could have retained.

A lower decision threshold can identify more potential churners, but it also increases false positives. The selected threshold reflects this trade-off.

## 9. Final Test Results

The following results were obtained on the held-out test set using the selected model and a threshold of 0.25.

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

The model identified 264 of the 281 actual churners in the test set.

However, it also flagged 356 customers who did not churn. This is an important limitation and demonstrates why precision and recall must be considered together.

The results show strong churn detection at the chosen threshold, but the number of false positives means that further business-cost analysis is necessary before real-world use.

## 10. Model Explainability

SHAP was used to investigate how input features influence model predictions.

The explainability analysis includes:

- Global feature importance
- Feature-level contribution plots
- Analysis of individual predictions

The objective is to make model outputs more interpretable and support investigation of potential churn drivers.

Feature contributions indicate model behavior; they do not prove that a particular feature causes churn.

## 11. Application Architecture

The application has two components:

### FastAPI Backend

The API accepts customer information, validates the request, performs feature engineering, and returns a prediction.

Endpoints:

- `GET /health` — checks API availability.
- `POST /predict` — returns the estimated churn probability, predicted class, and risk level.

Interactive API documentation is available at `/docs` when the API is running locally.

### Streamlit Frontend

The Streamlit application allows users to enter customer information through a form and view the prediction results.

The interface displays:

- Estimated churn probability
- Predicted outcome
- Risk level
- Decision threshold

The displayed risk categories are application-defined rules, not separate outputs learned by the model.

## 12. Project Structure

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

The data directories are excluded from Git. The raw dataset must be downloaded separately.

## 13. Installation and Local Execution

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

From the project root:

```bash
python -m uvicorn api.main:app --reload
```

Open:

http://127.0.0.1:8000/docs

### Step 5: Start Streamlit

Open a second terminal, activate the same virtual environment, and run:

```bash
python -m streamlit run app/streamlit_app.py
```

The local Streamlit application is normally available at:

http://localhost:8501

Keep the FastAPI server running while using Streamlit.

## 14. Example API Request

Send a `POST` request to:

`http://127.0.0.1:8000/predict`

Example JSON:

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

## 15. Example Prediction

An example application response has the following structure:

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

This example illustrates the response format; actual predictions depend on the supplied customer information.

The estimated churn probability is not the probability that the model's prediction is correct. Probability calibration has not yet been established.

## 16. Business Recommendations

Potential actions informed by churn predictions include:

- Investigating new customers with high churn scores.
- Reviewing contract flexibility and renewal offers.
- Examining billing and payment experiences.
- Reviewing customer support and service issues.
- Prioritizing retention outreach according to risk and available capacity.

Predictions should support, not replace, business judgment. A retention team should validate intervention policies and track their actual impact.

## 17. Limitations

- The project uses a static historical dataset.
- Observed associations do not establish causation.
- High recall comes with a substantial number of false positives at the selected threshold.
- Probability calibration still requires evaluation.
- The retention impact of acting on predictions has not been measured.
- Cloud deployment, Docker, and production monitoring are not yet completed.

## 18. Future Improvements

- Calibrate predicted probabilities.
- Evaluate cost-sensitive threshold selection.
- Improve precision while maintaining acceptable recall.
- Extend model explainability in the application.
- Add automated tests for preprocessing and API predictions.
- Improve dependency and model-version management.
- Complete cloud deployment and monitoring.
- Measure the business impact of retention interventions.

## 19. Author

**GitHub:** [reddyvaru](https://github.com/reddyvaru)

**Repository:** [Customer Churn Prediction System](https://github.com/reddyvaru/customer-churn-prediction-system)