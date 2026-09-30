# Customer Churn Prediction: Project Presentation

## Slide 1: Project Title
- Customer Churn Prediction
- End-to-End Machine Learning Deployment
- Week 4 AI Project Deployment & Capstone

## Slide 2: Problem Statement
- Customer churn affects recurring revenue and retention planning.
- Identify customers who may be at elevated risk of leaving.
- Provide an interpretable probability estimate for decision support.

## Slide 3: Project Objectives
- Build a reproducible data-to-deployment ML workflow.
- Compare Logistic Regression and Random Forest.
- Expose predictions through a Flask UI and REST API.

## Slide 4: Dataset and Features
- Public IBM Telco Customer Churn dataset; 7,043 customer records.
- Service, contract, billing, tenure, and charge attributes.
- Target: Churn (Yes/No); customerID excluded from model inputs.

## Slide 5: Data Preprocessing
- Inspect schema, missing values, duplicates, and descriptive statistics.
- Convert TotalCharges to numeric; impute missing values.
- Scale numeric fields and one-hot encode categorical fields.
- Split 80/20 with stratification and random_state=42.

## Slide 6: Exploratory Data Analysis
- Visualize churn distribution and churn across key services/contracts.
- Compare MonthlyCharges and tenure distributions.
- Review numeric feature correlations.
- Interpret plots as associations, not causal evidence.

## Slide 7: Machine Learning Models
- Logistic Regression (max_iter=1000).
- Random Forest (200 trees, random_state=42).
- Shared ColumnTransformer preprocessing within each pipeline.
- Select the model by held-out ROC-AUC, then F1 for ties.

## Slide 8: Model Evaluation
- Compare accuracy, precision, recall, F1, and ROC-AUC.
- Review the selected model's classification report and confusion matrix.
- Plot ROC curves for both models.
- Report actual test metrics from the training run.

## Slide 9: Flask Deployment and API
- Browser form at http://127.0.0.1:5000/.
- POST /predict accepts customer JSON and returns class and probability.
- Serialized joblib pipeline loads once; no per-request training.
- Validate missing fields, malformed JSON, and numeric inputs.

## Slide 10: Conclusion and Future Scope
- Delivered the data, training, evaluation, model artifact, API, and UI.
- Add threshold tuning, cross-validation, calibration, and monitoring.
- Assess fairness and secure the service before production deployment.