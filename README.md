# Customer Churn Prediction

## Project Overview

An end-to-end machine-learning capstone that estimates whether a Telco customer will churn. It includes data inspection and cleaning, exploratory plots, preprocessing, a held-out comparison of Logistic Regression and Random Forest, serialized artifacts, a Flask interface, and a JSON prediction endpoint.

## Problem Statement

Customer churn reduces recurring revenue and makes retention planning harder. This application uses service choices, contract details, tenure, and charges to estimate a customer's churn class and probability. The estimate is a decision-support signal, not a guarantee.

## Objectives

- Demonstrate a reproducible Python ML workflow from raw CSV to deployed prediction.
- Compare two classifiers using the same stratified 80/20 train/test split.
- Use a fitted scikit-learn preprocessing/model pipeline for consistent inference.
- Serve predictions through a responsive form and a validated Flask REST API.

## Dataset

`data/customer_churn.csv` contains the public IBM Telco Customer Churn dataset (7,043 customer records, including `customerID` and target `Churn`). The CSV is included. Source: [IBM Telco Customer Churn sample](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv).

To download a fresh copy, use PowerShell from the project root:

```powershell
New-Item -ItemType Directory -Force data
Invoke-WebRequest -UseBasicParsing `
  -Uri "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv" `
  -OutFile "data/customer_churn.csv"
```

`TotalCharges` contains blank strings for some new accounts. The training script parses it as numeric; the numeric pipeline imputes missing values using the training-set median. The customer ID is removed from model features.

## Technologies Used

Python 3.10+, pandas, NumPy, scikit-learn, matplotlib, seaborn, joblib, Flask, HTML, and CSS.

## Project Structure

```text
customer-churn-prediction/
├── data/customer_churn.csv
├── models/churn_model.pkl             # created by training
├── models/preprocessor.pkl             # created by training
├── notebooks/customer_churn_analysis.ipynb
├── static/style.css
├── templates/index.html
├── tests/test_app.py
├── app.py
├── train_model.py
├── predict.py
├── requirements.txt
├── README.md
├── presentation.md
└── .gitignore
```

Training also writes EDA and evaluation PNGs to `static/`. The complete model artifact contains the fitted preprocessor and classifier so inference uses the same transformations as training.

## Data Preprocessing

The training script displays shape, data types, missing values, duplicate count, and descriptive statistics. It removes duplicate records and `customerID`, maps `Churn` from Yes/No to 1/0, coerces `TotalCharges` to numeric, and imputes missing values. Numeric columns use median imputation and `StandardScaler`; categorical columns use most-frequent imputation and `OneHotEncoder(handle_unknown="ignore")`. All processing lives in a `ColumnTransformer` inside each model pipeline.

## Exploratory Data Analysis

The training script saves plots for churn distribution, churn by contract, internet service and payment method, MonthlyCharges and tenure distributions, and a numeric correlation heatmap. These are generated from the included dataset when training runs. Use the notebook for interactive exploration and interpretation; typical dataset-level patterns to inspect include higher churn for month-to-month contracts and shorter-tenure customers. Verify those patterns against the generated plots rather than treating them as guaranteed causal findings.

## Machine Learning Models

- Logistic Regression with `max_iter=1000`.
- Random Forest with 200 trees and `random_state=42`.

Both use the same stratified 80/20 split (`random_state=42`). The selected model is the one with the highest test ROC-AUC; F1 score breaks exact ties. Metrics are calculated from the test set at runtime, never hard-coded.

## Model Evaluation

The comparison includes accuracy, precision, recall, F1, ROC-AUC, and a selected-model classification report. A confusion matrix and ROC plot are saved under `static/`. These scores estimate performance on this one held-out split and should not be interpreted as a production guarantee.

## Model Serialization

`joblib` writes the complete winning pipeline to `models/churn_model.pkl` and its fitted preprocessing component to `models/preprocessor.pkl`. The Flask process loads the pipeline once at startup; it does not train on application startup or per request.

## Flask API

The application provides a form at `http://127.0.0.1:5000/` and a JSON endpoint at `POST http://127.0.0.1:5000/predict`. Missing fields, malformed JSON, and invalid numeric values return HTTP 400. If no trained model exists, prediction endpoints return HTTP 503 without exposing an internal traceback.

### API Endpoint

`POST /predict` with `Content-Type: application/json`.

### Example API Request

```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 5,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.5,
  "TotalCharges": 350.0
}
```

## How to Install

Open this folder in VS Code. In the integrated PowerShell terminal:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run `venv\Scripts\python.exe -m pip install -r requirements.txt` and use `venv\Scripts\python.exe` for the commands below.

## How to Run

Train and generate the model files and plots:

```powershell
python train_model.py
```

Start the local application:

```powershell
python app.py
```

Open [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in a browser. Stop the development server with Ctrl+C. Flask debug mode is disabled by default.

Try the command-line example after training:

```powershell
python predict.py
```

## How to Test

After training, run the built-in Flask integration tests:

```powershell
python -m unittest discover -s tests -v
```

They cover the homepage, a valid prediction, missing fields, malformed JSON, and invalid numeric input. A valid API request can also be tested with Python:

```powershell
python -c "import requests; p={'gender':'Male','SeniorCitizen':0,'Partner':'Yes','Dependents':'No','tenure':5,'PhoneService':'Yes','MultipleLines':'No','InternetService':'DSL','OnlineSecurity':'No','OnlineBackup':'Yes','DeviceProtection':'No','TechSupport':'No','StreamingTV':'No','StreamingMovies':'No','Contract':'Month-to-month','PaperlessBilling':'Yes','PaymentMethod':'Electronic check','MonthlyCharges':70.5,'TotalCharges':350.0}; r=requests.post('http://127.0.0.1:5000/predict',json=p); print(r.status_code, r.json())"
```

Equivalent curl example:

```powershell
curl.exe -X POST http://127.0.0.1:5000/predict `
  -H "Content-Type: application/json" `
  -d '{"gender":"Male","SeniorCitizen":0,"Partner":"Yes","Dependents":"No","tenure":5,"PhoneService":"Yes","MultipleLines":"No","InternetService":"DSL","OnlineSecurity":"No","OnlineBackup":"Yes","DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":70.5,"TotalCharges":350.0}'
```

## Example API Response

The prediction and probability depend on the trained model; this shows the response shape, not a promised result:

```json
{
  "prediction": "No",
  "churn_probability": 0.1842,
  "message": "Customer is not likely to churn."
}
```

## Screenshots Placeholder

Add a screenshot of the running form and prediction result here after launching the app.

## Results

The measured model comparison, winning model, and test metrics print at the end of `python train_model.py`. The script also saves the evaluation plots in `static/`. No scores are stated here in advance because they are calculated from the dataset on execution.

## Future Improvements

- Tune the decision threshold using business costs for false positives and false negatives.
- Validate with cross-validation and monitor data or concept drift.
- Add authentication, request rate limits, and production server configuration before public deployment.
- Collect feedback and assess fairness and calibration across customer groups.

## Conclusion

This capstone demonstrates the full path from a real public customer dataset to evaluated and serialized classification pipelines and a locally deployable Flask prediction application.

## GitHub Upload

From the project root, create an empty GitHub repository, then run:

```powershell
git init
git add .
git commit -m "Build customer churn prediction capstone"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Review the generated model size before publishing. The model and CSV are intentionally not ignored for this academic project; omit generated artifacts only if repository hosting limits require it, and document that `python train_model.py` recreates them.