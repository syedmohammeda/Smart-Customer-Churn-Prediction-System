"""Integration smoke tests for the Flask app; train the model before running."""

import unittest

import app as churn_app


SAMPLE_CUSTOMER = {
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
    "TotalCharges": 350.0,
}


class ChurnAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if churn_app.MODEL is None:
            raise RuntimeError("Run python train_model.py before the Flask integration tests.")
        cls.client = churn_app.app.test_client()

    def test_homepage(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Customer churn", response.data)

    def test_valid_prediction(self):
        response = self.client.post("/predict", json=SAMPLE_CUSTOMER)
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertIn(body["prediction"], {"Yes", "No"})
        self.assertGreaterEqual(body["churn_probability"], 0)
        self.assertLessEqual(body["churn_probability"], 1)

    def test_missing_fields(self):
        response = self.client.post("/predict", json={"gender": "Male"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("Missing required fields", response.get_json()["error"])

    def test_invalid_numeric_input(self):
        customer = SAMPLE_CUSTOMER.copy()
        customer["tenure"] = "not-a-number"
        response = self.client.post("/predict", json=customer)
        self.assertEqual(response.status_code, 400)
        self.assertIn("tenure", response.get_json()["error"])

    def test_malformed_json(self):
        response = self.client.post(
            "/predict", data="{not valid json", content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()