import pandas as pd
import numpy as np
from catboost import CatBoostClassifier



test_data = pd.read_csv("../prep_golden_test_data.csv").drop("Unnamed: 0", axis=1)

model = CatBoostClassifier()
model.load_model("catboost_cred_scoring.cbm")

print("Model loaded")
predictions = model.predict_proba(test_data)[:, 1]
print(predictions[:100])