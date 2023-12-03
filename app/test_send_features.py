import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import requests


def row_to_json(df, row_index):
    """
    Convert a row from a pandas DataFrame to JSON format.
    
    :param df: The pandas DataFrame.
    :param row_index: The index of the row to convert.
    :return: A JSON string representation of the row.
    """
    if row_index < 0 or row_index >= len(df):
        raise ValueError("Row index out of bounds")

    # Convert the specified row to a JSON formatted dictionary
    row_json = df.iloc[row_index].to_dict()

    return row_json

# Load your test data
test_data = pd.read_csv("../prep_golden_test_data.csv").drop("Unnamed: 0", axis=1)

# Convert a row to JSON
json_data = row_to_json(test_data, 0)  # Assuming you want to send the first row


print("JSON TO SEND:\n", json_data)


# The URL of your FastAPI application
url = 'http://localhost:2080/prediction'

# Send the POST request
response = requests.post(url, json=json_data)

# Print the response from the server
print(response.text)
