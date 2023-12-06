import pandas as pd
import requests
import json


def dataframe_to_json(df, batch_size):
    """
    Convert a batch of rows from a pandas DataFrame to a single JSON object with a list of records.

    :param df: The pandas DataFrame.
    :param batch_size: The number of rows in each batch.
    :return: A JSON object where each key corresponds to a list of feature values.
    """
    # Ensure the batch size does not exceed the number of rows in the DataFrame
    batch_size = min(batch_size, len(df))

    # Convert the specified batch of rows to a list of JSON formatted dictionaries
    batch_json = df.iloc[:batch_size].to_dict(orient='list')

    return batch_json

# Load your test data
test_data = pd.read_csv("../prep_golden_test_data.csv").drop("Unnamed: 0", axis=1)

# Define batch size
batch_size = 8

# Convert the DataFrame to a single JSON object containing a list of records
json_data = dataframe_to_json(test_data, batch_size)
json.dump(json_data, open("../stress_test/batch8.json", "w"))
# Print the JSON data
#print("JSON Data to Send:\n", json_data)

# # The URL of your FastAPI application
# url = 'http://localhost:8081/prediction'

# # Send the POST request
# response = requests.post(url, json=json_data)

# # Print the response from the server
# print(response.content)