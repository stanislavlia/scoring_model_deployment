
from preprocess_utils import preprocess_df
import pandas as pd

df = pd.read_csv("test_data.csv").drop("Unnamed: 0", axis=1)

print(df.columns)

preprocess_df = preprocess_df(df)


print(preprocess_df.info())

print(preprocess_df.head(3))