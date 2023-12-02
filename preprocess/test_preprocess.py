
from preprocess_utils import preprocess_df
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("test_data.csv").drop("Unnamed: 0", axis=1)

preprocess_df = preprocess_df(df)
preprocess_df.info(show_counts=False, verbose=True)

# print(preprocess_df.info())

# print(preprocess_df.head(3))