from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import pandas as pd
import warnings
import joblib
from preprocess_utils import *
warnings.filterwarnings('ignore')




class EducationTypeToOrdinal(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # No fitting needed for this transformation
        return self

    def transform(self, X):
        X = X.copy()
        X["EDUCATION_TYPE_ORD"] = X["NAME_EDUCATION_TYPE"].apply(convert_education_type_to_ordinal)
        X["EDUCATION_TYPE_ORD_EXP"] = np.exp(X["EDUCATION_TYPE_ORD"])
        return X


class CategoricalToInt(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.encoders = {}

    def fit(self, X, y=None):
        for col in self.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                le = LabelEncoder()
                le.fit(X[col].astype(str))
                self.encoders[col] = le
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            if col in self.encoders:
                X[col] = self.encoders[col].transform(X[col].astype(str))
        return X


train_df = pd.read_csv("data/train_data.csv").drop(["Unnamed: 0", "TARGET"], axis=1)

CAT_FEATURES = train_df.select_dtypes("object").columns.values



cat_pipeline =  Pipeline([
    ('education_ordinal', EducationTypeToOrdinal()),
    ('label_encoding', CategoricalToInt(columns=CAT_FEATURES)),
    # Add other transformers here (e.g., for filling missing values, dropping columns)
])

print("training encoders...")
cat_pipeline.fit(train_df)
print("done..")


joblib.dump(cat_pipeline, 'cat_pipeline.pkl')

