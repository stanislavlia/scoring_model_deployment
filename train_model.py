from catboost import CatBoostClassifier
from preprocess.preprocess_utils import *
import joblib

optuna_params_catboost = {
        "learning_rate": 0.1,
        "depth": 5,
        "l2_leaf_reg": 2.885007079099666,
        "random_strength": 2.0633263532281965,
        "rsm": 0.16528340110846054,
        "min_data_in_leaf": 92,
        "eval_metric": "AUC",
        "num_boost_round": 800,
        "early_stopping_rounds": 50,
    }




