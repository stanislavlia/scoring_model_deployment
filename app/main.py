import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import FastAPI, status
import uvicorn
from pydantic import BaseModel
import xgboost
import catboost
from catboost import CatBoostClassifier
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from typing import List
from prometheus_fastapi_instrumentator import Instrumentator




## app
app = FastAPI(
)
Instrumentator().instrument(app).expose(app)

# Path of the model
BASE_DIR = Path(__file__).resolve(strict = True).parent

# Load the model
model = CatBoostClassifier()
print("Loading ...")
model.load_model("catboost_cred_scoring.cbm")
print("Model loaded successfully")




class BatchCreditScoringData(BaseModel):
    SK_ID_CURR: List[int]
    NAME_CONTRACT_TYPE: List[int]
    CODE_GENDER: List[int]
    FLAG_OWN_CAR: List[int]
    AMT_INCOME_TOTAL: List[float]
    AMT_CREDIT: List[float]
    AMT_ANNUITY: List[float]
    AMT_GOODS_PRICE: List[float]
    NAME_INCOME_TYPE: List[int]
    NAME_FAMILY_STATUS: List[int]
    REGION_POPULATION_RELATIVE: List[float]
    DAYS_BIRTH: List[int]
    DAYS_EMPLOYED: List[int]
    DAYS_REGISTRATION: List[float]
    DAYS_ID_PUBLISH: List[int]
    OWN_CAR_AGE: List[float]
    FLAG_WORK_PHONE: List[int]
    FLAG_PHONE: List[int]
    OCCUPATION_TYPE: List[int]
    REGION_RATING_CLIENT_W_CITY: List[int]
    WEEKDAY_APPR_PROCESS_START: List[int]
    REG_CITY_NOT_LIVE_CITY: List[int]
    ORGANIZATION_TYPE: List[int]
    EXT_SOURCE_1: List[float]
    EXT_SOURCE_2: List[float]
    EXT_SOURCE_3: List[float]
    DEF_30_CNT_SOCIAL_CIRCLE: List[float]
    DAYS_LAST_PHONE_CHANGE: List[float]
    FLAG_DOCUMENT_3: List[int]
    AMT_REQ_CREDIT_BUREAU_QRT: List[float]
    CREDIT_ACTIVE_mode: List[int]
    avg_DAYS_CREDIT: List[float]
    avg_DAYS_CREDIT_ENDDATE: List[float]
    avg_AMT_CREDIT_MAX_OVERDUE: List[float]
    avg_AMT_CREDIT_SUM: List[float]
    avg_meanMONTHS_BALANCE: List[float]
    prev_AMT_ANNUITY: List[float]
    prev_AMT_DOWN_PAYMENT: List[float]
    prev_AMT_GOODS_PRICE: List[float]
    prev_NAME_CONTRACT_STATUS: List[int]
    prev_DAYS_DECISION: List[float]
    prev_NAME_CLIENT_TYPE: List[int]
    prev_SELLERPLACE_AREA: List[float]
    prev_CNT_PAYMENT: List[float]
    prev_NAME_YIELD_GROUP: List[int]
    prev_DAYS_FIRST_DUE: List[float]
    prev_DAYS_TERMINATION: List[float]
    prev_NFLAG_INSURED_ON_APPROVAL: List[float]
    prev_avg_cntinstallment: List[float]
    prev_avg_cntinstallment_future: List[float]
    prev_amtcredlimit_avg: List[float]
    prev_amtpaymentcurr_avg: List[float]
    prev_amtrecivable_avg: List[float]
    prev_cntdrawingsatmcurr_avg: List[float]
    prev_cntdrawingscurr_avg: List[float]
    LOG_AMT_INCOME: List[float]
    LOG_AMT_CREDIT: List[float]
    GEOM_AVG_EXT_SOURCES12: List[float]
    AMT_INCOME_PER_CREDIT: List[float]
    DAYS_EMPLOYED_PER_CREDIT: List[float]
    EXT_SOURCE_1_prod_AMT_CREDIT: List[float]
    EXT_SOURCE_2_prod_AMT_CREDIT: List[float]
    EXT_SOURCE_3_prod_AMT_CREDIT: List[float]
    AGE_VS_JOB_EXPERIENCE: List[float]
    CURR_CONSUMPTION_RATE: List[float]
    CONSUMPTION_RATIO: List[float]
    WORKING_TIME_FRACTION: List[float]
    EDUCATION_TYPE_ORD: List[int]
    EDUCATION_TYPE_ORD_EXP: List[float]
    EXT_SOURCE_3_sum_NAME_CONTRACT_TYPE: List[float]
    DEF_30_CNT_SOCIAL_CIRCLE_sum_EXT_SOURCE_3: List[float]
    FLAG_DOCUMENT_3_sum_EXT_SOURCE_3: List[float]
    EXT_SOURCE_3_multiply_EXT_SOURCE_1: List[float]
    EXT_SOURCE_3_ratio_REGION_RATING_CLIENT_W_CITY: List[float]
    REGION_RATING_CLIENT_W_CITY_ratio_EXT_SOURCE_3: List[float]
    EXT_SOURCE_3_ratio_prev_cntdrawingsatmcurr_avg: List[float]
    prev_cntdrawingsatmcurr_avg_ratio_EXT_SOURCE_3: List[float]
    EXT_SOURCE_3_ratio_FLAG_DOCUMENT_3: List[float]
    EXT_SOURCE_3_diff_FLAG_DOCUMENT_3: List[float]



# Data validation
class CreditScoringData(BaseModel):
    SK_ID_CURR: int
    NAME_CONTRACT_TYPE: int
    CODE_GENDER: int
    FLAG_OWN_CAR: int
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float
    NAME_INCOME_TYPE: int
    NAME_FAMILY_STATUS: int
    REGION_POPULATION_RELATIVE: float
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    DAYS_REGISTRATION: float
    DAYS_ID_PUBLISH: int
    OWN_CAR_AGE: float
    FLAG_WORK_PHONE: int
    FLAG_PHONE: int
    OCCUPATION_TYPE: int
    REGION_RATING_CLIENT_W_CITY: int
    WEEKDAY_APPR_PROCESS_START: int
    REG_CITY_NOT_LIVE_CITY: int
    ORGANIZATION_TYPE: int
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    DEF_30_CNT_SOCIAL_CIRCLE: float
    DAYS_LAST_PHONE_CHANGE: float
    FLAG_DOCUMENT_3: int
    AMT_REQ_CREDIT_BUREAU_QRT: float
    CREDIT_ACTIVE_mode: int
    avg_DAYS_CREDIT: float
    avg_DAYS_CREDIT_ENDDATE: float
    avg_AMT_CREDIT_MAX_OVERDUE: float
    avg_AMT_CREDIT_SUM: float
    avg_meanMONTHS_BALANCE: float
    prev_AMT_ANNUITY: float
    prev_AMT_DOWN_PAYMENT: float
    prev_AMT_GOODS_PRICE: float
    prev_NAME_CONTRACT_STATUS: int
    prev_DAYS_DECISION: float
    prev_NAME_CLIENT_TYPE: int
    prev_SELLERPLACE_AREA: float
    prev_CNT_PAYMENT: float
    prev_NAME_YIELD_GROUP: int
    prev_DAYS_FIRST_DUE: float
    prev_DAYS_TERMINATION: float
    prev_NFLAG_INSURED_ON_APPROVAL: float
    prev_avg_cntinstallment: float
    prev_avg_cntinstallment_future: float
    prev_amtcredlimit_avg: float
    prev_amtpaymentcurr_avg: float
    prev_amtrecivable_avg: float
    prev_cntdrawingsatmcurr_avg: float
    prev_cntdrawingscurr_avg: float
    LOG_AMT_INCOME: float
    LOG_AMT_CREDIT: float
    GEOM_AVG_EXT_SOURCES12: float
    AMT_INCOME_PER_CREDIT: float
    DAYS_EMPLOYED_PER_CREDIT: float
    EXT_SOURCE_1_prod_AMT_CREDIT: float
    EXT_SOURCE_2_prod_AMT_CREDIT: float
    EXT_SOURCE_3_prod_AMT_CREDIT: float
    AGE_VS_JOB_EXPERIENCE: float
    CURR_CONSUMPTION_RATE: float
    CONSUMPTION_RATIO: float
    WORKING_TIME_FRACTION: float
    EDUCATION_TYPE_ORD: int
    EDUCATION_TYPE_ORD_EXP: float
    EXT_SOURCE_3_sum_NAME_CONTRACT_TYPE: float
    DEF_30_CNT_SOCIAL_CIRCLE_sum_EXT_SOURCE_3: float
    FLAG_DOCUMENT_3_sum_EXT_SOURCE_3: float
    EXT_SOURCE_3_multiply_EXT_SOURCE_1: float
    EXT_SOURCE_3_ratio_REGION_RATING_CLIENT_W_CITY: float
    REGION_RATING_CLIENT_W_CITY_ratio_EXT_SOURCE_3: float
    EXT_SOURCE_3_ratio_prev_cntdrawingsatmcurr_avg: float
    prev_cntdrawingsatmcurr_avg_ratio_EXT_SOURCE_3: float
    EXT_SOURCE_3_ratio_FLAG_DOCUMENT_3: float
    EXT_SOURCE_3_diff_FLAG_DOCUMENT_3: float

# home endpoint 
@app.get("/")
def home():
    return {
        "Message": "ML API for Credit Scoring",
        "Health Check ": "OK",
        "Version": "0.0.1"
    }

# # Prediction endpoint 
# @app.post("/prediction", status_code = status.HTTP_201_CREATED)
# def inference(data : CreditScoringData):
#     # Features dictionary 
#     features = data.dict()

#     # feature dataframe
#     features = pd.DataFrame(features, index = [0])

#     # Inference - Predictions
#     pred = model.predict(features)
#     pred_prob = model.predict_proba(features)[:, 1]


#     #prob_default = pred_prob[0, 1]
#     print(pred_prob[:10])
#     print(list(features["SK_ID_CURR"].values))
#     predictions_d =  {
#         "predictions" : [float(p) for p in pred_prob],
#         "SK_IDS_CURR" : [int(id) for id in features["SK_ID_CURR"].values]
#     }

#     print(predictions_d)
#     return JSONResponse(content=predictions_d)

@app.post("/prediction", status_code=status.HTTP_201_CREATED)
def inference(data: BatchCreditScoringData):
    # Convert the incoming batch data to a DataFrame
    features = pd.DataFrame(data.dict())

    # Inference - Predictions
    pred_prob = model.predict_proba(features)[:, 1]

    predictions_d = {
        "predictions": pred_prob.tolist(),
        "SK_IDS_CURR": data.SK_ID_CURR
    }

    return JSONResponse(content=jsonable_encoder(predictions_d))
