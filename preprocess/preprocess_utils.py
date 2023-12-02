import numpy as np
import pydantic
from sklearn.preprocessing import LabelEncoder
import pandas as pd

PREPROCESSING_CONFIG = {
        "NAME_CONTRACT_TYPE": "categorical_to_int",
        "CODE_GENDER": "categorical_to_int",
        "FLAG_OWN_CAR": "categorical_to_int",
        "AMT_ANNUITY": "na_fill_median",
        "AMT_GOODS_PRICE": "na_fill_median",
        "NAME_INCOME_TYPE": "categorical_to_int",
        "NAME_EDUCATION_TYPE": "categorical_to_int",
        "NAME_FAMILY_STATUS": "categorical_to_int",
        "OWN_CAR_AGE": "na_fill_median",
        "OCCUPATION_TYPE": "categorical_to_int",
        "WEEKDAY_APPR_PROCESS_START": "categorical_to_int",
        "ORGANIZATION_TYPE": "categorical_to_int",
        "EXT_SOURCE_1": "na_fill_median",
        "EXT_SOURCE_2": "na_fill_median",
        "EXT_SOURCE_3": "na_fill_median",
        "DEF_30_CNT_SOCIAL_CIRCLE": "na_fill_median",
        "DAYS_LAST_PHONE_CHANGE": "na_fill_median",
        "AMT_REQ_CREDIT_BUREAU_QRT": "na_fill_median",
        "CREDIT_ACTIVE_mode": "categorical_to_int",
        "avg_DAYS_CREDIT": "na_fill_median",
        "avg_DAYS_CREDIT_ENDDATE": "na_fill_median",
        "avg_AMT_CREDIT_MAX_OVERDUE": "na_fill_median",
        "avg_AMT_CREDIT_SUM": "na_fill_median",
        "avg_meanMONTHS_BALANCE": "na_fill_median",
        "prev_AMT_ANNUITY": "na_fill_median",
        "prev_AMT_DOWN_PAYMENT": "na_fill_median",
        "prev_AMT_GOODS_PRICE": "na_fill_median",
        "prev_NAME_CONTRACT_STATUS": "categorical_to_int",
        "prev_DAYS_DECISION": "na_fill_median",
        "prev_NAME_CLIENT_TYPE": "categorical_to_int",
        "prev_SELLERPLACE_AREA": "na_fill_median",
        "prev_CNT_PAYMENT": "na_fill_median",
        "prev_NAME_YIELD_GROUP": "categorical_to_int",
        "prev_DAYS_FIRST_DUE": "na_fill_median",
        "prev_DAYS_TERMINATION": "na_fill_median",
        "prev_NFLAG_INSURED_ON_APPROVAL": "na_fill_median",
        "prev_avg_cntinstallment": "na_fill_median",
        "prev_avg_cntinstallment_future": "na_fill_median",
        "prev_amtcredlimit_avg": "na_fill_median",
        "prev_amtpaymentcurr_avg": "na_fill_median",
        "prev_amtrecivable_avg": "na_fill_median",
        "prev_cntdrawingsatmcurr_avg": "na_fill_median",
        "prev_cntdrawingscurr_avg": "na_fill_median",
        "prev_skdpd_min": "remove_column",
        "prev_skdpddef_min": "remove_column",
        "GEOM_AVG_EXT_SOURCES12": "na_fill_median",
        "EXT_SOURCE_1_prod_AMT_CREDIT": "na_fill_median",
        "EXT_SOURCE_2_prod_AMT_CREDIT": "na_fill_median",
        "EXT_SOURCE_3_prod_AMT_CREDIT": "na_fill_median",
        "CURR_CONSUMPTION_RATE": "na_fill_median",
        "CONSUMPTION_RATIO": "na_fill_median",
        "NAME_CONTRACT_TYPE" : "categorical_to_int",
        "CODE_GENDER" : "categorical_to_int",
       "FLAG_OWN_CAR" : "categorical_to_int",
        "NAME_INCOME_TYPE" : "categorical_to_int",
        "NAME_INCOME_TYPE" : "categorical_to_int",
        "NAME_INCOME_TYPE" : "categorical_to_int",
        "NAME_INCOME_TYPE" : "categorical_to_int",
        "NAME_EDUCATION_TYPE" : "remove_column",
        "OCCUPATION_TYPE" : "categorical_to_int",
        "NAME_FAMILY_STATUS" : "categorical_to_int",
        "WEEKDAY_APPR_PROCESS_START" : "categorical_to_int",
        "ORGANIZATION_TYPE" : "categorical_to_int",
        "NAME_FAMILY_STATUS" : "categorical_to_int",
        "CREDIT_ACTIVE_mode" : "categorical_to_int",
        "prev_NAME_CONTRACT_STATUS" : "categorical_to_int",
        "prev_NAME_CLIENT_TYPE" : "categorical_to_int",
        "prev_NAME_YIELD_GROUP" : "categorical_to_int",

    }

REDUNDANT_FEATURES = ['prev_amtrecivable_avg', 'prev_DAYS_TERMINATION', 'prev_cntdrawingscurr_max', 'prev_NAME_PORTFOLIO',  'prev_CODE_REJECT_REASON', 'YEARS_BEGINEXPLUATATION_MEDI', 'DEF_60_CNT_SOCIAL_CIRCLE',
                       'prev_amtrecprinc_avg', 'prev_amtcredlimit_avg', 'prev_DAYS_FIRST_DUE', 'prev_NFLAG_INSURED_ON_APPROVAL', 'avg_DAYS_ENDDATE_FACT', 'prev_DAYS_LAST_DUE', 'prev_AMT_APPLICATION', 'REGION_POPULATION_RELATIVE', 'prev_avg_poscashMONTH_BALANCE', 
                       'FLAG_PHONE', 'prev_DAYS_LAST_DUE_1ST_VERSION', 'YEARS_BEGINEXPLUATATION_MODE', 'prev_RATE_DOWN_PAYMENT', 'avg_meanMONTHS_BALANCE', 'TOTALAREA_MODE', 'AMT_REQ_CREDIT_BUREAU_YEAR', 'prev_amtpaymentcurr_avg', 
                       'prev_NAME_GOODS_CATEGORY', 'FLAG_DOCUMENT_18', 'FLOORSMAX_AVG', 'ENTRANCES_AVG', 'prev_amtdrawingsATMcurr_avg', 'prev_amtpaymenttotalcurr_avg', 'LIVINGAREA_MEDI', 'prev_amtpaymentcurr_max', 'prev_AMT_CREDIT', 'HOUR_APPR_PROCESS_START', 
                       'REGION_RATING_CLIENT', 'FLAG_DOCUMENT_16', 'prev_CHANNEL_TYPE', 'YEARS_BEGINEXPLUATATION_AVG', 'avg_AMT_CREDIT_SUM_OVERDUE', 'prev_amtrecivable_max', 'FLOORSMAX_MEDI', 'prev_amtrecivable_min', 'LIVINGAREA_AVG', 'prev_amtrecprinc_min', 'ENTRANCES_MEDI', 
                       'prev_amtinstminreg_max', 'prev_PRODUCT_COMBINATION', 'prev_amttotrec_avg', 'prev_cntinstalmaturecum_avg', 'NAME_HOUSING_TYPE', 'LIVINGAREA_MODE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'APARTMENTS_MODE', 'prev_NAME_PRODUCT_TYPE', 'OBS_30_CNT_SOCIAL_CIRCLE', 'FONDKAPREMONT_MODE', 
                       'prev_WEEKDAY_APPR_PROCESS_START', 'prev_amtpaymenttotalcurr_max', 'prev_amttotrec_max', 'prev_amtrecprinc_max', 'BASEMENTAREA_MEDI', 'LANDAREA_MODE', 'BASEMENTAREA_AVG', 'prev_cntdrawingsatmcurr_max', 'CNT_FAM_MEMBERS', 'prev_NAME_SELLER_INDUSTRY', 'prev_skdpddef_avg',
                         'prev_avgMobBAL', 'CREDIT_CURRENCY_mode', 'prev_cntinstalmaturecum_max', 'YEARS_BUILD_AVG', 'FLOORSMIN_AVG', 'prev_DAYS_FIRST_DRAWING', 'EMERGENCYSTATE_MODE', 'APARTMENTS_AVG', 'prev_amtinstminreg_avg', 'prev_cntdrawingsposcurr_avg', 'prev_cntdrawingsposcurr_max', 
                         'ELEVATORS_AVG', 'LANDAREA_AVG', 'COMMONAREA_MEDI', 'HOUSETYPE_MODE', 'prev_cntdrawingsposcurr_min', 'APARTMENTS_MEDI', 'avg_DAYS_ENDDATE_FACT:1', 'prev_last_name_status', 'prev_NAME_CONTRACT_TYPE', 'prev_NAME_CASH_LOAN_PURPOSE', 'LIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG',
                           'YEARS_BUILD_MEDI', 'FLAG_DOCUMENT_14', 'LIVINGAPARTMENTS_MODE', 'BASEMENTAREA_MODE', 'prev_amtdrawings_poscurrent', 'FLOORSMAX_MODE', 'prev_NAME_PAYMENT_TYPE', 'WALLSMATERIAL_MODE', 'FLAG_DOCUMENT_13', 'prev_NAME_TYPE_SUITE', 'YEARS_BUILD_MODE', 'prev_mode_name_status', 
                           'CNT_CHILDREN', 'COMMONAREA_MODE', 'REG_CITY_NOT_WORK_CITY', 'COMMONAREA_AVG', 'prev_amtpaymentcurr_min', 'ENTRANCES_MODE', 'LIVINGAPARTMENTS_MEDI', 'prev_amtdrawingsothercurrent_avg', 'NAME_TYPE_SUITE', 'LANDAREA_MEDI', 'NONLIVINGAREA_MODE', 'prev_skdpd_avg', 'AMT_REQ_CREDIT_BUREAU_WEEK', 
                           'prev_cntdrawingsothercurr_avg', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAPARTMENTS_AVG', 'FLOORSMIN_MEDI', 'prev_skdpd_max', 'NONLIVINGAREA_MEDI', 'prev_FLAG_LAST_APPL_PER_CONTRACT', 'prev_RATE_INTEREST_PRIMARY', 'prev_skdpddef_max', 'FLOORSMIN_MODE', 'prev_NFLAG_LAST_APPL_IN_DAY',
                             'FLAG_DOCUMENT_6', 'prev_amttotrec_min', 'FLAG_OWN_REALTY', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_MON', 'prev_cntinstalmaturecum_min', 'prev_RATE_INTEREST_PRIVILEGED', 'FLAG_DOCUMENT_9', 'FLAG_EMAIL', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_8', 'prev_amtinstminreg_min',
                               'ELEVATORS_MODE', 'REG_REGION_NOT_LIVE_REGION', 'prev_amtpaymenttotalcurr_min', 'prev_cntdrawingsothercurr_max', 'LIVE_REGION_NOT_WORK_REGION', 'LIVE_CITY_NOT_WORK_CITY', 'prev_cntdrawingscurr_min', 'prev_cntdrawingsatmcurr_min', 'FLAG_DOCUMENT_21', 'FLAG_DOCUMENT_15', 'FLAG_EMP_PHONE', 'FLAG_MOBIL', 'FLAG_CONT_MOBILE', 'AMT_REQ_CREDIT_BUREAU_HOUR', 
                               'FLAG_DOCUMENT_2', 'prev_cntdrawingsothercurr_min', 'REG_REGION_NOT_WORK_REGION', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_10', 'ELEVATORS_MEDI', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_20'
                               ]



"""Expected data for model

Data columns (total 58 columns): After dropping redundant feats by SHAP (treshold = 0.01)
 #   Column                          Non-Null Count  Dtype  
---  ------                          --------------  -----  
 0   SK_ID_CURR                      48744 non-null  int64  
 1   NAME_CONTRACT_TYPE              48744 non-null  object 
 2   CODE_GENDER                     48744 non-null  object 
 3   FLAG_OWN_CAR                    48744 non-null  object 
 4   AMT_INCOME_TOTAL                48744 non-null  float64
 5   AMT_CREDIT                      48744 non-null  float64
 6   AMT_ANNUITY                     48720 non-null  float64
 7   AMT_GOODS_PRICE                 48744 non-null  float64
 8   NAME_INCOME_TYPE                48744 non-null  object 
 9   NAME_EDUCATION_TYPE             48744 non-null  object 
 10  NAME_FAMILY_STATUS              48744 non-null  object 
 11  REGION_POPULATION_RELATIVE      48744 non-null  float64
 12  DAYS_BIRTH                      48744 non-null  int64  
 13  DAYS_EMPLOYED                   48744 non-null  int64  
 14  DAYS_REGISTRATION               48744 non-null  float64
 15  DAYS_ID_PUBLISH                 48744 non-null  int64  
 16  OWN_CAR_AGE                     16432 non-null  float64
 17  FLAG_WORK_PHONE                 48744 non-null  int64  
 18  FLAG_PHONE                      48744 non-null  int64  
 19  OCCUPATION_TYPE                 48744 non-null  object 
 20  REGION_RATING_CLIENT_W_CITY     48744 non-null  int64  
 21  WEEKDAY_APPR_PROCESS_START      48744 non-null  object 
 22  REG_CITY_NOT_LIVE_CITY          48744 non-null  int64  
 23  ORGANIZATION_TYPE               48744 non-null  object 
 24  EXT_SOURCE_1                    28212 non-null  float64
 25  EXT_SOURCE_2                    48736 non-null  float64
 26  EXT_SOURCE_3                    40076 non-null  float64
 27  DEF_30_CNT_SOCIAL_CIRCLE        48715 non-null  float64
 28  DAYS_LAST_PHONE_CHANGE          48744 non-null  float64
 29  FLAG_DOCUMENT_3                 48744 non-null  int64  
 30  AMT_REQ_CREDIT_BUREAU_QRT       42695 non-null  float64
 31  CREDIT_ACTIVE_mode              48744 non-null  object 
 32  avg_DAYS_CREDIT                 42320 non-null  float64
 33  avg_DAYS_CREDIT_ENDDATE         41984 non-null  float64
 34  avg_AMT_CREDIT_MAX_OVERDUE      29085 non-null  float64
 35  avg_AMT_CREDIT_SUM              42319 non-null  float64
 36  avg_meanMONTHS_BALANCE          42311 non-null  float64
 37  prev_AMT_ANNUITY                47737 non-null  float64
 38  prev_AMT_DOWN_PAYMENT           45148 non-null  float64
 39  prev_AMT_GOODS_PRICE            47711 non-null  float64
 40  prev_NAME_CONTRACT_STATUS       48744 non-null  object 
 41  prev_DAYS_DECISION              47800 non-null  float64
 42  prev_NAME_CLIENT_TYPE           48744 non-null  object 
 43  prev_SELLERPLACE_AREA           47800 non-null  float64
 44  prev_CNT_PAYMENT                47737 non-null  float64
 45  prev_NAME_YIELD_GROUP           48744 non-null  object 
 46  prev_DAYS_FIRST_DUE             47580 non-null  float64
 47  prev_DAYS_TERMINATION           47580 non-null  float64
 48  prev_NFLAG_INSURED_ON_APPROVAL  47580 non-null  float64
 49  prev_avg_cntinstallment         47387 non-null  float64
 50  prev_avg_cntinstallment_future  47387 non-null  float64
 51  prev_amtcredlimit_avg           14513 non-null  float64
 52  prev_amtpaymentcurr_avg         9099 non-null   float64
 53  prev_amtrecivable_avg           14513 non-null  float64
 54  prev_cntdrawingsatmcurr_avg     9094 non-null   float64
 55  prev_cntdrawingscurr_avg        14513 non-null  float64
 56  prev_skdpd_min                  14513 non-null  float64
 57  prev_skdpddef_min               14513 non-null  float64

dtypes: float64(36), int64(9), object(13)
"""

def drop_redundant_feats(df):
   
    df = df.drop(REDUNDANT_FEATURES, axis=1)
    return df

def replace_missing_cat(df):
    
    cat_feature = list(df.select_dtypes(["object"]).columns)

    for col in cat_feature:
        df[col] = df[col].fillna("MISSING")
        df[col] = df[col].fillna("MISSING")

    return df

def compute_hand_crafted_feats(df):
    df["LOG_AMT_INCOME"] = np.log(df["AMT_INCOME_TOTAL"])
    df["LOG_AMT_CREDIT"] = np.log(df["AMT_CREDIT"])
    df["GEOM_AVG_EXT_SOURCES12"] = np.sqrt(df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"])
    df["AMT_INCOME_PER_CREDIT"] = df["AMT_INCOME_TOTAL"] / (df["AMT_CREDIT"] + 0.0001)
    df["DAYS_EMPLOYED_PER_CREDIT"] = df["DAYS_EMPLOYED"] / (df["AMT_CREDIT"] + 0.0001)
    df["EXT_SOURCE_1_prod_AMT_CREDIT"] = df["EXT_SOURCE_1"] * df["AMT_CREDIT"]
    df["EXT_SOURCE_2_prod_AMT_CREDIT"] = df["EXT_SOURCE_2"] * df["AMT_CREDIT"]
    df["EXT_SOURCE_3_prod_AMT_CREDIT"] = df["EXT_SOURCE_3"] * df["AMT_CREDIT"]
    df["AGE_VS_JOB_EXPERIENCE"] = df["DAYS_BIRTH"] / (df["DAYS_EMPLOYED"] + 0.0001)
    df["CURR_CONSUMPTION_RATE"] = df["AMT_GOODS_PRICE"] / (df["AMT_CREDIT"] + 0.0001)
    df["CONSUMPTION_RATIO"] = df["AMT_GOODS_PRICE"] / (df["prev_AMT_GOODS_PRICE"] + 0.0001)
    df["WORKING_TIME_FRACTION"] = df["DAYS_EMPLOYED"] / (df["DAYS_BIRTH"] + 0.0001)
    
    return df

def compute_golden_feats(df):
    # Sum Operations
    df['EXT_SOURCE_3_sum_NAME_CONTRACT_TYPE'] = df['EXT_SOURCE_3'] + df['NAME_CONTRACT_TYPE']
    df['DEF_30_CNT_SOCIAL_CIRCLE_sum_EXT_SOURCE_3'] = df['DEF_30_CNT_SOCIAL_CIRCLE'] + df['EXT_SOURCE_3']
    df['FLAG_DOCUMENT_3_sum_EXT_SOURCE_3'] = df['FLAG_DOCUMENT_3'] + df['EXT_SOURCE_3']

    # Multiply Operation
    df['EXT_SOURCE_3_multiply_EXT_SOURCE_1'] = df['EXT_SOURCE_3'] * df['EXT_SOURCE_1']

    # Ratio Operations (with a small number to avoid division by zero)
    df['EXT_SOURCE_3_ratio_REGION_RATING_CLIENT_W_CITY'] = df['EXT_SOURCE_3'] / (df['REGION_RATING_CLIENT_W_CITY'] + 0.0001)
    df['REGION_RATING_CLIENT_W_CITY_ratio_EXT_SOURCE_3'] = df['REGION_RATING_CLIENT_W_CITY'] / (df['EXT_SOURCE_3'] + 0.0001)
    df['EXT_SOURCE_3_ratio_prev_cntdrawingsatmcurr_avg'] = df['EXT_SOURCE_3'] / (df['prev_cntdrawingsatmcurr_avg'] + 0.0001)
    df['prev_cntdrawingsatmcurr_avg_ratio_EXT_SOURCE_3'] = df['prev_cntdrawingsatmcurr_avg'] / (df['EXT_SOURCE_3'] + 0.0001)
    df['EXT_SOURCE_3_ratio_FLAG_DOCUMENT_3'] = df['EXT_SOURCE_3'] / (df['FLAG_DOCUMENT_3'] + 0.0001)

    # Difference Operation
    df['EXT_SOURCE_3_diff_FLAG_DOCUMENT_3'] = df['EXT_SOURCE_3'] - df['FLAG_DOCUMENT_3']

    return df

def convert_education_type_to_ordinal(education_type):
    """
    Convert education types to ordinal numbers.

    Parameters:
    education_type (str): The education type as a string.

    Returns:
    int: Ordinal number representing the education level.
    """
    if education_type in [0, 1, 2, 3, 4, 5]:
      return education_type
    mapping = {
        'Lower secondary': 1,
        'Secondary / secondary special': 2,
        'Incomplete higher': 3,
        'Higher education': 4,
        'Academic degree': 5
    }

    return mapping.get(education_type, 0)

def convert_types(df):
    #Conver education + create 1 more feature
    df["EDUCATION_TYPE_ORD"] = df["NAME_EDUCATION_TYPE"].apply(convert_education_type_to_ordinal)
    df["EDUCATION_TYPE_ORD_EXP"] =  np.exp(df["EDUCATION_TYPE_ORD"])


    # Helper function to convert categorical columns to integers
    def categorical_to_int(column):
        if df[column].dtype == 'object' or 'category':
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].astype(str))

    # Iterate over each column and apply the specified preprocessing steps
    for column in PREPROCESSING_CONFIG.keys():

        if PREPROCESSING_CONFIG[column] == "categorical_to_int":
            categorical_to_int(column)
        elif PREPROCESSING_CONFIG[column] == "na_fill_median":
            df[column] = df[column].fillna(df[column].median())
        elif PREPROCESSING_CONFIG[column] == "remove_column":
            df.drop(column, axis=1, inplace=True)


    return df

def preprocess_df(df):

    #replace nan with "MISSING" class
    df = replace_missing_cat(df)
    
    #Feature engineering
    df = compute_hand_crafted_feats(df)

    #categorical encoding & imputing nans
    df = convert_types(df)
    
    df = compute_golden_feats(df)
    

    if (len(df.columns) != 58):
            #drop redundant features    
            df = drop_redundant_feats(df)
            if ("Unnamed: 0" in list(df.columns)):
                df = df.drop("Unnamed: 0", axis=1)
    
    

    return df
