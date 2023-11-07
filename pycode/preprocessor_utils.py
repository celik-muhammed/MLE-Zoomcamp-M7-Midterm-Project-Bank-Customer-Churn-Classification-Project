
# preprocessor_utils.py

import pandas as pd
import xgboost as xgb

def to_dict_records(df: pd.DataFrame = pd.DataFrame()):
    return df.to_dict(orient='records')

def convert_to_dmatrix(preprocessor, X_df, y=None, class_weights={0: 0.6278449223041909, 1: 2.4554941682013505}):
    data = preprocessor.transform(X_df)
    feature_names = preprocessor[-1].feature_names_

    weight = [class_weights[i] for i in y] if y is not None else None
    dmatrix = xgb.DMatrix(data=data, label=y, feature_names=feature_names, weight=weight)
    return dmatrix
