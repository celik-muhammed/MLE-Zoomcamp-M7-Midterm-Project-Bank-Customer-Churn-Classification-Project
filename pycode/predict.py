
# client_test_local

# In your other Python script or Jupyter Notebook
from preprocessor_utils import to_dict_records, convert_to_dmatrix
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

import xgboost as xgb
import pandas as pd
import pickle

def load(filename: str):
    # Load the object with pickle
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)
    
model = xgb.Booster()
model.load_model("../model/xgb_model.json")
dv = load('../model/dv.pkl')

# Load the data preprocessing object and the trained model
preprocessor = Pipeline([
    ('to_dict', FunctionTransformer(func=to_dict_records)),
    ('dv', dv),
])

# Define the client data as a dictionary
client_data = {
    'credit_score': {0: 6.429719478039138},
    'geography': {0: 'France'},
    'gender': {0: 'Female'},
    'age': {0: 3.7612001156935624},
    'tenure': {0: 2},
    'balance': {0: 0.0},
    'num_of_products': {0: 1},
    'has_cr_card': {0: 1},
    'is_active_member': {0: 1},
    'estimated_salary': {0: 11.526333967863659},
    # 'exited': {0: 1}
}

# Use the data preprocessing object to transform the client data
sample_df = pd.DataFrame.from_dict(client_data)
matrix_df = convert_to_dmatrix(preprocessor, sample_df)

# Make a probability prediction using the loaded model
probability = model.predict(matrix_df)

print(f"The probability that this client will get a credit is: {probability[0]:.3f}")
