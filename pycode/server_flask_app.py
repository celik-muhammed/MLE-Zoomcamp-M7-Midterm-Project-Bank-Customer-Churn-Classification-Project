
from flask import Flask, request, jsonify

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

# Load the data preprocessing object and the trained model
preprocessor = Pipeline([
    ('to_dict', FunctionTransformer(func=to_dict_records)),
    ('dv', load('dv.pkl')),
])
    
model = xgb.Booster()
model.load_model("xgb_model.json")

app = Flask('get-credit')

@app.route('/predict', methods=['POST'])
def predict():
    # client = request.get_json()
    client = request.json

    # Use the data preprocessing object to transform the client data
    sample_df = pd.DataFrame.from_dict(client)
    matrix_df = convert_to_dmatrix(preprocessor, sample_df)

    # Make a probability prediction using the loaded model
    probability = model.predict(matrix_df)

    result = {
        'get_credit_probability': float(probability),
        'get_credit': bool(probability >= 0.5)
    }
    return jsonify(result)

if __name__ == '__main__': 
    app.run(debug=True, host='0.0.0.0', port=9696, use_reloader=False)
