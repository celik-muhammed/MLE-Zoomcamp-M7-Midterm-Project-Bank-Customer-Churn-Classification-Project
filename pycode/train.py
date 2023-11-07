
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import pandas as pd
import numpy as np
import pickle
import sys
from contextlib import redirect_stdout

# Read the CSV file
SEED = 42
zip_filepath = f'data/archive.zip'
df = pd.read_csv(zip_filepath)
df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])
cont_features = df.select_dtypes('number').apply(np.ptp)[(df.select_dtypes('number').apply(np.ptp) > 10)].index
df.loc[:, cont_features] = df[cont_features].apply(np.log1p)

# Define features and target variable
y = df['Exited'].values
X = df.drop('Exited', axis=1)
# Create a DMatrix for XGBoost



# Define a function to transform DataFrame to dictionary records
def to_dict_records(df: pd.DataFrame = pd.DataFrame()):
    return df.to_dict(orient='records')

# Create a preprocessor to handle categorical columns with one-hot encoding
preprocessor = Pipeline([
    ('to_dict', FunctionTransformer(func=to_dict_records)),
    ('dv', DictVectorizer(sparse=True)),
]).fit(X, y)

# Save the model to a pickle file
with open("model/dv2.pkl", "wb") as f_out:
    pickle.dump(preprocessor[-1], f_out)



def convert_to_dmatrix(preprocessor, X_df, y=None, class_weights={0: 0.6278449223041909, 1: 2.4554941682013505}):
    data = preprocessor.transform(X_df)
    feature_names = preprocessor[-1].feature_names_    

    # Create a DMatrix for XGBoost
    weight = [class_weights[i] for i in y] if y is not None else None
    dmatrix = xgb.DMatrix(data=data, label=y, feature_names=feature_names, weight=weight)
    return dmatrix

dtrain = convert_to_dmatrix(preprocessor, X, y)


def xgb_train_parameters(dtrain, dval, params={}):
    # Specify the objective function as 'binary:logistic', "binary:hinge" or "rank:pairwise" for binary classification
    default_params  = {
        # 'n_estimators': 100,           # Number of boosting rounds
        'objective': 'binary:logistic',  # Objective function for binary classification
        "eval_metric": 'logloss',        # Logarithmic loss (use for monitoring), logloss
        'eta': 0.1,                      # Learning rate 0.3 or 0.1, depending on the XGBoost version
        'max_depth': 2,                  # Maximum depth of each tree
        'min_child_weight': 1,           # Minimum sum of instance weight (hessian) needed in a child
        'gamma': 0,                      # Minimum loss reduction required to make a further partition on a leaf node
        'subsample': 1,                  # Fraction of training data to be used for building trees
        'colsample_bytree': 0.4,         # Fraction of features to be randomly sampled for each tree
        'reg_alpha': 1,                  # L1 regularization term on weights
        'reg_lambda': 1,                 # L2 regularization term on weights
        'scale_pos_weight': 1.66,
        'nthread': 8, 'seed': SEED, 'verbosity': 1,
    }
    # Update params
    default_params.update(params)

    # Create a watchlist for validation
    watchlist = [(dtrain, 'train'), (dval, 'val')]
    return dict(params=default_params, dtrain=dtrain, evals=watchlist, num_boost_round=100, early_stopping_rounds=25, verbose_eval=5)



# Define a function to save training results to a text file without permanently modifying sys.stdout
def save_training_results(booster_params):
    # Create a file to save the results (replace 'training_results.txt' with your desired file name)
    with open('training_results.txt', 'w') as f_in:
        # Use contextlib.redirect_stdout to temporarily redirect sys.stdout
        with redirect_stdout(f_in):
            # Train the XGBoost model while printing the results to the file
            model = xgb.train(**booster_params)

    # Return the trained XGBoost model
    return model


# Get booster parameters
booster_params = xgb_train_parameters(dtrain, dtrain)
# Call the function to save the training results to 'training_results.txt'
model = save_training_results(booster_params)

# save to JSON or ubj format
model.save_model("model/xgb_model_v2.json")
model_xgb = xgb.Booster()
model_xgb.load_model("model/xgb_model_v2.json")


def parse_xgb_output(output):
    results = []

    for line in output.strip().split('\n'):
        it_line, train_line, val_line = line.strip().split('\t')

        it = int(it_line.strip('[]'))
        train = float(train_line.split(':')[1])
        val = float(val_line.split(':')[1])

        results.append((it, train, val))

    columns = ['num_iter', 'train_logloss', 'val_logloss']
    df_results = pd.DataFrame(results, columns=columns)
    return df_results


# Read the XGBoost training output from a text file (replace 'output.txt' with your file's name)
with open('training_results.txt', 'r') as file:
    xgb_output = file.read()

# Parse the XGBoost training output
parsed_results = parse_xgb_output(xgb_output)

# Print the parsed results as a DataFrame
print(parsed_results)
