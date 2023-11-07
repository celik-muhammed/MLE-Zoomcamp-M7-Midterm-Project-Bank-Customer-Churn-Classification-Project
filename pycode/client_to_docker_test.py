
import requests

url = "http://localhost:9696/predict"  # Replace with your actual URL

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

response = requests.post(url, json=client_data).json()
print(f"Does the client approve the loan?: {response['get_credit']}")
print(f"The probability that this client will get a credit is: {response['get_credit_probability']:.3f}")
