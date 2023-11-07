# MLE-Zoomcamp-Midterm-Project-Bank-Customer-Churn-Classification-Project
MLE-Zoomcamp-Midterm-Project-Bank-Customer-Churn-Classification-Project

> EVERY STEP IS AVAILABLE IN THE NOTEBOOK

> [Bank-Customer-Churn-Classification-Project-v2.ipynb](Bank-Customer-Churn-Classification-Project-v2.ipynb)

---

# **Bank Customer Churn Prediction Project**

### Task 1: Importing Libraries and Data Loading

- Create a Jupyter notebook for data analysis and machine learning.
- Import necessary Python libraries (e.g., pandas, numpy, scikit-learn).
- Load the dataset (e.g., "bank_customer_churn.csv") into a DataFrame.

### Task 2: Data Exploration

- Examine the first few rows of the dataset to get a sense of the data.
- Check for missing values in the dataset and handle them as needed.
- Explore summary statistics and data distributions for numeric features.
- Visualize categorical feature distributions (e.g., gender, geography).
- Explore correlations between features using correlation matrices.

**Note:** Identify trends or patterns in the data. Look for factors that may be influencing customer churn, such as age, balance, and credit score.

### Task 3: Data Cleaning and Preprocessing

- Handle any outliers or anomalies in the data.
- Encode categorical variables (e.g., one-hot encoding).
- Scale or normalize numeric features if necessary.
- Split the data into training and testing sets.

**Note:** Clean and preprocess the data to prepare it for machine learning modeling. Ensure data quality for accurate predictions.

### Task 4: Exploratory Data Analysis (EDA)

- Conduct more in-depth data analysis.
- Create data visualizations to understand feature relationships.
- Investigate feature importance using techniques like feature importance scores or permutation importance.

**Note:** Understand the factors that contribute to customer churn. Identify key features that impact customer retention or attrition.

### Task 5: Feature Selection

- Implement feature selection techniques (e.g., SelectKBest, feature importance) to identify the most relevant features for the model.

**Note:** Focus on the most influential features for building a predictive model. This can improve model accuracy and efficiency.

### Task 6: Model Training

- Select appropriate machine learning algorithms (e.g., Logistic Regression, Random Forest, XGBoost) for classification.
- Train multiple models with the training data.
- Tune hyperparameters for better model performance.

**Note:** Develop machine learning models to predict customer churn. The choice of algorithms and hyperparameters can impact prediction accuracy.

### Task 7: Model Evaluation

- Evaluate model performance using metrics like accuracy, precision, recall, and F1-score.
- Create confusion matrices and ROC curves for model assessment.
- Compare and select the best-performing model.

**Note:** Assess the effectiveness of the models in predicting customer churn. Make informed decisions based on model performance.

### Task 8: Interpretation and Reporting

- Interpret the model results and feature importance.
- Generate insights and recommendations for business decisions.
- Create a project report summarizing findings and predictions.

**Note:** Translate model results into actionable insights for reducing customer churn. Provide recommendations for strategies to improve customer retention.

### Task 9: Model Deployment

- If required, deploy the best-performing model for real-time predictions.

**Note:** Deploying the model can automate churn prediction and improve decision-making within the organization.


---

# **Dataset More Info**

> https://www.kaggle.com/datasets/shubh0799/churn-modelling

<!-- https://cio-wiki.org/wiki/Customer_Churn -->

**About Dataset**

In this dataset, we have details of a bank's customers with a binary target variable indicating whether the customer left the bank (closed their account) or continues as a customer.

**Features in the Dataset: Inventory of Customer Assets**

0. **Row Number:** Sequential row numbers from 1 to 10,000.
1. **Customer ID:** A unique identifier for each customer.
2. **Surname:** The customer's last name.
3. **Credit Score:** A numerical representation (ranging from 300 to 850) of a customer's creditworthiness.
4. **Geography:** The country from which the customer originates.
5. **Gender:** The gender of the customer (Male or Female).
6. **Age:** The current age of the customer in years at the time of their association with the bank.
7. **Tenure:** The number of years the customer has been with the bank.
8. **Balance:** The bank balance of the customer.
9. **Number of Products:** The count of bank products the customer is currently utilizing.
10. **Has Credit Card:** A binary indicator (0 or 1) representing whether the customer has been issued a credit card by the bank.
11. **Is Active Member:** A binary flag indicating whether the customer is an active member with the bank prior to their exit, recorded in the "Exited" variable.
12. **Estimated Salary:** An estimation of the customer's annual salary.
13. **Exited:** A binary flag (1 or 0) indicating whether the customer closed their account with the bank (1) or remains a retained customer (0).


---

# Dependency and environment management guide

```py
# cd created env folder

# Initialize Pipenv Environment, build pipfile and check for library versions
# pipenv --python=3.10

# build pipfile.lock, Install project's Dependencies from a Pipfile
# pipenv install

# Optionally, Install Development Dependencies
# pipenv install --dev

# Optionally, Generate or Update a Lock File
# pipenv lock

# Activate the Environment
# pipenv shell

```

---


# Containerization


```pd
# Buid Dockerfile then create Image
docker build -t churn-flask-app .

# Check Images
docker images

# Run Container
# -d  means container in detached mode (in the background)
# -it means container in interactively
# -rm means Automatically removes the container when it stops.
# -p 9696:9696: Maps port 9696 on the host to port 9696 in the container.
# docker run -d --rm -p 9696:9696 --name my-flask-container churn-flask-app

# List running containers and their ports
docker ps -a

# Stop container
docker stop my-flask-container
```


---

# Midterm Project Requirements (Evaluation Criteria)

> Questions: https://github.com/DataTalksClub/machine-learning-zoomcamp/edit/master/projects/README.md

The idea is that you now apply everything we learned so far yourself.

There are three projects in this course:

* **Midterm project**
* Capstone 1
* Capstone 2

This is what you need to do for each project

* Think of a problem that's interesting for you and find a dataset for that
* Describe this problem and explain how a model could be used
* Prepare the data and doing EDA, analyze important features
* Train multiple models, tune their performance and select the best model
* Export the notebook into a script
* Put your model into a web service and deploy it locally with Docker
* Bonus points for deploying the service to the cloud


## Deliverables

For a project, you repository/folder should contain the following:

* `README.md` with
  * Description of the problem
  * Instructions on how to run the project
* Data
  * You should either commit the dataset you used or have clear instructions how to download the dataset
* Notebook (suggested name - `notebook.ipynb`) with
  * Data preparation and data cleaning
  * EDA, feature importance analysis
  * Model selection process and parameter tuning
* Script `train.py` (suggested name)
  * Training the final model
  * Saving it to a file (e.g. pickle) or saving it with specialized software (BentoML)
* Script `predict.py` (suggested name)
  * Loading the model
  * Serving it via a web service (with Flask or specialized software - BentoML, KServe, etc)
* Files with dependencies
  * `Pipenv` and `Pipenv.lock` if you use Pipenv
  * or equivalents: conda environment file, requirements.txt or pyproject.toml
* `Dockerfile` for running the service
* Deployment
  * URL to the service you deployed or
  * Video or image of how you interact with the deployed service


## Peer reviewing

> [!IMPORTANT]  
> To evaluate the projects, we'll use peer reviewing. This is a great opportunity for you to learn from each other.
> * To get points for your project, you need to evaluate 3 projects of your peers
> * You get 3 extra points for each evaluation

Tip: you can use https://nbviewer.org/ to render notebooks if GitHub doesn't work


## Evaluation Criteria

The project will be evaluated using these criteria:

* Problem description
* EDA
* Model training
* Exporting notebook to script
* Model deployment
* Reproducibility
* Dependency and environment management
* Containerization
* Cloud deployment

[Criteria](https://docs.google.com/spreadsheets/d/e/2PACX-1vQCwqAtkjl07MTW-SxWUK9GUvMQ3Pv_fF8UadcuIYLgHa0PlNu9BRWtfLgivI8xSCncQs82HDwGXSm3/pubhtml)


## Cheating and plagiarism

Plagiarism in any form is not allowed. Examples of plagiarism:

* Taking somebody's else notebooks and projects (in full or partly) and using it for the capstone project
* Re-using your own projects (in full or partly) from other courses and bootcamps
* Re-using your midterm project from ML Zoomcamp in capstone
* Re-using your ML Zoomcamp from previous iterations of the course

Violating any of this will result in 0 points for this project.


**Submit the results**

<!-- - Submit your results here: https://forms.gle/Qa2SuzG7QGZNCaoV9
- If your answer doesn't match options exactly, select the closest one.
- You can submit your solution multiple times. In this case, only the last submission will be used -->

**Deadline**

<!-- The deadline for submitting is October 23 (Monday), 23:00 CET. After that the form will be closed. -->

