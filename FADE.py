import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from generate_counterfactuals import *

def load_data():
    column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num', 
    'marital_status', 'occupation', 'relationship', 'race', 
    'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 
    'native_country', 'income']
    
    train_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    train_data = pd.read_csv(train_url, names=column_names, sep=',', skipinitialspace=True)
    
    test_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
    test_data = pd.read_csv(test_url, names=column_names, sep=',', skipinitialspace=True)
    
    return train_data, test_data
    
# train ensemble models: (we choose 2) 
def train(model, data, labels, counterfactuals, cf_labels):
    # Observed data
    model_obs = model()
    model_obs.fit(data, labels)
    
    # Counterfactual data
    model_cf = model()
    model_cf.fit(data, labels)
    
    return model_obs, model_cf


def evaluate_counterfactual_fairness(orig_predictions, cf_predictions):
    """
    A model is counterfactually fair if its predictions remain unchanged 
    given that the senstitive attributes have been altered
    """
    same = (orig_predictions == cf_predictions).sum()
    cf_fairness_score = same/len(orig_predictions)
    
    return cf_fairness_score

if __name__ == "__main__":
    train_data, test_data = load_data()
    
    print(train_data.head())
    print(test_data.head())
    # Define the models you want to use
    model = RandomForestClassifier()
    
    # Generate counterfactuals
    counterfactuals = simulation()
    
    
    # train(model, train_data, labels, counterfactuals, cf_labels)
    