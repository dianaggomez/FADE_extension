from sklearn.ensemble import RandomForest
from sklearn.metrics import accuracy_score

# train ensemble models: (we choose 2) 
def train(model, data, labels, counterfactuals, cf_labels):
    # Observed data
    model_obs = model()
    model_obs.fit(data, labels)
    
    # Counterfactual data
    model_cf = model()
    model_cf.fit(data, labels)
    
    return model_obs, model_cf



    

