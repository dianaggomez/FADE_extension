"""
References: 
    - https://www.mathworks.com/help/risk/explore-fairness-metrics-for-credit-scoring-model.html
    - https://aif360.readthedocs.io/en/latest/modules/sklearn.html#module-aif360.sklearn.metrics
"""
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import rbf_kernel


def statistical_parity_diff(y_pred, A):
    """
    Calculate the Statistical Parity Difference (SPD) based on observable data.
    
        SPD = P(y_pred = 1 | A = minority) - P(y_pred = 1 | A = majority)
        
    SPD measures the deviation from a scenario where the same proportion of each sensitive group 
    receives the favorable outcome. 
    
    If SPD = 0, the model satisfies statistical parity.
    
    Args:
        y_pred: Predicted labels
        A: Binary sensitive attribute
    
    Returns:
        Statistical Parity Difference
    """
    # Separate predictions by sensitive attribute
    mask_a0 = (A == 0)
    mask_a1 = (A == 1)
    
    # Calculate selection rates
    selection_rate_a0 = np.mean(y_pred[mask_a0])
    selection_rate_a1 = np.mean(y_pred[mask_a1])
    
    # Calculate statistical parity difference
    return selection_rate_a1 - selection_rate_a0


def equal_opportunity_diff(y_true, y_pred, A):
    """
    Calculate the Equal Opportunity Difference (EOD) based on observable data.
        
        EOD = TPR(A = minority) - TPR(A = majority), where TPR = TP / (TP + FN)
            = P(y_pred = 1 | y_true = 1, A = minority) - P(y_pred = 1 | y_true = 1, A = majority)
    
    EOD measures the deviation from a scenario where the same proportion of each sensitive group
    that received the favorable outcome is correctly classified. 
    
    If EOD = 0, the model satisfies equal opportunity.
        
    Args:
        y_true: True labels
        y_pred: Predicted labels
        A: Binary sensitive attribute
    
    Returns:
        Equal Opportunity Difference
    """
    # Separate data by sensitive attribute
    mask_a0 = (A == 0)
    mask_a1 = (A == 1)
    
    # Calculate true positive rates
    tn_a0, fp_a0, fn_a0, tp_a0 = confusion_matrix(y_true[mask_a0], y_pred[mask_a0]).ravel()
    tn_a1, fp_a1, fn_a1, tp_a1 = confusion_matrix(y_true[mask_a1], y_pred[mask_a1]).ravel()
    tpr_a0 = tp_a0 / (tp_a0 + fn_a0)
    tpr_a1 = tp_a1 / (tp_a1 + fn_a1)
    
    # Calculate equal opportunity difference
    return tpr_a1 - tpr_a0


def counterfactual_fairness_mmd(model, W, sensitive_groups, **kernel_params):
    """
    Compute the Counterfactual Fairness (CF) using the Maximum Mean Discrepancy (MMD).
    
    CF assesses how the model's predictions change when the sensitive attribute is modified while keeping 
    the other features constant. The MMD is used to measure the difference between the prediction distributions 
    for every pair of counterfactuals, and the average value is taken as the final result.
    
    If the MMD is close to zero, the model is considered to be counterfactually fair.
    
    Args:
        model: Trained machine learning model with a predict method
        W: Input features, where the last column corresponds to the sensitive attribute
        sensitive_groups: List of possible values for the sensitive attribute (e.g., [0, 1] for binary attributes)
        **kernel_params: Additional parameters for the kernel function (e.g., gamma for RBF kernel)
    
    Returns:
        Average MMD value
    """
    mmd_values = []
    
    # Consider all pairs of sensitive attribute values
    for i in range(len(sensitive_groups)):
        # Create counterfactual data by swapping sensitive attribute values
        W_cf_i = W.copy()
        W_cf_i[:, -1] = sensitive_groups[i]
        
        # Get predictions for counterfactual data
        preds_cf_i = model.predict(W_cf_i)
        
        for j in range(i + 1, len(sensitive_groups)):            
            # Create counterfactual data by swapping sensitive attribute values
            W_cf_j = W.copy()
            W_cf_j[:, -1] = sensitive_groups[j]
            
            # Get predictions for counterfactual data
            preds_cf_j = model.predict(W_cf_j)
            
            # Compute MMD between the two prediction distributions
            mmd = compute_mmd(preds_cf_i.reshape(-1, 1), preds_cf_j.reshape(-1, 1), **kernel_params)
            mmd_values.append(mmd)
    
    # Return the average MMD value
    return np.mean(mmd_values)


def compute_mmd(X, Y, kernel='rbf', gamma=1.0):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two distributions X and Y.
    
    Args:
        X: Samples from distribution X
        Y: Samples from distribution Y
        kernel: Kernel type ('rbf' for the Radial Basis Function kernel)
        gamma: Kernel coefficient for RBF kernel
    
    Returns:
        MMD value
    """
    if kernel == 'rbf':
        XX = rbf_kernel(X, X, gamma=gamma)
        YY = rbf_kernel(Y, Y, gamma=gamma)
        XY = rbf_kernel(X, Y, gamma=gamma)
        return np.mean(XX) + np.mean(YY) - 2 * np.mean(XY)
    else:
        raise ValueError("Unsupported kernel type")
