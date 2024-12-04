from collections.abc import Iterable
from itertools import product
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.special import expit

from sklearn import preprocessing
from sklearn import svm
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.validation import check_is_fitted, NotFittedError

import seaborn as sns
import warnings

####################
## Basis learners ##
####################

def predict_basis(data, learners, covariates, bounds, continuous=True):
    """Generate predictions from learners"""
    n = data.shape[0]
    v = len(learners)
    Z = np.zeros((n, v))   # matrix of predictions

    for i, learner in enumerate(learners):
        if continuous and hasattr(learner, 'predict_proba'):
            Z[:, i] = learner.predict_proba(data[covariates])[:, 1]
        else:
            Z[:, i] = learner.predict(data[covariates])

    if bounds:
        Z = np.clip(Z, *bounds)

    return Z


#######################
## Nuisance learners ##
#######################

def fit_nuis(train, covariates, decision, outcome, learner_pi=None, learner_mu=None, learner_nu=None):
    """Fit nuisance learners."""
#     train = train.reset_index()
    if learner_pi is not None:
        learner_pi.fit(train[covariates], train[decision])
    if learner_mu is not None:
        learner_mu.fit(train.loc[train[decision].eq(0), covariates],
                       train.loc[train[decision].eq(0), outcome])
    if learner_nu is not None:
        learner_nu.fit(train.loc[train[decision].eq(0), covariates],
                       train.loc[train[decision].eq(0), outcome]**2)


def predict_nuis(data, covariates, decision, outcome,
                 learner_pi=None, learner_mu=None, learner_nu=None,
                 trunc_pi=0.975, clip_mu=None, clip_nu=None):
    """Generate predictions from nuisance learners"""
    out_dict = {}
    if hasattr(learner_pi, 'predict_proba'):
        pihat = pd.Series(learner_pi.predict_proba(data[covariates])[:, 1],
                          name='pihat').clip(upper=trunc_pi)
    else:
        pihat = pd.Series(learner_pi.predict(data[covariates]),
                          name='pihat').clip(upper=trunc_pi)
    if hasattr(learner_mu, 'predict_proba'):
        muhat0 = pd.Series(learner_mu.predict_proba(data[covariates])[:, 1],
                           name='muhat0')
        if clip_mu:
            muhat0 = muhat0.clip(*clip_mu)
    else:
        muhat0 = pd.Series(learner_mu.predict(data[covariates]), name='muhat0')
        if clip_mu:
            muhat0 = muhat0.clip(*clip_mu)

    out_dict['pihat'] = pihat
    out_dict['muhat0'] = muhat0

    phihat = pd.Series(
        (1 - data[decision]) / (1 - pihat) * (data[outcome] - muhat0) + muhat0,
        name='phihat')
    out_dict['phihat'] = phihat

    if learner_nu:
        if hasattr(learner_nu, 'predict_proba'):
            nuhat0 = pd.Series(learner_nu.predict_proba(data[covariates])[:, 1],
                               name='nuhat0')
        if clip_nu:
            nuhat0 = nuhat0.clip(*clip_nu)
        else:
            nuhat0 = pd.Series(learner_nu.predict(
                data[covariates]), name='nuhat0')
            if clip_nu:
                nuhat0 = nuhat0.clip(*clip_nu)
        out_dict['nuhat0'] = nuhat0
        phibarhat = pd.Series(
            (1 - data[decision]) / (1 - pihat) *
            (data[outcome]**2 - nuhat0) + nuhat0,
            name='phibarhat')

    out = pd.DataFrame(out_dict)
#     out = pd.concat([pihat, muhat0, phihat], axis=1)

    return out

##########################
## Fairness constraints ##
##########################

def compute_ghat_independence(protected_vec):
    """Compute the influence function vector for Independence."""
    group0 = (1 - protected_vec)
    group1 = protected_vec
    ghat = group0/np.mean(group0) - group1/np.mean(group1)
    return ghat


def compute_ghat_cFPR(protected_vec, outcome_vec):
    """Compute the influence function vector for the cFPR difference."""
    group0 = (1 - outcome_vec)*(1 - protected_vec)
    group1 = (1 - outcome_vec)*protected_vec
    ghat = group0/np.mean(group0) - group1/np.mean(group1)
    return ghat


def compute_ghat_cFNR(protected_vec, outcome_vec):
    """Compute the influence function vector for the cFNR difference."""
    group0 = outcome_vec*(1 - protected_vec)
    group1 = outcome_vec*protected_vec
    ghat = group0/np.mean(group0) - group1/np.mean(group1)
    return ghat



##################################
## Estimate coefficient vectors ##
##################################

def compute_betahats_inefficient(predictions, ghats, lambdas, outcome_vec):
    """betahat = (Q0 + \sum_{j=1}^r Q1_j)^{-1}Q2, where Q0 = Pn(bb^T),
    Q1_j = \lambda Pn(b * ghat_j) Pn(b * ghat_j)^T, Q2 = Pn(b * outcome_vec)"""
    # TODO: make this efficient using Sherman-Morrison
    lambdas = listify(lambdas)
    n = predictions.shape[0]
    Q0 = np.matmul(np.transpose(predictions), predictions)/n
    bghats = [np.mean(np.matmul(np.diag(gg), predictions), axis=0)
              for gg in ghats]
    Q1s = [np.outer(bb, bb) for bb in bghats]
    Q2 = np.mean(np.matmul(np.diag(outcome_vec), predictions), axis=0)

    if len(Q1s) == 1:
        Q = [Q0 + lambd*Q1s[0] for lambd in lambdas]
    else:
        Q = [Q0 + np.tensordot(lambd, Q1s, axes=1) for lambd in lambdas]
    betahats = [np.linalg.solve(QQ, Q2) for QQ in Q]

    return betahats


def _compute_betahat(Q0_inv, bghats, Q2, lambd):
    """Compute betahat efficiently for a single lambda vector, where Q0 = Pn(bb^T),
    bghats = Pn(b * ghat_j) for each j, and Q2 = Pn(b * outcome_vec).
    """
    Q = Q0_inv
    lambd = listify(lambd)
    for i in range(len(lambd)):
        Q = sherman_morrison(Q, bghats[i], bghats[i], lambd[i])
    Q = np.matmul(Q, Q2)
    return Q


def compute_betahats(Q0_inv, predictions, ghats, lambdas, outcome_vec):
    """Use the pre-inverted inverse of Q0 = Pn(bb^T), and use the
    Sherman-Morrison update for the rest, to avoid repeated matrix inversions."""
    bghats = [np.mean(np.matmul(np.diag(gg), predictions), axis=0)
              for gg in ghats]
    Q2 = np.mean(np.matmul(np.diag(outcome_vec), predictions), axis=0)
    betahats = [_compute_betahat(
        Q0_inv, bghats, Q2, ll) for ll in lambdas]
    return betahats


###########################################
## Estimate risk and fairness properties ##
###########################################

def independence_violation(protected_vec, pred_vec, absval=True):
    """Difference in expected value of the predictor for the two groups"""
    rate0 = pred_vec[protected_vec == 0].mean()
    rate1 = pred_vec[protected_vec == 1].mean()
    if absval:
        diff = np.abs(rate0 - rate1)
    else:
        diff = rate0 - rate1
    return diff


def deltahatplus(protected_vec, outcome_vec, pred_vec, absval=True):
    """Difference in generalized cFPRs for the two groups"""
    ghat = compute_ghat_cFPR(protected_vec, outcome_vec)
    diff = np.dot(ghat, pred_vec)/len(pred_vec)
    if absval:
        diff = np.abs(diff)

    return diff


def deltahatminus(protected_vec, outcome_vec, pred_vec, absval=True):
    """Difference in generalized cFNRs for the two groups"""
    ghat = compute_ghat_cFNR(protected_vec, outcome_vec)
    diff = np.dot(ghat, pred_vec)/len(pred_vec)
    if absval:
        diff = np.abs(diff)

    return diff


def get_active(df):
    """Return column that gives active fairness penalties."""
    actives = pd.DataFrame({'ind': df.lambda_independence.apply(lambda x: "ind" if x > 0 else None),
                            'cFPR': df.lambda_cFPR.apply(lambda x: "cFPR" if x > 0 else None),
                            'cFNR': df.lambda_cFNR.apply(lambda x: "cFNR" if x > 0 else None)})
    out = actives.apply(lambda x: '_'.join(
        [el for el in x if el is not None]), axis=1)
    return out


def eval_model(data, protected, phihat, phibarhat, base_predictions, betahats,
               unfairness='all', clip=None, lambdas=None, lambda_cols=None,
               err='all', absval=True):

    out_dict = {}

    # Predictions
    predlist = [np.matmul(base_predictions, bb) for bb in betahats]
    if clip is not None:
        predlist = [np.clip(preds, *clip) for preds in predlist]

    # Prediction variance
    learner_variance = [np.var(preds) for preds in predlist]
    out_dict['learner_variance'] = learner_variance

    # Error metrics
    if err == 'all':
        err = ['mse', 'bernoulli_01', 'threshold_01', 'auc', 'logistic']
    if 'mse' in err:
        error = [np.mean(preds**2 - 2*preds*data[phihat] + data[phibarhat])
                 for preds in predlist]
        out_dict['error_mse'] = error
    if 'bernoulli_01' in err:
        error = [np.mean(preds*(1 - data[phihat]) + (1 - preds)*data[phihat])
                 for preds in predlist]
        out_dict['error_bernoulli_01'] = error
    if 'auc' in err:
        try:
            error = [roc_auc_score(data[phihat], preds) for preds in predlist]
        except ValueError:
            error = np.nan
        out_dict['error_auc'] = error
    if 'logistic' in err:  # Using the same logistic loss as in Amanda and Ahesh's paper
        error = [np.mean(np.log(1 + np.exp(-5*(2*data[phihat] - 1) *
                                           (2*preds - 1)))/np.log(1 + np.exp(5))) for preds in predlist]
        out_dict['error_logistic'] = error

    # Unfairness metrics
    if unfairness == 'all':
        unfairness = ['independence', 'cFPR', 'cFNR', 'independence_binary',
                      'cFPR_binary', 'cFNR_binary']
    if 'independence' in unfairness:
        rate_diff = [independence_violation(
            data[protected], preds, absval) for preds in predlist]
        out_dict['rate_diff'] = rate_diff
    if 'cFPR' in unfairness:
        cFPR_diff = [deltahatplus(data[protected], data[phihat], preds, absval)
                     for preds in predlist]
        out_dict['cFPR_diff'] = cFPR_diff
    if 'cFNR' in unfairness:
        cFNR_diff = [deltahatminus(
            data[protected], data[phihat], preds, absval) for preds in predlist]
        out_dict['cFNR_diff'] = cFNR_diff


    out = pd.DataFrame(out_dict)
    if lambdas is not None:
        if lambda_cols is not None:
            columns = ["lambda_" + cc for cc in lambda_cols]
        elif isinstance(lambdas[0], Iterable):
            columns = ["lambda{}".format(i)
                       for i in range(1, 1 + len(lambdas[0]))]
        else:
            columns = ['lambda1']
        lambd_df = pd.DataFrame(lambdas, columns=columns)
        out = pd.concat([out, lambd_df], axis=1)
        out = out.assign(active_penalties=get_active(out))

    return out




from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# TODO: Define the datasets for each

Dlearn_train = ...
Dlearn_test = ...
Dnuis_train = ...
Dnuis_test = ...
Dtarget_test = ...
# Predictors b(W)=(b1(W),b2(W),...,bk(W))b(W)=(b1​(W),b2​(W),...,bk​(W)) can include:
    # - Previously trained models (SS).
    # - Newly trained models on XX and AA.
    # - Arbitrary mathematical transformations or orthogonal basis functions like polynomials or splines.

learner1 = LogisticRegression()
learner2 = RandomForestClassifier()

#TODO: 
# Define covariates
covariates = [] #list of names of the covairiates so ['str', 'str']
target = "income" # In our case the outcome is income level, alternatively we can change this to 'Y'
# Train learners on subsets or different transformations of the data
learner1.fit(Dlearn_train[covariates], Dlearn_train[target])
learner2.fit(Dlearn_train[covariates], Dlearn_train[target])

# Use them as basis learners
learners = [learner1, learner2]

# Predict basis
# TODO: Is this data Dlearn???
Z = predict_basis(data=test_data, learners=learners, covariates=covariates, bounds=(0, 1))

# Train the Nuisance Learners: Use fit_nuis to train models for:

#     π(W): Propensity score learner (e.g., Logistic Regression).
#     μ0​(W): Outcome model for untreated individuals (e.g., Random Forest).
#     ν0(W): Squared outcome model for untreated individuals (optional; can also use ν0=μ0​ if outcomes are binary).

# Generate Predictions: Use predict_nuis to:

#     Predict π(W), μ0​(W), ν0​(W).
#     Compute the adjusted predictions (ϕ(Z) and ϕˉ(Z)) based on the trained models.


learner_pi = LogisticRegression()          # Propensity score learner
learner_mu = RandomForestRegressor()       # Outcome model
learner_nu = RandomForestRegressor()       # Squared outcome model (optional)

# Train nuisance learners
fit_nuis(
    train=Dnuis_train,
    covariates=covariates,         # Covariates (features)
    decision="D",                  # Treatment assignment (D)
    outcome="Y",                   # Outcome variable (Y)
    learner_pi=learner_pi,
    learner_mu=learner_mu,
    learner_nu=learner_nu
)

# Predict nuisance parameters on the test set
nuisance_predictions = predict_nuis(
    data=Dnuis_train,
    covariates=covariates,         # Covariates (features)
    decision="D",                  # Treatment assignment (D) #TODO: rename target or keep Y
    outcome="Y",                   # Outcome variable (Y)
    learner_pi=learner_pi,
    learner_mu=learner_mu,
    learner_nu=learner_nu,
    trunc_pi=0.975,                # Truncate propensity scores for numerical stability
    clip_mu=(0, 1),                # Clip predictions for mu (if applicable)
    clip_nu=(0, 1)                 # Clip predictions for nu (if applicable)
)

# View nuisance predictions
print(nuisance_predictions.head())

# 4. Combine nuisance predictions and basis matrix for fairness optimization
# Need to first get Q0, ghats. lambdas
# Compute Q0
n = Z.shape[0]  
Q0 = np.dot(Z.T, Z) / n
try: 
    Q0_inv = np.linalg.inv(Q0)
except:
    print("Not invertible")
    
# In case it is in not invertible 
epsilon = 1e-6
Q0_inv = np.linalg.inv(Q0 + epsilon * np.eye(Q0.shape[0]))


# Compute ghats - these are the fairness constraints
# Protected attribute (e.g., race or gender)
# TODO: Input the proper sensitive attribute, also what test data goes here?
protected_vec = Dtarget_test["A"]  # Sensitive attribute

# Outcome (e.g., binary label)
outcome_vec = Dtarget_test["Y"]

# Compute fairness influence functions, we don't need all of them but here they are
ghat_independence = compute_ghat_independence(protected_vec)
ghat_cFPR = compute_ghat_cFPR(protected_vec, outcome_vec)
ghat_cFNR = compute_ghat_cFNR(protected_vec, outcome_vec)

# Combine the influence functions into a list for multi-constraint optimization
# TODO: Choose that constriant that we want to use first
# for now lets just use cFPR
# ghats = [ghat_independence, ghat_cFPR, ghat_cFNR]
ghats = ghat_cFPR
#TODO: Check dim of lambda
lambdas = [
    [0.1],  # Small penalty for cFPR
    [0.5],  # Medium penalty for cFPR
    [1.0],  # Large penalty for cFPR
]

phihat = nuisance_predictions["phihat"]

betahats = compute_betahats(Q0_inv, Z, ghats, lambdas, phihat)

# 5. Evaluate model and fairness metrics
results = eval_model(
    data=Dtarget_test,
    protected="A",
    phihat="phihat",
    phibarhat="phibarhat",
    base_predictions=Z,
    betahats=betahats,
    unfairness=["cFPR"],  # Only evaluate cFPR constraint
    err=["mse", "auc"],   # Evaluate mean squared error and AUC
    lambdas=lambdas
)

# Print results
print(results)
