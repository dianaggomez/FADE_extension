import pandas
import numpy

#propensity score
def propensity_score(df, covariates, decision):
    """
    Using the counting method to calculate propensity score
    """
    grouped = df.groupby([covariates])
    df["pi(W)"] = grouped[decision].transform(lambda x: x.mean())
    
    print(df.head())
    return df
    
    
def calculate_mu0(df, covariates, decision):
    """
    Using the counting method to calculate mu0(W)
    """
    df_untreated = df[df[decision] == 0]
    grouped = df_untreated.groupby([covariates])
    mu_0 = grouped["Y"].mean().reset_index()
    mu_0.rename(columns={"Y": "mu0(W)"}, inplace=True)
    df = df.merge(mu_0, on=covariates, how="left")
    
    print(df.head())
    return df
    
    
def calculate_phi(df): 
    """
    We assume that the propensity_score() and calculate_mu0() were employed.
    Therefore the df should have the columns: pi(W) and mu0(W)
    """
    df["phi(Z)"] = ((1 - df["D"]) / (1 - df["pi(W)"])) * (df["Y"] - df["mu0(W)"]) + df["mu0(W)"]
    return df
