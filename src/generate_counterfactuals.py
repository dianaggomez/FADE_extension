
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.claire_cf_data_augmentation import CounterfactualDataGenerator


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def simulation(n=50000):
    """ 
    Generate counterfactuals via simulation
    This data generation process if fully synthetic and 
    the governing equations are found in the FADE paper 
    under section 6.1 Data Generation process
    """
    
    # n = 50,000  this is the number of samples in their experiments pg 23
    
    # A = 1 represents the minority group
    # I4 denotes the 4 × 4 identity matrix
    # D = 1 is the treatment
    # Y^0 is the potential outcome under no treatment
    # Y^1 is the potential outcome under treatment
    # Y is the generated observed outcome
    
    I4 = np.eye(4)
    
    #P(A = 1) = 0.3
    # Generate A (this indicates the minority group)
    A = np.random.binomial(1, 0.3, n)
    
    # Generate  X |A ~ N ( A x (1,−0.8,4,2).T,I4)
    X = np.array([np.random.multivariate_normal(mean = A[i] * np.array([1,-0.8,4,2]), cov = I4) for i in range(n)])
    
    print("A shape: ", A.shape)
    print("X shape: ", X.shape)
    A = A.reshape(-1, 1)
    print("A shape: ", A.shape)
    # The propensity score π(A,X) = P(D = 1 | A,X)
    # Generate D
    D1 = np.minimum(0.975,
                    sigmoid(np.dot(np.hstack((A,X)), np.array([0.2,-1,1,-1,1]).reshape(-1, 1))))
    D = np.random.binomial(1, D1)
    
    # Generate Y^0 
    Y0_ = sigmoid(np.dot(np.hstack((A,X)), np.array([-5,2,-3,4,-5]).reshape(-1, 1)))
    Y0 = np.random.binomial(1, Y0_)
    
    # Generate Y^1 
    Y1_ = sigmoid(np.dot(np.hstack((A,X)), np.array([1,-2, 3,-4,5]).reshape(-1, 1)))
    Y1 = np.random.binomial(1, Y0_)
    
    # Finally, generate Y
    Y = (1-D) * Y0 + D * Y1
    
    print("Y shape: ", Y.shape)
    print("D shape: ", D.shape)
    Y = Y.reshape(-1)
    A = A.reshape(-1)
    D = D.reshape(-1)
    
    print("A shape: ", A.shape)
    print("Y shape: ", Y.shape)
    generated_data = pd.DataFrame({
        'A': A, #race
        'X1': X[:, 0],
        'X2': X[:, 1],
        'X3': X[:, 2],
        'X4': X[:, 3],
        'D' : D,
        'Y' : Y 
    }
    )
    return generated_data


def claire_vae(vae_model, dataset, sensitive_groups=[0, 1], num_samples=20,
               batch_size=32, device=DEVICE, keep_original_Y=False) -> pd.DataFrame:
    """
    Generate counterfactuals using a VAE trained with the CLAIRE-M loss.
    
    Args:
        vae_model: The trained VAE model
        dataset: The dataset to generate counterfactuals for
            Will be used to generate batches of X, A, Y, D using a DataLoader
            Contains attributes X_names, A_name, Y_name, D_name
        sensitive_groups: Unique values of the sensitive attribute,
            defaults to [0, 1] for a binary sensitive attribute 
        batch_size: The batch size for inference
        device: The device to use
        keep_original_Y: Whether to keep the original outcome in the counterfactual dataset
        
    Returns:
        A DataFrame containing the counterfactuals with X, A, Y, D columns from the dataset
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    vae_model.to(device)
    cf_gen = CounterfactualDataGenerator(vae_model, sensitive_groups=sensitive_groups, K=num_samples)
    
    X_cfs = {a: [] for a in sensitive_groups}
    Y_cfs = {a: [] for a in sensitive_groups}
    D_orig = {a: [] for a in sensitive_groups}
    
    for X_batch, A_batch, Y_batch, D_batch in dataloader:
        X_cf_batch, Y_cf_batch = cf_gen.generate_counterfactuals(X_batch, Y_batch)
        
        for a in sensitive_groups:
            # Only add counterfactuals for the opposite sensitive group
            not_a_mask = (A_batch.cpu().detach().numpy() != a).flatten()   
            X_cfs[a].append(X_cf_batch[a][not_a_mask])
            if keep_original_Y:
                Y_cfs[a].append(Y_batch[not_a_mask])  # keep the original outcome, not reconstructed
            else:
                Y_cfs[a].append(Y_cf_batch[a][not_a_mask])  # keep the reconstructed outcome
                
            D_orig[a].append(D_batch[not_a_mask])  # keep the decision

    # Gather the counterfactual data
    cf_df = []
    for a in sensitive_groups:
        adf = pd.DataFrame(np.vstack(X_cfs[a]), columns=dataset.X_names)
        adf[dataset.A_name] = a
        adf[dataset.Y_name] = np.vstack(Y_cfs[a])
        adf[dataset.D_name] = np.vstack(D_orig[a])
        cf_df.append(adf)
    
    return pd.concat(cf_df, ignore_index=True)


if __name__ == "__main__":
    n = 1000
    data = simulation(n)
    
    print(data.head())
    print(data.describe())
    
    for i in range(1,5):
        plt.figure()
        sns.kdeplot(data=data, x="X{}".format(i), hue = 'A', fill = True)
        plt.savefig('./simulation_results_X{}.png'.format(i))