
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def VAE():
    """ Implementation of CLAIRE VAE """
    pass

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

if __name__ == "__main__":
    n = 1000
    data = simulation(n)
    
    print(data.head())
    print(data.describe())
    
    for i in range(1,5):
        plt.figure()
        sns.kdeplot(data=data, x="X{}".format(i), hue = 'A', fill = True)
        plt.savefig('./simulation_results_X{}.png'.format(i))