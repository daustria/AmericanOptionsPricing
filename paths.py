import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV

# Simulate the path of a non-dividend paying stock with price S_0 at n_trials time intervals,
# with risk-free interest rate per-annum r, and sigma, the volatility per annum. The path is simulated
# from 0 to year T.

# There are probably more efficient ways to implement this. 

def simulate_path(params):    
    # Since we are assuming r and sigma are constant, we can compute S(T) quickly.
    # See Hull 21.17, edition 10    
    T = params.T
    n = params.n
    sigma = params.s 
    r = params.r

    S = np.zeros(n)
    delta_t = (T / (params.n - 1)) # Number of years in a given time interval

    S[0] = params.S
    
    # We can also generate pairs of paths, for the same amount of work. One being the mirror image of the other, with
    # directions happening symmetrically
    for i in range(1, n):
        S[i] = S[i-1] * np.exp((r - sigma**2 / 2) * delta_t + sigma * np.random.normal(0, 1) * np.sqrt(delta_t))

    return S

def simulate_paths(params):
    return np.array([simulate_path(params) for _ in range(params.N)])

def plot_path(path, params):
    T = params.T
    n_trials = params.n

    t = np.linspace(0, T, num=n_trials+1)

    fig, ax = plt.subplots()
    ax.plot(t, path)
    ax.set(xlabel='time', ylabel='price', title='Price of Stock')
    ax.grid()
    fig.savefig("path.png")     

