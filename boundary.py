from paths import *
from sklearn.model_selection import ParameterSampler
from scipy.stats import uniform
import numpy as np
import warnings

def estimate_price(paths, t, B, optimal_exercise, params):
    price = 0
    N = len(paths)
    delta_t = params.T
    r = params.r
    K = params.K

    value = 0
    for j in range(N):        
        if paths[j][t] < B:
            # We exercise the option at time t
            value += max(K - paths[j][t], 0) / N
        else:
            # We exercise at a later time period, given by y
            t_exec = optimal_exercise[j]

            if t_exec != -1:
                # cash flow is discounted back to t.
                value += max(K - paths[j][t_exec], 0) * np.exp(-r * (t_exec - t) * delta_t) / N

    return value
    
def find_boundary(params):
    paths = simulate_paths(params)
    n = params.n
    K = params.K
    N = len(paths)

    boundary = [ 0 for _ in range(n)]
    # We know the exercise boundary, at maturity time T, must be the strike price of the option.    
    boundary[n-1] = K

    # A table such that optimal_exercise[j][t] contains the optimal exercise time in for path j,
    # assuming it has not been exercised before time period t in [0,1,...,n-1]. Contains -1 if the option is never exercised.
    optimal_exercise = np.zeros(N, dtype=int)
 
    for j in range(N):
        optimal_exercise[j] = n-1 if paths[j][-1] < boundary[-1] else -1
    
    print("Time period: %d Boundary:%.2f" % (n-1, boundary[n-1]))

    param_grid = {'B': uniform(0, K)}

    for t in range(n-2, -1, -1):        
        prices = paths[:,t]

        param_list = list(ParameterSampler({'B': uniform(loc=min(prices), scale=max(prices)-min(prices))}, n_iter=100)) # Can put random_state=rng for reproducible results
        max_price = -1
        B = K

        for entry in param_list:            

            price = estimate_price(paths, t, entry['B'], optimal_exercise, params)
            max_price = max(price, max_price)

            if max_price == price:
                B = entry['B']
        
        boundary[t] = B        
        print("Time period: %d Value: %.2f Boundary:%.2f" % (t, max_price, B))

        # Update the optimal exercise time, if we exercise at time period t.
        for j in range(N):
            optimal_exercise[j] = t if paths[j][t] < B else optimal_exercise[j]

    return boundary

def compute_price(params, boundary):
    paths = simulate_paths(params)
    n = params.n
    K = params.K
    N = len(paths)
    
    # A table such that optimal_exercise[j][t] contains the optimal exercise time in for path j,
    # assuming it has not been exercised before time period t in [0,1,...,n-1]. Contains -1 if the option is never exercised.
    optimal_exercise = [n-1 if paths[j][-1] < boundary[-1] else -1 for j in range(N)]      
    print("Time period: %d Boundary:%.2f" % (n-1, boundary[n-1]))

    for t in range(n-2, -1, -1):        
                        
        price = estimate_price(paths, t, boundary[t], optimal_exercise, params)                

        print("Time period: %d Value: %.2f Boundary:%.2f" % (t, price, boundary[t]))

        # Update the optimal exercise time, if we exercise at time period t, which is determined by the boundary.
        for j in range(N):
            optimal_exercise[j] = t if paths[j][t] < boundary[t] else optimal_exercise[j]

        if t == 0:
            return price

    return -1

def price_parameterize_boundary(params):
    boundary = find_boundary(params)
    price = compute_price(params, boundary)
    return price

if __name__ == "__main__":
    pass
