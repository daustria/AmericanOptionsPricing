from paths import *
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
import numpy as np
import warnings

class PriceEstimator(ClassifierMixin, BaseEstimator):
    def __init__(self, t, B, params):
        self.t = t # Time period
        self.B = B # Initial guess for the boundary value
        self.params = params

    def fit(self, X, y):
        # We don't need to do much here, since this is not really an estimator in the traditional sense defined by sklearn
        # y represents the optimal exercise time of each path assuming we don't exercise it at time period t in [0, 1, 2, ..., n-1]
        self.optimal_exercise_after = y
        return self

    def predict(self, X):
        # Predict the value of the option among all paths with the given boundary price, at the given
        # time period.
        value = 0
        N = len(X)
        delta_t = self.params.T / (self.params.n - 1)
        t = self.t
        r = self.params.r
        K = self.params.K

        for j in range(N):
            if X[j][t] < self.B:
                # We exercise the option
                value += max(K - X[j][t], 0) / N
            else:
                # We exercise at a later time period, given by y
                t_exec = self.optimal_exercise_after[j]

                if t_exec != -1:
                    # cash flow is discounted back to t.
                    value += max(K - X[j][t_exec], 0) * np.exp(-r * (t_exec - t) * delta_t) / N

        return value
        print(value)
                    
    def score(self, X, y, sample_weight=None):
        
        # Our score is the average value of the option .
        self.fit(X, y)
        return self.predict(X) 
    
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
    for t in range(n-2, -1, -1):
        
        # Estimate the boundary value
        price_estimator = PriceEstimator(t, K, params)
        search = RandomizedSearchCV(estimator=price_estimator, param_distributions={'B': uniform(loc=0, scale=K)}, n_iter=100)
        sh = search.fit(paths, optimal_exercise)
        boundary[t] = sh.best_params_['B']

        print("Time period: %d Value: %.2f Boundary:%.2f" % (t, sh.best_score_, sh.best_params_['B']))
        # Update the optimal exercise time, if we exercise at time period t.
        for j in range(N):
            optimal_exercise[j] = t if paths[j][t] < boundary[t] else optimal_exercise[j]

    return boundary

def compute_price(params, boundary):
    # We generate new paths to estimate the price of the option.
    # We don't use the same paths that were used to find the boundary values.
    paths = simulate_paths(params)
    n = params.n
    N = len(paths)

    optimal_exercise = [n-1 if paths[j][-1] < boundary[-1] else -1]

    for t in range(n-2, -1, -1):
        optimal_exercise = [t if path[t] < boundary[t] else optimal_exercise[j] for j in range(N)]


    estimator = PriceEstimator(0, boundary[0], params)
    estimator.fit(paths, optimal_exercise)
    print(estimator.predict(paths))

def price_parameterize_boundary(params):
    boundary = find_boundary(params)
    price = compute_price(params, boundary)
    return price

if __name__ == "__main__":
    pass
