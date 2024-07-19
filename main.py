import numpy as np
import argparse
import math
from paths import simulate_path, simulate_paths
from boundary import price_parameterize_boundary

def get_parser():
    """
    Generates a parser parameter
    """
    parser = argparse.ArgumentParser(
        description="Simulate Pricing of American or Bermudan Put Options with various Monte Carlo based methods.")

    parser.add_argument("--S", type=float, default=100,
                            help="The inital price of the asset.")

    parser.add_argument("--K", type=float, default=110,
                            help="The strike price of the American put option.")                            

    parser.add_argument("--T", type=float, default=1,
                            help="The maturity time of the put option, in years.")

    parser.add_argument("--n", type=int, default=50,
                                help="The number of discrete time periods at which exercise is available from [0,T].")

    parser.add_argument("--N", type=int, default=1000,
                                help="The number of stock price paths to simulate, assuming a geometric brownian motion.")

    parser.add_argument("--r", type=float, default=0.06,
                                help="The risk-free interest rate per annum, with continuous compounding. Note that we assume the stock does not pay dividends.")

    parser.add_argument("--s", type=float, default=0.12,
                                help="The volatility per annum")

    parser.add_argument("--p", type=float, default=100,
                                help="The initial price of the option contract per share (assuming we are buying the option for 100 stocks)")

    return parser

if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()
    print(params)
    price = price_parameterize_boundary(params)
    print(price)
