# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 23:58:51 2019

@author: Haley Schettler
"""


import numpy as np
from scipy.stats import binom

class VanillaOption:
    def __init__(self, strike, expiry, payoff):
        self.__strike = strike
        self.__expiry = expiry
        self.__payoff = payoff

    @property
    def strike(self):
        return self.__strike

    @strike.setter
    def strike(self, new_strike):
        self.__strike = new_strike

    @property
    def expiry(self):
        return self.__expiry

    @expiry.setter
    def expiry(self, new_expiry):
        self.__expiry = new_expiry

    def payoff(self, spot):
        return self.__payoff(self, spot)

def call_payoff(option, spot):
    return np.maximum(spot - option.strike, 0.0)

def put_payoff(option, spot):
    return np.maximum(option.strike - spot, 0.0)

def european_binomial(option, spot, rate, vol, div, steps):
    strike = option.strike
    expiry = option.expiry
    call_t = 0.0
    spot_t = 0.0
    h = expiry / steps
    num_nodes = steps + 1
    u = np.exp((rate - div) * h + vol * np.sqrt(h))
    d = np.exp((rate - div) * h - vol * np.sqrt(h))
    pstar = (np.exp(rate * h) - d) / ( u - d)
    
    for i in range(num_nodes):
        spot_t = spot * (u ** (steps - i)) * (d ** (i))
        call_t += option.payoff(spot_t) * binom.pmf(steps - i, steps, pstar)

    call_t *= np.exp(-rate * expiry)
    
    return call_t

def american_binomial(option, spot, rate, vol, div, steps):
    strike = option.strike
    expiry = option.expiry
    call_t = 0.0
    spot_t = 0.0
    h = expiry / steps
    num_nodes = steps + 1
    u = np.exp((rate - div) * h + vol * np.sqrt(h))
    d = np.exp((rate - div) * h - vol * np.sqrt(h))
    pstar = (np.exp(rate * h) - d) / ( u - d)
    disc = np.exp(-rate * h) 
    spot_t = np.zeros(num_nodes)
    prc_t = np.zeros(num_nodes)
    
    for i in range(num_nodes):
        spot_t[i] = spot * (u ** (steps - i)) * (d ** (i))
        prc_t[i] = option.payoff(spot_t[i])


    for i in range((steps - 1), -1, -1):
        for j in range(i+1):
            prc_t[j] = disc * (pstar * prc_t[j] + (1 - pstar) * prc_t[j+1])
            spot_t[j] = spot_t[j] / u
            prc_t[j] = np.maximum(prc_t[j], option.payoff(spot_t[j]))
                    
    return prc_t[0]




strike = 40
expiry = 0.5
spot = 40
rate = 0.08
vol = 0.30
div = 0.0
steps = 3




the_call = VanillaOption(strike, expiry, call_payoff)
the_put = VanillaOption(strike, expiry, put_payoff)

call_price = european_binomial(the_call, spot, rate, vol, div, steps)
print(f"The European Call Option Price is: {call_price : 0.2f}")

put_price = european_binomial(the_put, spot, rate, vol, div, steps)
print(f"The European Put Option Price is: {put_price : 0.2f}")

put_price_american = american_binomial(the_put, spot, rate, vol, div, steps)
print(f"The American Put Option is: {put_price_american : 0.2f}")

american_call_price = american_binomial(the_call, spot, rate, vol, div, steps)
print(f"The American Call Option is: {american_call_price : 0.2f}")