import numpy as np
from scipy.stats import binom
from collections import namedtuple
import scipy.stats as stats
from scipy.stats import norm

def european_binomial(option, rate, vol, div, steps):
    strike = option.strike
    expiry = option.expiry
    spot = option.spot
    call_t = 0.0
    spot_t = 0.0
    option_t = 0.0
    h = expiry/steps
    u = np.exp((rate - div) * h + vol * np.sqrt(h))
    d = np.exp((rate - div) * h - vol * np.sqrt(h))
    num_nodes = steps + 1
    pstar = (np.exp(rate * h) - d) / ( u - d)
    
    for i in range(num_nodes):
        spot_t =  spot * (u ** (steps - i)) * (d ** (i))
        option_t += option.payoff(spot_t) * binom.pmf(steps - i, steps, pstar)
    
    option_t *= np.exp(-rate * option.expiry)
    
    return option_t


def american_binomial(option, rate, vol, div, steps):
    strike = option.strike
    expiry = option.expiry
    spot = option.spot
    call_t = 0.0
    spot_t = 0.0
    h = expiry/steps
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
            prc_t[j] = disc * (pstar * prc_t[j] + (1-pstar) * prc_t[j+1])
            spot_t[j] = spot_t[j] / u
            prc_t[j] = np.maximum(prc_t[j], option.payoff(spot_t[j]))
            
    return prc_t[0]

def naive_monte_carlo_pricer(option, spot, rate, vol, div, nreps):
    PricerResult = namedtuple('PricerResult', ['price', 'stderr'])
    expiry = option.expiry    
    strike = option.strike
    h = expiry 
    disc = np.exp(-rate * h)
    spot_t = np.empty(nreps)
    z = np.random.normal(size = nreps)

    for j in range(1, nreps):
        
        spot_t[j] = spot *  np.exp((rate - div - 0.5 * vol * vol) * h + vol * np.sqrt(h) * z[j])

    payoff_t = option.payoff(spot_t)

    prc = payoff_t.mean() * disc
    se = payoff_t.std(ddof=1) / np.sqrt(nreps)

    return PricerResult(prc, se)

def antithetic_monte_carlo_pricer(option, spot, rate, vol, div, nreps):
    PricerResult = namedtuple('PricerResult', ['price', 'stderr'])
    expiry = option.expiry
    strike = option.strike
    h = expiry 
    disc = np.exp(-rate * h)
    spot_t = np.empty(nreps*2)
    z = np.random.normal(size = nreps)
    z_neg=-z
    X = np.concatenate((z, z_neg))

    for j in range(1, nreps*2):
        
        spot_t[j] = spot *  np.exp((rate - div - 0.5 * vol * vol) * h + vol * np.sqrt(h) * X[j])

    payoff_t = option.payoff(spot_t)

    prc = payoff_t.mean() * disc
    se = payoff_t.std(ddof=1) / np.sqrt(nreps*2)

    return PricerResult(prc, se)
            
    
def stratified_random(nreps):
    u = np.random.uniform(size = nreps)
    u_hat = np.zeros(nreps)
    
    for i in range(nreps):
        u_hat[i] = (i+u[i])/nreps
    
    return stats.norm.ppf(u_hat)

def stratified_monte_carlo_pricer(option, spot, rate, vol, div, nreps):
    PricerResult = namedtuple('PricerResult', ['price', 'stderr'])
    expiry = option.expiry    
    strike = option.strike
    h = expiry 
    disc = np.exp(-rate * h)
    spot_t = np.empty(nreps)
    z = stratified_random(nreps)

    for j in range(1, nreps):
        
        spot_t[j] = spot *  np.exp((rate - div - 0.5 * vol * vol) * h + vol * np.sqrt(h) * z[j])

    payoff_t = option.payoff(spot_t)

    prc = payoff_t.mean() * disc
    se = payoff_t.std(ddof=1) / np.sqrt(nreps)

    return PricerResult(prc, se)


def bsmCallDelta(spot, strike, rate, vol, div, t):
    d1 = (np.log(spot/strike) + (rate - div + 0.5 * vol * vol) * t) / (vol * np.sqrt(t))

    return np.exp(-div * t) * norm.cdf(d1)

def bsmCallPrice(spot, strike, rate, vol, div, t):
    d1 = (np.log(spot/strike) + (rate - div + 0.5 * vol * vol) * t) / (vol * np.sqrt(t))
    d2 = (np.log(spot/strike) + (rate - div - 0.5 * vol * vol) * t) / (vol * np.sqrt(t))
    callPrc = spot * np.exp(-div * t) * norm.cdf(d1) - strike * np.exp(-rate * t) * norm.cdf(d2)
    
    return callPrc

def bsmPutPrice(spot, strike, rate, vol, div, t):
    d1 = (np.log(spot/strike) + (rate - div + 0.5 * vol * vol) * t) / (vol * np.sqrt(t))
    d2 = (np.log(spot/strike) + (rate - div - 0.5 * vol * vol) * t) / (vol * np.sqrt(t))
    putPrc =  strike * np.exp(-rate * t) * norm.cdf(-d2) - spot * np.exp(-div * t) * norm.cdf(-d1)
    
    return putPrc

PricerResult = namedtuple('PricerResult', ['price', 'stderr'])

def BlackScholesDelta(spot, t, strike, expiry, vol, rate, div):
    tau = expiry - t
    d1 = (np.log(spot/strike) + (rate - div + 0.5 * vol * vol) * tau) / (vol * np.sqrt(tau))
    delta = np.exp(-div * tau) * norm.cdf(d1) 
    return delta

def naive_pricer(option, spot, rate, vol, div, steps, reps):
    pass

import numpy as np
from scipy.stats import norm
from collections import namedtuple


PricerResult = namedtuple('PricerResult', ['price', 'stderr'])

def BlackScholesDelta(spot, t, strike, expiry, vol, rate, div):
    tau = expiry - t
    d1 = (np.log(spot/strike) + (rate - div + 0.5 * vol * vol) * tau) / (vol * np.sqrt(tau))
    delta = np.exp(-div * tau) * norm.cdf(d1) 
    return delta

def naive_pricer(option, spot, rate, vol, div, steps, reps):
    pass

def control_variate_pricer(option, spot, rate, vol, div, steps, reps, beta):
    PricerResult = namedtuple('PricerResult', ['price', 'stderr'])
    expiry = option.expiry
    strike = option.strike
    h = expiry / steps
    nudt = (rate - div - 0.5 * vol * vol) * h
    sigsdt = vol * np.sqrt(h)
    erddt = np.exp((rate - div) * h)    
    cash_flow_t = np.zeros(reps)
    price = 0.0

    for j in range(reps):
        spot_t = spot
        spot_t2 = spot
        convar = 0.0
        convar2 = 0.0
        z = np.random.normal(size=int(steps))

        for i in range(int(steps)):
            t = i * h
            delta = BlackScholesDelta(spot_t, t, strike, expiry, vol, rate, div)
            delta2 = BlackScholesDelta(spot_t2, t, strike, expiry, vol, rate, div)
            spot_tn = spot_t * np.exp(nudt + sigsdt * z[i])
            spot_tn2 = spot_t2 * np.exp(nudt + sigsdt * -z[i])
            convar = convar + delta * (spot_tn - spot_t * erddt)
            convar2 = convar2 + delta2 * (spot_tn2 - spot_t2 * erddt)
            spot_t = spot_tn
            spot_t2 = spot_tn2

        cash_flow_t[j] = (option.payoff(spot_t) + beta * convar + option.payoff(spot_t2) + beta * convar2)/2

    prc = np.exp(-rate * expiry) * cash_flow_t.mean()
    se = np.std(cash_flow_t, ddof=1) / np.sqrt(reps)
    
    return PricerResult(prc, se)



def BlackScholesGamma(spot, t, strike, expiry, vol, rate, div):
    tau = expiry - t
    d1 = (np.log(spot/strike) + (rate - div + .5 * vol * vol) * tau) / (vol * np.sqrt(tau))
    gamma = np.exp(-div * (tau)) * norm.pdf(d1) / (spot * vol * np.sqrt(tau))
    return gamma


# Setting initial values
strike = 100
expiry = 1
spot = 100
vol = .2
rate = .06
div = .03
n = 52
steps = n
nreps = 10000
M = nreps
h = expiry/n
nudt = (rate - div - 0.5 * vol * vol) * h
sigsdt = vol * np.sqrt(h)
erddt = np.exp((rate - div) * h)
egamma = np.exp((2 * (rate - div) + (vol * vol)) * h) - (2 * erddt) + 1
beta = -1
beta2 = -.5

def control_variate_pricer_gamma(option, spot, rate, vol, div, steps, reps, beta, beta2):
    PricerResult = namedtuple('PricerResult', ['price', 'stderr'])
    expiry = option.expiry
    strike = option.strike
    h = expiry / steps
    nudt = (rate - div - 0.5 * vol * vol) * h
    sigsdt = vol * np.sqrt(h)
    erddt = np.exp((rate - div) * h)
    egamma = np.exp((2 * (rate - div) + (vol * vol)) * h) - (2 * erddt) + 1
    cash_flow_t = np.zeros(reps)
    cash_flow_t2 = np.zeros(reps)
    price = 0.0

    for j in range(reps):
        spot_t = spot
        spot_t2 = spot
        convar = 0.0
        convar2 = 0.0
        z = np.random.normal(size=int(steps))

        for i in range(int(steps)):
            t = i * h
            delta = BlackScholesDelta(spot_t, t, strike, expiry, vol, rate, div)
            delta2 = BlackScholesDelta(spot_t2, t, strike, expiry, vol, rate, div)
            gamma = BlackScholesGamma(spot_t, t, strike, expiry, vol, rate, div)
            gamma2 = BlackScholesGamma(spot_t2, t, strike, expiry, vol, rate, div)
            spot_tn = spot_t * np.exp(nudt + sigsdt * z[i])
            spot_tn2 = spot_t2 * np.exp(nudt + sigsdt * -z[i])
            convar = convar + delta * (spot_tn - spot_t * erddt) + delta2 * ((spot_tn2 - spot_t2 * erddt))
            convar2 = convar2 + gamma * ((spot_tn - spot_t)**2 - spot_t**2 * egamma) + gamma2 * ((spot_tn2 - spot_t2)**2 - spot_t2**2 * egamma)
            spot_t = spot_tn
            spot_t2 = spot_tn2

            cash_flow_t[j] = (option.payoff(spot_t) + beta * convar + option.payoff(spot_t2) + beta2 * convar2)/2

    prc = np.exp(-rate * expiry) * cash_flow_t.mean()
    se = np.std(cash_flow_t, ddof=1) / np.sqrt(reps)
    
    return PricerResult(prc, se)

    
if __name__ == "__main__":
    print("This is a module. Not intended to be run standalone")
    