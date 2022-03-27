# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10

@author: jaehyuk
"""

from random import random
import numpy as np
import scipy.stats as ss
import scipy.optimize as sopt
import pyfeng as pf
import scipy.integrate as spint
from . import bsm
'''
MC model class for Beta=1
'''
class ModelBsmMC:
    beta = 1.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    bsm_model = None
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = pf.Bsm(sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        Use self.bsm_model.impvol() method
        '''
        price= self.price(strike,spot,texp,sigma)
        
        return self.bsm_model.impvol(price=price,strike=strike, spot=spot, texp=texp)
    
    def price(self, strike, spot, texp=None, sigma=None, cp=1,random =False):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''

        #Generate paths for vol
        time_step=0.05
        n= int(texp/time_step)
        if random ==False:
            np.random.seed(12345)
        Z_1 =np.random.normal(size=(n,10000))
        if random ==False:
            np.random.seed(12346)
        X_1 = np.random.normal(size=(n,10000))
        W_1 = self.rho * Z_1 + np.sqrt(1 - self.rho**2)*X_1
        
        sigma_t = self.sigma if sigma is None else sigma
        s_t = spot
        for t in range(n):
            s_t = s_t * np.exp(sigma_t*np.sqrt(time_step)*W_1[t,:] - 0.5*sigma_t**2*time_step)
            sigma_t = sigma_t * np.exp(self.vov*np.sqrt(time_step)*Z_1[t,:] - 0.5*self.vov**2*time_step)        
        
        final_price = s_t

        prices = []
        for strike_i in strike:
            price = np.mean(np.fmax(cp*(final_price - strike_i), 0))
            prices.append(price)
        return np.array(prices)
        #return np.zeros_like(strike)

'''
MC model class for Beta=0
'''
class ModelNormalMC:
    beta = 0.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    normal_model = None
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = pf.Norm(sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol.
        Use self.normal_model.impvol() method        
        '''
        #return np.zeros_like(strike)
        price= self.price(strike,spot,texp,sigma)
        
        return self.normal_model.impvol(price=price,strike=strike, spot=spot, texp=texp)
    
        
    def price(self, strike, spot, texp=None, sigma=None, cp=1,random = False):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        
        time_step=0.05
        n= int(texp/time_step)
        if random==False:
            np.random.seed(12345)
        Z_1 =np.random.normal(size=(n,10000))
        if random==False:
            np.random.seed(12346)
        X_1 = np.random.normal(size=(n,10000))
        W_1 = self.rho * Z_1 + np.sqrt(1 - self.rho**2)*X_1
        
        sigma_t = self.sigma if sigma is None else sigma
        s_t = spot
        for t in range(n):
            s_t = s_t + sigma_t*W_1[t,:]*np.sqrt(time_step)
            sigma_t = sigma_t * np.exp(self.vov*np.sqrt(time_step)*Z_1[t,:] - 0.5*self.vov**2*time_step)        
        
        final_price = s_t

        prices = []
        for strike_i in strike:
            price = np.mean(np.fmax(cp*(final_price - strike_i), 0))
            prices.append(price)
        return np.array(prices)

        
      

'''
Conditional MC model class for Beta=1
'''
class ModelBsmCondMC:
    beta = 1.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    bsm_model = None
    time_step = 0.01
    samples = 10000
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = pf.Bsm(sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None):
        ''''
        should be same as bsm_vol method in ModelBsmMC (just copy & paste)
        '''
        price= self.price(strike,spot,texp)
        
        return self.bsm_model.impvol(price=price,strike=strike, spot=spot, texp=texp)
    
    def price(self, strike, spot, texp=None, cp=1,random = False):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and BSM price.
        Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        if random == False:
            np.random.seed(12345)
        time_step=0.05
        n= int(texp/time_step)

        Z_1 =np.random.normal(size=(n,10000))
        sigma_path = np.ones_like(Z_1) *self.sigma
        for t in range(1,n-1):
            sigma_path[t+1,:]  = sigma_path[t,:]  * np.exp(self.vov*np.sqrt(time_step)*Z_1[t,:] - 0.5*self.vov**2*time_step)
  

        int_var = spint.simps(sigma_path**2, dx=1, axis=0)*time_step
        sigma_final = sigma_path[-1,:]      


        S_0 = spot*np.exp(self.rho*(sigma_final - self.sigma)/self.vov - 0.5*((self.rho**2)*(self.sigma**2)*texp*int_var))
        
        sigma_bs = np.sqrt((1-self.rho**2)*int_var)

        bsm =pf.Bsm(sigma_bs, intr=0, divr=0)
        prices = []
        for strike_i in strike:
            price = np.mean(bsm.price(spot=S_0,strike = strike_i,texp=texp,cp=cp))
            prices.append(price)
        return np.array(prices)
        #return np.zeros_like(strike)
        

'''
Conditional MC model class for Beta=0
'''
class ModelNormalCondMC:
    beta = 0.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    normal_model = None
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = pf.Norm(sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None):
        ''''
        should be same as norm_vol method in ModelNormalMC (just copy & paste)
        '''
        price= self.price(strike,spot,texp)
        
        return self.normal_model.impvol(price=price,strike=strike, spot=spot, texp=texp)
        
    def price(self, strike, spot,texp=None, cp=1,random = False):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and normal price.
        You may fix the random number seed
        '''
        if random == False:
            np.random.seed(12345)
  
        time_step=0.05
        n= int(texp/time_step)

        Z_1 =np.random.normal(size=(n,10000))
        sigma_path = np.ones_like(Z_1) *self.sigma
        for t in range(1,n-1):
            sigma_path[t+1,:]  = sigma_path[t,:]  * np.exp(self.vov*np.sqrt(time_step)*Z_1[t,:] - 0.5*self.vov**2*time_step)
  

        int_var = spint.simps(sigma_path**2, dx=1, axis=0)*time_step
        sigma_final = sigma_path[-1,:]      
        #print('sigma_final:',sigma_final)

        S_0 = spot+ self.rho*(sigma_final-self.sigma)/self.vov
        #sigma_n = self.sigma*np.sqrt((1-self.rho**2)*int_var)
        sigma_n = np.sqrt((1-self.rho**2)*int_var)

        norm =pf.Norm(sigma_n, intr=0, divr=0)
        prices = []
        for strike_i in strike:
            price = np.mean(norm.price(spot=S_0,strike = strike_i,texp=texp,cp=cp))
            prices.append(price)
        return np.array(prices)
