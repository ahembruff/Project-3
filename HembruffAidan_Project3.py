# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 15:48:01 2024

@author: aidan
"""
# Project 3

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

# Part 1

def ODE(r, ystate):
    rho = ystate[0]
    m = ystate[1]
    
    x = rho**(1/3)
    gamma = (x**2)/(3*((1+(x**2))**0.5))
    
    drho = -(m*rho)/(gamma*(r**2))
    dm = (r**2)*rho
    
    dystate = [drho,dm]
    
    return dystate