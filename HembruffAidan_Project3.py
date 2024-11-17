# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 15:48:01 2024

@author: aidan
"""
# Project 3

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Part 1

def ODE(r, ystate):
    rho = ystate[0]
    m = ystate[1]
    
    x = rho**(1/3)
    gamma = (x**2)/(3*(np.sqrt(1+(x**2))))
    
    drho = -(m*rho)/(gamma*(r**2))
    dm = (r**2)*rho
    
    dystate = [drho,dm]
    
    return dystate

rho_c = np.linspace(1e-1,2.5e6,10)
r_outer = np.sqrt((6*np.sqrt((rho_c**(2/3))+1))/(rho_c**(1/3)))

soln_list = []
for i in range(len(rho_c)):
    boundary_conds = [rho_c[i],0]
    soln = solve_ivp(ODE,(1e-6,r_outer[i]),boundary_conds,method="RK45")
    soln_list.append(soln)
    plt.plot(soln.t,soln.y[0])
    plt.show()