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

# function to pass to solve_ivp
def ODE(r, ystate):
    rho = ystate[0]  # dimensionless density
    m = ystate[1]    # dimensionless mass
    
    x = np.cbrt(rho)   # defining x to be used in gamma function
    gamma = (x**2)/(3*(np.sqrt(1+(x**2)))) # gamma function for derivative of rho
    
    drho = (-m*rho)/(gamma*(r**2)) # derivative of rho
    dm = (r**2)*rho                # derivative of mass
    
    dystate = [drho,dm]            # state vector derivative
    
    return dystate

# event function to find when mass and density are zero
def event(r,ystate):
    if ystate[0] - 1e-5 < 0:
        ystate[0] = 0
    return ystate[0] 

# stop integration when rho equals 0
event.terminal = True
event.direction= -1

# initializing the range of rho_c values and the outer limit of r for which rho = 0
rho_c = np.logspace(-1,6.3,10)

# desired number of nucleons per electron for this problem
mu_e = 2.
# constants which remove dimensionality from r and m, reintroduced to return
# dimensionality for plotting when solution is found
R_0 = (7.72e8)/mu_e
M_0 = (5.67e33)/(mu_e**2)

soln_list = []
# loop over all values of rho_c
for i in range(len(rho_c)):
    r_eval = np.linspace(1e-6,10,200)
    boundary_conds = [rho_c[i],0] # boundary conditions for each instance of rho_c
    # solve_ivp using RK45 method
    soln = solve_ivp(ODE,(1e-6,10),boundary_conds,method="RK45",t_eval = r_eval, events = event) 
    
    dimensional_r = soln.t_events[0]*R_0
    dimensional_m = soln.y_events[0][0][1]*M_0

    plt.scatter(dimensional_m,dimensional_r, label = rho_c[i])
    
plt.title("Radius as a function of Mass for White Dwarf Stars")
plt.xlabel("Mass (M)")
plt.ylabel("Radius (R)")
plt.legend()

plt.show()