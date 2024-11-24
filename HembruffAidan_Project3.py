# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 15:48:01 2024

@author: aidan
"""
# Project 3

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from pandas import read_csv

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

R_sun = 6.96e10
M_sun = 1.99e33

# loop over all values of rho_c
for i in range(len(rho_c)):
    r_eval = np.linspace(1e-6,10,200)
    boundary_conds = [rho_c[i],0] # boundary conditions for each instance of rho_c
    # solve_ivp using RK45 method
    soln = solve_ivp(ODE,(1e-6,10),boundary_conds,method="RK45",t_eval = r_eval, events = event) 
    
    if i == 0 or 5 or 9:
        new_soln = solve_ivp(ODE,(1e-6,10),boundary_conds,method="DOP853",t_eval = r_eval, events = event)
        
        r1 = soln.t_events[0]
        m1 = soln.y_events[0][0][1]
        rho1 = soln.y_events[0][0][0]
        
        r2 = new_soln.t_events[0]
        m2 = new_soln.y_events[0][0][1]
        rho2 = new_soln.y_events[0][0][0]
        
        r_diff = np.abs(r2-r1)/((r2+r1)/2)
        m_diff = np.abs(m2-m1)/((m2+m1)/2)
        rho_diff = np.abs(rho2-rho1)/((rho2+rho1)/2)
        
    dimensional_r = soln.t_events[0]*R_0/R_sun
    dimensional_m = soln.y_events[0][0][1]*M_0/M_sun

    plt.scatter(dimensional_m,dimensional_r, label = rho_c[i], color = "blue")
    
plt.title("Radius as a function of Mass for White Dwarf Stars")
plt.xlabel("Mass (M_sun)")
plt.ylabel("Radius (R_sun)")
plt.legend()
plt.savefig("HembruffAidan_Project3_Fig1.png")


    

# %%
# Part 4

# Reading in the data file and assigning each column to a variable
data =  read_csv("C:/Users/aidan/Desktop/4th_Year/Comp_Sims/wd_mass_radius.csv")
mass_data =  data["M_Msun"]
mass_unc = data["M_unc"]
radius_data = data["R_Rsun"]
radius_unc = data["R_unc"]

# plotting observational data
plt.scatter(mass_data,radius_data)
plt.errorbar(mass_data,radius_data,mass_unc,radius_unc)

plt.show()
