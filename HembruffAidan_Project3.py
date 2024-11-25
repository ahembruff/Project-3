# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 15:48:01 2024

@author: aidan
"""
# Project 3

# imports
import numpy as np
from scipy.integrate import solve_ivp
import scipy.interpolate as interpolate
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pandas import read_csv

# function to pass to solve_ivp
def ODE(r, ystate):
    '''
    A function describes the relationship between a state vector
    for the white dwarf and its derivative. Passed to Solve_IVP

    Parameters
    ----------
    r : The radius of the white dwarf 
    
    ystate : The state vector, which is the initial (central) mass and density

    Returns
    -------
    dystate : The derivative of the state vector

    '''
    rho = ystate[0]  # dimensionless density
    m = ystate[1]    # dimensionless mass
    
    x = np.cbrt(rho)   # defining x to be used in gamma function
    gamma = (x**2)/(3*(np.sqrt(1+(x**2)))) # gamma function for derivative of rho
    
    drho = (-m*rho)/(gamma*(r**2)) # derivative of rho
    dm = (r**2)*rho                # derivative of mass
    
    dystate = [drho,dm]            # state vector derivative
    
    return dystate

# event function to find when mass and density are zero
def rho_zero_event(r,ystate):
    """
    A function which has the sole purpose of being used in the events
    option of Solve_IVP. Determines when the density at radius r is 0

    Parameters
    ----------
    r : The radius of the white dwarf
    
    ystate : The white dwarf state vector

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if ystate[0] - 1e-5 < 0:
        ystate[0] = 0
    return ystate[0] 

# stop integration when rho equals 0
rho_zero_event.terminal = True
# ensuring the direction is from above the x-axis when termination is reached
rho_zero_event.direction= -1

# initializing the range of rho_c values and the outer limit of r for which rho = 0
rho_c = np.logspace(-1,6.3,10)

# desired number of nucleons per electron for this problem
mu_e = 2.
# constants which remove dimensionality from r and m, reintroduced to return
# dimensionality for plotting when solution is found
R_0 = (7.72e8)/mu_e
M_0 = (5.67e33)/(mu_e**2)

# the mass and radius of the sun in cm and g respectively
R_sun = 6.96e10
M_sun = 1.99e33

# Initialize empty lists to store parameters of interest
R_diff_list = []
M_diff_list = []
Rho_diff_list = []

R1_list = []
M1_list = []
Rho1_list = []

R2_list = []
M2_list = []
Rho2_list = []

dimensional_r = []
dimensional_m = []

# loop over all values of rho_c
for i in range(len(rho_c)):
    r_eval = np.linspace(1e-6,10,200)
    boundary_conds = [rho_c[i],0] # boundary conditions for each instance of rho_c
    # solve_ivp using RK45 method
    soln = solve_ivp(ODE,(1e-6,10),boundary_conds,method="RK45",t_eval = r_eval, events =rho_zero_event) 
    
    # use a second integration method for three values of rho_c, for comparison of methods
    if i == 0 or i == 5 or i == 9:
        
        # solution using the DOP853 method
        new_soln = solve_ivp(ODE,(1e-6,10),boundary_conds,method="DOP853",t_eval = r_eval, events = rho_zero_event)
        
        # RK45 results
        r1 = soln.t_events[0][0]
        m1 = soln.y_events[0][0][1]
        
        R1_list.append(r1)
        M1_list.append(m1)
        
        # DOP853 results
        r2 = new_soln.t_events[0][0]
        m2 = new_soln.y_events[0][0][1]
        
        R2_list.append(r2)
        M2_list.append(m2)
        
        # finding the percent difference between values from each method
        r_diff = (np.abs(r2-r1)/((r2+r1)/2))*100
        m_diff = (np.abs(m2-m1)/((m2+m1)/2))*100
        
        R_diff_list.append(r_diff)
        M_diff_list.append(m_diff)
        
    dimensional_r.append(soln.t_events[0][0]*R_0/R_sun)
    dimensional_m.append(soln.y_events[0][0][1]*M_0/M_sun)

# printing relevant information for the report
print("The radius values using the RK45 method are: " + str(R1_list))
print("\nThe radius values using the DOP853 method are: " + str(R2_list))
print("\nThe mass values using the RK45 method are: " + str(M1_list))
print("\nThe mass values using the DOP853 method are: " + str(M2_list))

print("\nThe percent difference in radius values between the two integration methods is: " + str(R_diff_list))
print("\nThe percent difference in mass values between the two integration methods is: " + str(M_diff_list))


def linear(x,m,b):
    """
    Function of a line to be passed to curve_fit

    Parameters
    ----------
    x : The x values
    
    m : The slope of the line

    b : The y_intercept of the line

    Returns
    -------
    y : The equation of the line

    """
    y = m*x + b
    return y

# radius and mass values for the last three rho_c values
m_vals = np.asarray(dimensional_m[7:10])
r_vals = np.asarray(dimensional_r[7:10])

# linear fit for the last three points to allow for an estimate of the 
# Chandrasekhar limit when the radius reaches 0 for the mass-radius relationship curve
curve =  curve_fit(linear,m_vals,r_vals)
# the estimate is the y-intercept divided by the slope since y=0 at the intersection with the x-axis
Chandrasekhar_estimate = -curve[0][1]/curve[0][0]


print("\nThe estimation of the Chandrasekhar limit from the plot is: %f"%Chandrasekhar_estimate)
# Finding the difference between our estimate of the Chandrasekhar limit and the Weigert-Kippenhan estimate
limit_diff = (np.abs(Chandrasekhar_estimate - (5.836/4))/((Chandrasekhar_estimate+(5.836/4))/2))*100
print("\nThe percent difference between the estimated Chandrasekhar limit and Kippenhahn & Weigert's" +
      " estimation is : %f percent" %limit_diff)

# plotting the ODE solution and Chandrasekhar limit estimations
fig1 = plt.figure()
plt.plot(dimensional_m,dimensional_r, color = "red",label = "Solve_IVP ODE Solution",marker ='o')
plt.axvline(Chandrasekhar_estimate,color='orange',label="Chandrasekhar Limit Estimate", linestyle = "dashed")
plt.axvline(5.836/4,color='yellow',label="Kippenhahn & Weigert Chandrasekhar Limit", linestyle = "dashed")
plt.title("Radius as a function of Mass for White Dwarf Stars")
plt.xlabel("Mass (M_sun)")
plt.ylabel("Radius (R_sun)")
plt.legend()
plt.savefig("HembruffAidan_Project3_Fig1.png")
plt.close()
    

# Reading in the data file and assigning each column to a variable
data =  read_csv("C:/Users/aidan/Desktop/4th_Year/Comp_Sims/wd_mass_radius.csv")
mass_data =  data["M_Msun"]
mass_unc = data["M_unc"]
radius_data = data["R_Rsun"]
radius_unc = data["R_unc"]

# replot ODE solution to plot with observational data
fig2 = plt.figure()
plt.plot(dimensional_m,dimensional_r, color = "red",label = "Solve_IVP ODE Solution",marker ='o')
# plotting observational data
plt.scatter(mass_data,radius_data, label = "Tremblay et al.(2017) Data")
plt.errorbar(mass_data,radius_data,xerr=mass_unc,yerr=radius_unc,fmt='o')
plt.title("Radius as a function of Mass for White Dwarf Stars")
plt.xlabel("Mass (M_sun)")
plt.ylabel("Radius (R_sun)")
plt.legend()
plt.savefig("HembruffAidan_Project3_Fig2.png")
plt.close()

# END
