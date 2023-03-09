# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 21:54:13 2023

@author: 26 Vetrax 99
"""
# FCTT Project 3
# Part 4: Improvement of the model introducing a degradation mechanism

# %% Importing Packages

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from shapely.geometry import LineString

# Battery Datasheet Values and Constants

Nom_capacity = 2500E-3  # Nominal Discharge Capacity of INR18650-25R Battery
R_u = 8.314             # Universal Gas Constant

# Reading Files

df_cycle = pd.read_csv('Degradation_45DegC_1C.csv')
df_cycle_R = pd.read_csv('Degradation_45DegC_1C_resistance.csv')
df_calendar = pd.read_excel('Calendar_ageing_60degC_SOC_100.xlsx')
df_calendar_R = pd.read_excel('Calendar_ageing_60degC_SOC_100_resistance_increase.xlsx')

T_cycle = 45 + 273.15
T_calendar = 60 + 273.15
Ea_cycle = 22406
Ea_cycle_R = 51800

# %% General Arrhenius Function for Cycle and Calendar Ageing

# Cycle Ageing

def cycle_capacity_loss(x, a, z): # x refers to either energy throughput or time. Energy throughput for this function
    return a * np.exp(-Ea_cycle/(R_u*T_cycle)) * (x ** z) 

def cycle_resistance_increase(x, a):
    return a * np.exp(-Ea_cycle_R/(R_u*T_cycle)) * x

# Calendar Ageing

def calendar_capacity_loss(x, a, z):
    return a * np.exp(-Ea_cycle/(R_u*T_calendar)) * (x ** z)

def calendar_resistance_increase(x, a):
    return a * np.exp(-Ea_cycle_R/(R_u*T_calendar)) * x

# %% Parameter Fitting

# Cycle-Ageing Model

# Capacity Loss

Ah_cycle = np.array((df_cycle['Cycle'].values)*Nom_capacity*2)  # *2 due to charging and discharging in one full cycle
S_loss_cycle = np.array(100 - df_cycle['Capacity %'].values)

fig, ax = plt.subplots()

ax.scatter(Ah_cycle, S_loss_cycle, c= 'deepskyblue', marker = 'o', label = 'Experimental Data at 45$\degree$C', 
           s = 60, zorder = 1, edgecolor = 'k')
popt_cycle, pcov_cycle = curve_fit(cycle_capacity_loss, Ah_cycle, S_loss_cycle, 
                                   bounds = ([-np.inf, -np.inf], 
                                             [np.inf, np.inf]))
ax.plot(Ah_cycle, cycle_capacity_loss(Ah_cycle, *popt_cycle), linestyle = 'dashed', c = 'deepskyblue', zorder = 0, label = 'Fitted Line')

ax.set_xlabel('Energy Throughput [Ah]', labelpad = 10)
ax.set_ylabel('Capacity Loss [%]', labelpad = 10)
ax.grid(True, alpha = 0.5)
ax.legend(loc = 'best')
ax.set_title('Parameter Fitting of Capacity Loss for Cycle Ageing Model')
fig.tight_layout()

# Resistance Increase

fig, ax = plt.subplots()

Ah_cycle_R = np.array((df_cycle_R['Cycle'].values)*Nom_capacity*2)
R_increase_cycle = np.array(df_cycle_R['Resistance %'].values)
ax.scatter(Ah_cycle_R, R_increase_cycle, c= 'magenta', marker = 'v', label = 'Experimental Data at 45$\degree$C', 
           s= 60, zorder = 1, edgecolor = 'k')
popt_cycle_R, pcov_cycle_R = curve_fit(cycle_resistance_increase, Ah_cycle_R, R_increase_cycle,
                                       bounds = ([-np.inf], [np.inf]))
ax.plot(Ah_cycle_R, cycle_resistance_increase(Ah_cycle_R, *popt_cycle_R), linestyle = 'dashed', 
        c = 'magenta', zorder = 0, label = 'Fitted Line')

ax.set_xlabel('Energy Throughput [Ah]', labelpad = 10)
ax.set_ylabel('Resistance Increase [%]', labelpad = 10)
ax.grid(True, alpha = 0.5)
ax.legend(loc = 'best')
ax.set_title('Parameter Fitting of Resistance Increase for Cycle Ageing Model')
fig.tight_layout()

# Calendar-Ageing Model

# Capacity Loss

fig, ax = plt.subplots()

Ah_calendar = np.array((df_calendar['Time(h)'].values)*Nom_capacity*2)
S_loss_calendar = np.array(df_calendar['Capacity loss %'].values)

ax.scatter(Ah_calendar, S_loss_calendar, c = 'deepskyblue', marker = 'o', label = 'Experimental Data at 60$\degree$C',
           s = 60, zorder = 1, edgecolor = 'k')
popt_calendar, pcov_calendar = curve_fit(calendar_capacity_loss, Ah_calendar, S_loss_calendar,
                                         bounds = ([-np.inf, -np.inf], 
                                                  [np.inf, np.inf]))
ax.plot(Ah_calendar, calendar_capacity_loss(Ah_calendar, *popt_calendar), linestyle = 'dashed', c = 'deepskyblue', zorder = 0, 
        label = 'Fitted Line')


ax.set_xlabel('Time [h]', labelpad = 10)
ax.set_ylabel('Capacity Loss [%]', labelpad = 10)
ax.grid(True, alpha = 0.5)
ax.legend(loc = 'best')
ax.set_title('Parameter Fitting of Capacity Loss for Calendar Ageing Model')
fig.tight_layout()

# Resistance Increase

fig, ax = plt.subplots()

Ah_calendar_R = np.array((df_calendar_R['Time(h)'].values)*Nom_capacity*2)
R_increase_calendar = np.array(df_calendar_R['Resistance increase[%]'].values)

ax.scatter(Ah_calendar_R, R_increase_calendar, c= 'magenta', marker = 'v', label = 'Experimental Data at 60$\degree$C',
           s = 60, zorder = 1, edgecolor = 'k')
popt_calendar_R, pcov_calendar_R = curve_fit(calendar_resistance_increase, Ah_calendar_R, R_increase_calendar,
                                             bounds = ([-np.inf], [np.inf]))
ax.plot(Ah_calendar_R, calendar_resistance_increase(Ah_calendar, *popt_calendar_R), linestyle = 'dashed', c = 'magenta',
        zorder = 0, label = 'Fitted Line')

ax.set_xlabel('Time [h]', labelpad = 10)
ax.set_ylabel('Resistance Increase [%]', labelpad = 10)
ax.grid(True, alpha = 0.5)
ax.legend(loc = 'best')
ax.set_title('Parameter Fitting of Resistance Increase for Calendar Ageing Model')
fig.tight_layout()


# %% Questions

# Temperature Dependent Equation

def cycle_capacity_loss_Q(x, a, z, T_Q): # x refers to either energy throughput or time. Energy throughput for this function
    return a * np.exp(-Ea_cycle/(R_u*T_Q)) * (x ** z) 

def cycle_resistance_increase_Q(x, a, T_Q):
    return a * np.exp(-Ea_cycle_R/(R_u*T_Q)) * x

def calendar_capacity_loss_Q(x, a, z, T_Q):
    return a * np.exp(-Ea_cycle/(R_u*T_Q)) * (x ** z)

def calendar_resistance_increase_Q(x, a, T_Q):
    return a * np.exp(-Ea_cycle_R/(R_u*T_Q)) * x

# Q1: Plot Capacity loss against Energy throughput

T_range_degC = np.arange(10, 70, 10) # Temperature Range to Test Over
T_range = [i + 273.15 for i in T_range_degC] # Changing from deg C to K

fig, ax = plt.subplots()

for i in range(0, len(T_range)):
    Q1_Ah_cycle = np.linspace(0, 5000*Nom_capacity, 5000)
    Q1_capacity_loss = cycle_capacity_loss_Q(Q1_Ah_cycle, popt_cycle[0], popt_cycle[1], T_range[i])
    ax.plot(Q1_Ah_cycle, Q1_capacity_loss, zorder = 0, label = str(T_range_degC[i]) + '$\degree$C')

ax.set_xlabel('Energy Throughput [Ah]', labelpad = 10)
ax.set_ylabel('Capacity Loss [%]', labelpad = 10)
ax.grid(True, alpha = 0.5)
ax.legend(loc = 'best', bbox_to_anchor=(0.75, 0.25, 0.5, 0.5))
ax.set_title('Question 1')
fig.tight_layout()

# Q2: Plot resistance increase against energy throughput

fig, ax = plt.subplots()

for i in range(0, len(T_range)):
    Q2_Ah_cycle = np.linspace(0, 5000*Nom_capacity, 5000)
    Q2_capacity_loss = cycle_resistance_increase_Q(Q2_Ah_cycle, popt_cycle_R[0], T_range[i])
    ax.plot(Q2_Ah_cycle, Q2_capacity_loss, zorder = 0, label = str(T_range_degC[i]) + '$\degree$C')

ax.set_xlabel('Energy Throughput [Ah]', labelpad = 10)
ax.set_ylabel('Resistance Increase [%]', labelpad = 10)
ax.grid(True, alpha = 0.5)
ax.legend(loc = 'best', bbox_to_anchor=(0.75, 0.25, 0.5, 0.5))
ax.set_title('Question 2')
fig.tight_layout()


# Q3: Plot Capacity Loss against Storage Time

Time = 10*365*24 # Time in hours

fig, ax = plt.subplots()

for i in range(0, len(T_range)):
    Q3_time = np.linspace(0, Time, 5000)
    Q3_capacity_loss = calendar_capacity_loss_Q(Q3_time, popt_calendar[0], popt_calendar[1], T_range[i])
    ax.plot(Q3_time, Q3_capacity_loss, zorder = 0, label = str(T_range_degC[i]) + '$\degree$C')

ax.set_xlabel('Time [h]', labelpad = 10)
ax.set_ylabel('Capacity Loss [%]', labelpad = 10)
ax.grid(True, alpha = 0.5)
ax.legend(loc = 'best', bbox_to_anchor=(0.75, 0.25, 0.5, 0.5))
ax.set_title('Question 3')
fig.tight_layout()

# Q4: Plot Resistance Increase against Storage Time

#curve_func(x, a, E, T, z)

fig, ax = plt.subplots()

for i in range(0, len(T_range)):
    Q4_time = np.linspace(0, Time, 5000)
    Q4_resistance_increase = calendar_resistance_increase_Q(Q4_time, popt_calendar_R[0], T_range[i])
    ax.plot(Q4_time, Q4_resistance_increase, zorder = 0, label = str(T_range_degC[i]) + '$\degree$C')

ax.set_xlabel('Time [h]', labelpad = 10)
ax.set_ylabel('Resistance Increase [%]', labelpad = 10)
ax.grid(True, alpha = 0.5)
ax.legend(loc = 'best', bbox_to_anchor=(0.75, 0.25, 0.5, 0.5))
ax.set_title('Question 4')
fig.tight_layout()

# %% Discussion

# Q1: How long battery lasts at different operating temperatures, assuming 30% capacity fade or 100% resistance increase is
# end of life. 

# Using calendar ageing model

end_capacity = 30
end_resistance = 100 
x_time = np.linspace(0, Time, 5000)
y_end_capacity = np.full((len(x_time), 1), end_capacity)
y_end_resistance = np.full((len(x_time), 1), end_resistance)

fig, ax = plt.subplots()

ax.plot(x_time, y_end_capacity, linestyle = 'dashed', c = 'black', zorder = 1)

for i in range(0, len(T_range)):
    Q3_time = np.linspace(0, Time, 5000)
    Q3_capacity_loss = calendar_capacity_loss_Q(Q3_time, popt_calendar[0], popt_calendar[1], T_range[i])
    ax.plot(Q3_time, Q3_capacity_loss, zorder = 0, label = str(T_range_degC[i]) + '$\degree$C')
    line_Q3_capacity_loss = LineString(np.column_stack((Q4_time, Q3_capacity_loss)))
    line_weight = LineString(np.column_stack((x_time, y_end_capacity)))
    intersection_capacity = line_Q3_capacity_loss.intersection(line_weight)
    ax.plot(*intersection_capacity.xy, 'ro')

ax.set_xlabel('Time [h]', labelpad = 10)
ax.set_ylabel('Capacity Loss [%]', labelpad = 10)
ax.grid(True, alpha = 0.5)
ax.legend(loc='best', bbox_to_anchor=(0.75, 0.25, 0.5, 0.5))
ax.set_title('Discussion 1 (End of Life)')
fig.tight_layout()

fig, ax = plt.subplots()

ax.plot(x_time, y_end_resistance, linestyle = 'dashed',c = 'black', zorder = 1)

for i in range(0, len(T_range)):
    Q4_time = np.linspace(0, Time, 5000)
    Q4_resistance_increase = calendar_resistance_increase_Q(Q4_time, popt_calendar_R[0], T_range[i])
    ax.plot(Q4_time, Q4_resistance_increase, zorder = 0, label = str(T_range_degC[i]) + '$\degree$C')
    line_Q4_resistance_increase = LineString(np.column_stack((Q4_time, Q4_resistance_increase)))
    line_weight = LineString(np.column_stack((x_time, y_end_resistance)))
    intersection_resistance = line_Q4_resistance_increase.intersection(line_weight)
    ax.plot(*intersection_resistance.xy, 'ro')
    # 60 to 0

ax.set_xlabel('Time [h]', labelpad = 10)
ax.set_ylabel('Resistance Increase [%]', labelpad = 10)
ax.grid(True, alpha = 0.5)
ax.legend(loc = 'best', bbox_to_anchor=(0.75, 0.25, 0.5, 0.5))
ax.set_title('Discussion 1 (End of Life)')
fig.tight_layout()

















