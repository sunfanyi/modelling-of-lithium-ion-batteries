# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 21:54:13 2023

@author: 26 Vetrax 99
"""
# FCTT Project 3
# Part 4: Improvement of the model introducing a degradation mechanism

# %% Importing Packages

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from shapely.geometry import LineString

# Battery Datasheet Values and Constants

Nom_capacity = 2500E-3  # Nominal Discharge Capacity of INR18650-25R Battery
R_u = 8.314             # Universal Gas Constant

# Specificying Local Directory

data_dir = os.path.abspath('battery_experimental_data/RE__Data_for_Degradation')
    
# Reading Files

df_cycle = pd.read_csv(os.path.join(data_dir, 'Degradation_45DegC_1C.csv'))
df_cycle_R = pd.read_csv(os.path.join(data_dir, 'Degradation_45DegC_1C_resistance.csv'))
df_calendar = pd.read_excel(os.path.join(data_dir, 'Calendar_ageing_60degC_SOC_100.xlsx'))
df_calendar_R = pd.read_excel(os.path.join(data_dir, 'Calendar_ageing_60degC_SOC_100_resistance_increase.xlsx'))

# Constant Values

T_cycle = 45 + 273.15                   # Temperature at which experiment for cycle ageing was conducted
T_calendar = 60 + 273.15                # Temperature at which experiment for calendar ageing was conducted
Ea_cycle = 22406                        # Activation Energy for Capacity Loss (Andrea et.al)
Ea_cycle_R = 51800                      # Activation Energy for Resistance Increase (Andrea et.al)

# %% General Arrhenius Function for Cycle and Calendar Ageing

# Cycle Ageing Function

def cycle_capacity_loss(x, a, z): # x refers to either energy throughput or time. Energy throughput for this function
    return a * np.exp(-Ea_cycle/(R_u*T_cycle)) * (x ** z) 

def cycle_resistance_increase(x, a):
    return a * np.exp(-Ea_cycle_R/(R_u*T_cycle)) * x

# Calendar Ageing Function

def calendar_capacity_loss(x, a, z):
    return a * np.exp(-Ea_cycle/(R_u*T_calendar)) * (x ** z)

def calendar_resistance_increase(x, a):
    return a * np.exp(-Ea_cycle_R/(R_u*T_calendar)) * x

# %% Parameter Fitting

# Part 1: Cycle-Ageing Model

# Part 1.1: Capacity Loss

# Extracting Experimental Data. Ah_cycle is Energy Throughput. S_loss_cycle is capacity loss
Ah_cycle = np.array((df_cycle['Cycle'].values)*Nom_capacity*2)  # *2 due to charging and discharging in one full cycle
S_loss_cycle = np.array(100 - df_cycle['Capacity %'].values)

# Parameter Fitting to Experimental Data
popt_cycle, pcov_cycle = curve_fit(cycle_capacity_loss, Ah_cycle, S_loss_cycle,  bounds = ([-np.inf, -np.inf], [np.inf, np.inf]))

# Capacity Loss vs Energy Throughput Plot
fig, ax = plt.subplots()
fig.tight_layout()

ax.scatter(Ah_cycle, S_loss_cycle, c= 'deepskyblue', marker = 'o', label = 'Experimental Data at 45$\degree$C', 
           s = 60, zorder = 1, edgecolor = 'k')
ax.plot(Ah_cycle, cycle_capacity_loss(Ah_cycle, *popt_cycle), linestyle = 'dashed', c = 'deepskyblue', zorder = 0, label = 'Fitted Line')
ax.set_xlabel('Energy Throughput [Ah]', labelpad = 10)
ax.set_ylabel('Capacity Loss [%]', labelpad = 10)
ax.grid(True, alpha = 0.5)
ax.legend(loc = 'best')
plt.savefig('Parameter_Fitting_Cycle_CapacityLoss.pdf', bbox_inches='tight')
ax.set_title('Parameter Fitting of Capacity Loss for Cycle Ageing Model')

# Part 1.2: Resistance Increase

# Extracting Experimental Data. Ah_cycle_R is Energy Throughput. R_increase_cycle is resistance increase
Ah_cycle_R = np.array((df_cycle_R['Cycle'].values)*Nom_capacity*2)
R_increase_cycle = np.array(df_cycle_R['Resistance %'].values)

# Parameter Fitting to Experimental Data
popt_cycle_R, pcov_cycle_R = curve_fit(cycle_resistance_increase, Ah_cycle_R, R_increase_cycle, bounds = ([-np.inf], [np.inf]))

# Resistance Increase vs Energy Throughput Plot
fig, ax = plt.subplots()
fig.tight_layout()

ax.scatter(Ah_cycle_R, R_increase_cycle, c= 'magenta', marker = 'v', label = 'Experimental Data at 45$\degree$C', 
           s= 60, zorder = 1, edgecolor = 'k')
ax.plot(Ah_cycle_R, cycle_resistance_increase(Ah_cycle_R, *popt_cycle_R), linestyle = 'dashed', 
        c = 'magenta', zorder = 0, label = 'Fitted Line')
ax.set_xlabel('Energy Throughput [Ah]', labelpad = 10)
ax.set_ylabel('Resistance Increase [%]', labelpad = 10)
ax.grid(True, alpha = 0.5)
ax.legend(loc = 'best')
plt.savefig('Parameter_Fitting_Cycle_ResistanceIncrease.pdf', bbox_inches='tight')
ax.set_title('Parameter Fitting of Resistance Increase for Cycle Ageing Model')


# %% Parameter Fitting

# Part 2: Calendar-Ageing Model

# Part 2.1: Capacity Loss

# Extracting Experimental Data. Ah_calendar is time(h). S_loss_calendar is capacity loss
Ah_calendar = np.array((df_calendar['Time(h)'].values)*Nom_capacity*2)
S_loss_calendar = np.array(df_calendar['Capacity loss %'].values)

# Parameter Fitting to Experimental Data
popt_calendar, pcov_calendar = curve_fit(calendar_capacity_loss, Ah_calendar, S_loss_calendar,bounds = ([-np.inf, -np.inf], [np.inf, np.inf]))

# Capacity Loss vs Time(h) Plot
fig, ax = plt.subplots()
fig.tight_layout()

ax.scatter(Ah_calendar, S_loss_calendar, c = 'deepskyblue', marker = 'o', label = 'Experimental Data at 60$\degree$C', s = 60, zorder = 1, edgecolor = 'k')
ax.plot(Ah_calendar, calendar_capacity_loss(Ah_calendar, *popt_calendar), linestyle = 'dashed', c = 'deepskyblue', zorder = 0, label = 'Fitted Line')
ax.set_xlabel('Time [h]', labelpad = 10)
ax.set_ylabel('Capacity Loss [%]', labelpad = 10)
ax.grid(True, alpha = 0.5)
ax.legend(loc = 'best')
plt.savefig('Parameter_Fitting_Calendar_CapacityLoss.pdf', bbox_inches='tight')
ax.set_title('Parameter Fitting of Capacity Loss for Calendar Ageing Model')

# Resistance Increase

# Extracting Experimental Data. Ah_calendar_R is time(h). R_increase_calendar is capacity loss
Ah_calendar_R = np.array((df_calendar_R['Time(h)'].values)*Nom_capacity*2)
R_increase_calendar = np.array(df_calendar_R['Resistance increase[%]'].values)

# Parameter Fitting to Experimental Data
popt_calendar_R, pcov_calendar_R = curve_fit(calendar_resistance_increase, Ah_calendar_R, R_increase_calendar, bounds = ([-np.inf], [np.inf]))

# Capacity Loss vs Time(h) Plot

fig, ax = plt.subplots()
fig.tight_layout()

ax.scatter(Ah_calendar_R, R_increase_calendar, c= 'magenta', marker = 'v', label = 'Experimental Data at 60$\degree$C',
           s = 60, zorder = 1, edgecolor = 'k')
ax.plot(Ah_calendar_R, calendar_resistance_increase(Ah_calendar, *popt_calendar_R), linestyle = 'dashed', c = 'magenta',
        zorder = 0, label = 'Fitted Line')
ax.set_xlabel('Time [h]', labelpad = 10)
ax.set_ylabel('Resistance Increase [%]', labelpad = 10)
ax.grid(True, alpha = 0.5)
ax.legend(loc = 'best')
plt.savefig('Parameter_Fitting_Calendar_ResistanceIncrease.pdf', bbox_inches='tight')
ax.set_title('Parameter Fitting of Resistance Increase for Calendar Ageing Model')

# %% Questions

# Temperature Dependent Equation. Earlier function set T as a constant (following T of experiment)

def cycle_capacity_loss_Q(x, a, z, T_Q): # x refers to either energy throughput or time. Energy throughput for this function
    return a * np.exp(-Ea_cycle/(R_u*T_Q)) * (x ** z) 

def cycle_resistance_increase_Q(x, a, T_Q):
    return a * np.exp(-Ea_cycle_R/(R_u*T_Q)) * x

def calendar_capacity_loss_Q(x, a, z, T_Q):
    return a * np.exp(-Ea_cycle/(R_u*T_Q)) * (x ** z)

def calendar_resistance_increase_Q(x, a, T_Q):
    return a * np.exp(-Ea_cycle_R/(R_u*T_Q)) * x

# Plot Capacity loss against Energy throughput

T_range_degC = np.arange(10, 70, 10) # Temperature Range to Test Over
T_range = [i + 273.15 for i in T_range_degC] # Changing from deg C to K

fig, ax = plt.subplots()
fig.tight_layout()

for i in range(0, len(T_range)):
    Q1_Ah_cycle = np.linspace(0, 5000*Nom_capacity, 5000)
    Q1_capacity_loss = cycle_capacity_loss_Q(Q1_Ah_cycle, popt_cycle[0], popt_cycle[1], T_range[i])
    ax.plot(Q1_Ah_cycle, Q1_capacity_loss, zorder = 0, label = str(T_range_degC[i]) + '$\degree$C')

ax.set_xlabel('Energy Throughput [Ah]', labelpad = 10)
ax.set_ylabel('Capacity Loss [%]', labelpad = 10)
ax.grid(True, alpha = 0.5)
ax.legend(loc = 'best', bbox_to_anchor=(0.7, 0.25, 0.5, 0.5))
plt.savefig('Part4a.pdf', bbox_inches='tight')
ax.set_title('Part 4a: Cycle Capacity Loss vs Energy Throughput')

# Plot resistance increase against energy throughput

fig, ax = plt.subplots()
fig.tight_layout()

for i in range(0, len(T_range)):
    Q2_Ah_cycle = np.linspace(0, 5000*Nom_capacity, 5000)
    Q2_capacity_loss = cycle_resistance_increase_Q(Q2_Ah_cycle, popt_cycle_R[0], T_range[i])
    ax.plot(Q2_Ah_cycle, Q2_capacity_loss, zorder = 0, label = str(T_range_degC[i]) + '$\degree$C')

ax.set_xlabel('Energy Throughput [Ah]', labelpad = 10)
ax.set_ylabel('Resistance Increase [%]', labelpad = 10)
ax.grid(True, alpha = 0.5)
ax.legend(loc = 'best', bbox_to_anchor=(0.7, 0.25, 0.5, 0.5))
plt.savefig('Part4b.pdf', bbox_inches='tight')
ax.set_title('Part 4b: Cycle Resistance Increase vs Energy Throughput')

# Plot Capacity Loss against Storage Time

Time = 10*365*24 # Time in hours

fig, ax = plt.subplots()
fig.tight_layout()

for i in range(0, len(T_range)):
    Q3_time = np.linspace(0, Time, 5000)
    Q3_capacity_loss = calendar_capacity_loss_Q(Q3_time, popt_calendar[0], popt_calendar[1], T_range[i])
    ax.plot(Q3_time, Q3_capacity_loss, zorder = 0, label = str(T_range_degC[i]) + '$\degree$C')

ax.set_xlabel('Time [h]', labelpad = 10)
ax.set_ylabel('Capacity Loss [%]', labelpad = 10)
ax.grid(True, alpha = 0.5)
ax.legend(loc = 'best', bbox_to_anchor=(0.7, 0.25, 0.5, 0.5))
plt.savefig('Part4c.pdf', bbox_inches='tight')
ax.set_title('Part 4c: Calendar Capacity Loss vs Time')


# Plot Resistance Increase against Storage Time

#curve_func(x, a, E, T, z)

fig, ax = plt.subplots()
fig.tight_layout()

for i in range(0, len(T_range)):
    Q4_time = np.linspace(0, Time, 5000)
    Q4_resistance_increase = calendar_resistance_increase_Q(Q4_time, popt_calendar_R[0], T_range[i])
    ax.plot(Q4_time, Q4_resistance_increase, zorder = 0, label = str(T_range_degC[i]) + '$\degree$C')

ax.set_xlabel('Time [h]', labelpad = 10)
ax.set_ylabel('Resistance Increase [%]', labelpad = 10)
ax.grid(True, alpha = 0.5)
ax.legend(loc = 'best', bbox_to_anchor=(0.7, 0.25, 0.5, 0.5))
plt.savefig('Part4d.pdf', bbox_inches='tight')
ax.set_title('Part 4d: Calendar Capacity Loss vs Time')

# %% Discussion

# Question 4.3: How long battery lasts at different operating temperatures, assuming 30% capacity fade or 100% resistance increase is
# end of life. 

# Using calendar ageing model

# Defining End Life Properties

end_capacity = 30
end_resistance = 100 
x_time = np.linspace(0, Time, 5000)
y_end_capacity = np.full((len(x_time), 1), end_capacity)
y_end_resistance = np.full((len(x_time), 1), end_resistance)

# Plot for Capacity Loss End of Life

fig, ax = plt.subplots()
fig.tight_layout()

for i in range(0, len(T_range)):
    Q3_time = np.linspace(0, Time, 5000)
    Q3_capacity_loss = calendar_capacity_loss_Q(Q3_time, popt_calendar[0], popt_calendar[1], T_range[i])
    ax.plot(Q3_time, Q3_capacity_loss, zorder = 0, label = str(T_range_degC[i]) + '$\degree$C')
    line_Q3_capacity_loss = LineString(np.column_stack((Q4_time, Q3_capacity_loss)))
    line_weight = LineString(np.column_stack((x_time, y_end_capacity)))
    intersection_capacity = line_Q3_capacity_loss.intersection(line_weight)
    #ax.plot(*intersection_capacity.xy, 'ro')

ax.plot(x_time, y_end_capacity, linestyle = 'dashed', c = 'black', zorder = 1)
ax.set_xlabel('Time [h]', labelpad = 10)
ax.set_ylabel('Capacity Loss [%]', labelpad = 10)
ax.grid(True, alpha = 0.5)
ax.legend(loc='best', bbox_to_anchor=(0.7, 0.25, 0.5, 0.5))
plt.savefig('Part4_Discussion1_CapacityLoss.pdf', bbox_inches='tight')
ax.set_title('Discussion 1 (Capacity Loss End of Life)')

# Plot for Resistance Increase End of Life

fig, ax = plt.subplots()
fig.tight_layout()

for i in range(0, len(T_range)):
    Q4_time = np.linspace(0, Time, 5000)
    Q4_resistance_increase = calendar_resistance_increase_Q(Q4_time, popt_calendar_R[0], T_range[i])
    ax.plot(Q4_time, Q4_resistance_increase, zorder = 0, label = str(T_range_degC[i]) + '$\degree$C')
    line_Q4_resistance_increase = LineString(np.column_stack((Q4_time, Q4_resistance_increase)))
    line_weight = LineString(np.column_stack((x_time, y_end_resistance)))
    intersection_resistance = line_Q4_resistance_increase.intersection(line_weight)
    ax.plot(*intersection_resistance.xy, 'ro')
    
ax.plot(x_time, y_end_resistance, linestyle = 'dashed',c = 'black', zorder = 1)
ax.set_xlabel('Time [h]', labelpad = 10)
ax.set_ylabel('Resistance Increase [%]', labelpad = 10)
ax.grid(True, alpha = 0.5)
ax.legend(loc = 'best', bbox_to_anchor=(0.7, 0.25, 0.5, 0.5))
plt.savefig('Part4_Discussion1_ResistanceIncrease.pdf', bbox_inches='tight')
ax.set_title('Discussion 1 (Resistance Increase End of Life)')

# %% Question 4.4: Calculation of Time to End of Life of Life [hours]

# Function to calculate time to failure for calendar ageing model
def func_time_calendar(T, S_failure, R_failure):
    # Taking parameter fitted values
    b_C = popt_calendar[0]
    z = popt_calendar[1]
    b_R = popt_calendar_R[0]
    
    # Calculating failure time
    time_capacity = (S_failure/(b_C*np.exp(-Ea_cycle/(R_u*T))))**(1/z)
    time_resistance = R_failure/(b_R*np.exp(-Ea_cycle_R/(R_u*T)))
    # Converting time to years
    time_capacity_yr = np.round(time_capacity/(365*24), 2)
    time_resistance_yr = np.round(time_resistance/(365*24), 2)
    
    return min(time_capacity_yr, time_resistance_yr)

# Defining Failure Criteria
    
S_failure = 30
R_failure = 100

T_city = [['Cairo', 21.4], ['Kinshasa', 25.3],['Kuala Lumpur', 27.3],['Seoul', 12.5], ['London', 11.3],
          ['Rome', 15.2],  ['San Francisco', 14.6], ['Toronto', 9.4],['Auckland', 15.2],['Sydney', 17.1],
          ['Buenos Aires', 17.9], ['Sao Paulo', 19.2]]

for i in range(0, len(T_city)):
    print('T = ' + str(T_city[i][1]) + ' deg C. Failure Time (yrs) = ' + 
          str(func_time_calendar(T_city[i][1]+273.15, S_failure, R_failure)))

# %% Question 4.5: Temperature where vehicle would fail if an 8 year warranty is offered

def func_Q4_5(time, S_failure, R_failure):
    # Changing time in years to hours
    time_hr = time*365*24
    # Taking parameter fitted values
    b_C = popt_calendar[0]
    z = popt_calendar[1]
    b_R = popt_calendar_R[0]
    
    # Calculating failure temperature [K]
    T_failure_capacity = -(Ea_cycle/R_u)    *   (np.log((S_failure/(b_C*(time_hr**z))))**(-1))
    T_failure_resistance = -(Ea_cycle_R/R_u)    *   (np.log(R_failure/(b_R*time_hr))**(-1))
    
    # Converting to deg C
    T_K_capacity = np.round(T_failure_capacity - 273.15, 2)
    T_K_resistance = np.round(T_failure_resistance - 273.15, 2)
    print(T_K_capacity, T_K_resistance)
    return min(T_K_capacity, T_K_resistance)
    
warranty_time = 8 #years

print('For 8 year warranty, the temperature of the operating region must be below '
      + str(func_Q4_5(warranty_time, S_failure, R_failure)) + ' deg C')
    
    








