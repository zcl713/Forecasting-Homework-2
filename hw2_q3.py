# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 18:21:47 2022

@author: waseem
"""
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from statsmodels.tsa.stattools import acovf
from matplotlib.ticker import FormatStrFormatter
import datetime as dt
from scipy import signal
from scipy.fft import fft, ifft
from matplotlib.ticker import AutoMinorLocator
import math
from statsmodels.formula.api import ols
from statsmodels.graphics.api import interaction_plot, abline_plot
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm


dv = pd.read_excel('Viscosity.xlsx')
dw = pd.read_excel('WholeFood.xlsx')
dp = pd.read_excel('Pharmaceutical.xlsx')


# Function to create plots for datasets without date column
def create_plots_no_date(data, value_col, dataset_name, ylabel='Value'):
    """
    Create autocorrelation and PSD plots for a given dataset (without date filtering)
    
    Parameters:
    - data: DataFrame containing the data
    - value_col: name of the value column
    - dataset_name: name for labeling plots and files
    - ylabel: label for y-axis in time series plot
    """
    # Get values for autocorrelation and PSD
    x = data[value_col].values
    
    # Remove NaN values if any
    x = x[~np.isnan(x)]
    
    # Plot time series
    plt.figure(figsize=(10, 7))
    plt.plot(x)
    plt.title(f'{dataset_name} Time Series')
    plt.ylabel(ylabel)
    plt.xlabel('Time t')
    plt.legend([f'{dataset_name}'])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_TimeSeries.png')
    plt.show()
    
    # Autocorrelation plot
    plt.figure(figsize=(10, 7))
    fig, ax = plt.subplots()
    ax.acorr(x, maxlags=30)
    plt.title(f'{dataset_name} Autocorrelation')
    minor_locator = AutoMinorLocator(10)
    ax.xaxis.set_minor_locator(minor_locator)
    plt.grid(which='minor')
    plt.legend(['$R_X$(\u03C4)'], loc='upper left')
    plt.xlabel('\u03C4')
    ax.grid(True, which='both')
    plt.tight_layout()
    fig.savefig(f'{dataset_name}_Stochastic_Autocorrelation.png')
    fig.show()
    
    # PSD plot
    freqs, psd = signal.welch(x)
    plt.figure(figsize=(10, 7))
    fig, ax = plt.subplots()
    ax.plot(freqs, psd)
    plt.title(f'{dataset_name} Power Spectral Density')
    minor_locator = AutoMinorLocator(5)
    ax.xaxis.set_minor_locator(minor_locator)
    plt.grid(which='minor')
    plt.legend(['PSD'])
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    ax.grid(True, which='both')
    plt.tight_layout()
    fig.savefig(f'{dataset_name}_Stochastic_PSD.png')
    fig.show()


# ============================================================================
# Original ICFN plots (Ice Cream)
# ============================================================================
dicfn = pd.read_csv('IPN31152N_Fred_Industrial Production Manufacturing Non-Durable Goods Ice Cream and Frozen Dessert.csv')
dicfn.DATE = pd.to_datetime(dicfn.DATE)
plt.plot(dicfn.DATE[dicfn.DATE.dt.year > 2015], dicfn.IPN31152N[dicfn.DATE.dt.year > 2015])
plt.title('Ice Cream Production Time Series')
plt.ylabel('Production Index')
plt.xlabel('Time t')
plt.legend(['Icream Daily Production'])
plt.grid(True)
plt.tight_layout()
plt.savefig('IceCream_TimeSeries.png')
plt.show()

t_date = dicfn.DATE[dicfn.DATE.dt.year > 2015]
x = dicfn.IPN31152N[dicfn.DATE.dt.year > 2015].values

plt.figure(figsize=(10, 7))
fig, ax = plt.subplots()
ax.acorr(x, maxlags=30)
plt.title('Ice Cream Production Autocorrelation')
minor_locator = AutoMinorLocator(10)
ax.xaxis.set_minor_locator(minor_locator)
plt.grid(which='minor')
plt.legend(['$R_X$(\u03C4)'], loc='upper left')
plt.xlabel('\u03C4')
ax.grid(True, which='both')
plt.tight_layout()
fig.savefig('Stochastic_Autocorrelation.png')
fig.show()

# PSD
freqs, psd = signal.welch(x)
plt.figure(figsize=(10, 7))
fig, ax = plt.subplots()
ax.plot(freqs, psd)
plt.title('Ice Cream Production Power Spectral Density')
minor_locator = AutoMinorLocator(5)
ax.xaxis.set_minor_locator(minor_locator)
plt.grid(which='minor')
plt.legend(['PSD'])
plt.xlabel('Frequency')
plt.ylabel('Power')
ax.grid(True, which='both')
plt.tight_layout()
fig.savefig('Stochastic_PSD.png')
fig.show()


# ============================================================================
# Viscosity plots
# ============================================================================
print("\nGenerating Viscosity plots...")
create_plots_no_date(dv, 'Viscosity Hourly', 'Viscosity', ylabel='Viscosity')


# ============================================================================
# WholeFood plots
# ============================================================================
print("\nGenerating WholeFood plots...")
create_plots_no_date(dw, 'Weekly Sales', 'WholeFood', ylabel='Weekly Sales')


# ============================================================================
# Pharmaceutical plots
# ============================================================================
print("\nGenerating Pharmaceutical plots...")
create_plots_no_date(dp, 'Weekly_Sales', 'Pharmaceutical', ylabel='Weekly Sales')


print("\nAll plots generated successfully!")