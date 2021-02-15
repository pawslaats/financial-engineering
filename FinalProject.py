#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import scipy.fftpack
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.signal
from scipy.fftpack import fft

# Data from Prof. Evans
data_df = pd.read_csv('SPY7_25_16_8_2_16.txt', sep=",")
print(data_df.columns)
print data_df
AdjClose = data_df[u'Close']

# Data From .CSV file
#data_df = pd.read_csv('AAPL.csv')
#print(data_df.columns)
#AdjClose = data_df['AdjClose']

# Detrending Stock Data 
# AdjCloseTrend = scipy.signal.detrend(AdjClose)

# Detrending Logrithmic Growth Rate
N = len(AdjClose)
Nln = N - 1
AdjCloseLn = np.zeros(Nln)
for i in range(0,Nln):
    AdjCloseLn[i] = math.log(AdjClose[i+1]/AdjClose[i])
    
AdjCloseTrendLn = scipy.signal.detrend(AdjCloseLn)

# Variables
T = 1.0 / Nln
x = np.linspace(0.0, N*T, N)
xln = np.linspace(0.0, Nln*T, Nln)
yf = fft(AdjCloseLn)
xf = np.linspace(0.0, 1.0/(2.0*T), Nln//2)
youtput = 2.0/Nln * np.abs(yf[0:Nln//2])
# Define the Minimum and Maximum Frequencies to be observed on the FFT
freqmin = 4;
freqmax = 200;

# FFT Analysis
yf = fft(AdjCloseTrendLn)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
youtput = 2.0/N * np.abs(yf[0:N//2])


# Adjusted Close Data
plt.plot(xln, AdjCloseLn)
plt.grid()
plt.show()

max = np.max(AdjCloseLn)
avg = np.average(AdjCloseLn)
print avg
print max

# Adjusted Close Detrended Data
plt.plot(xln, AdjCloseTrendLn)
plt.grid()
plt.show()

max = np.max(AdjCloseTrendLn)
avg = np.average(AdjCloseTrendLn)
print avg
print max

# Plot FFT
plt.plot(xf, youtput)
plt.grid()
plt.show()

max = np.max(youtput)
avg = np.average(youtput)
print avg
print max

# Plot FFT
plt.plot(xf[freqmin:freqmax], youtput[freqmin:freqmax])
plt.grid()
plt.show()

max = np.max(youtput[freqmin:freqmax])
avg = np.average(youtput[freqmin:freqmax])
print avg
print max