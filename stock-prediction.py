'''
Pandas for organizing our datasets, Quandl because it is has the best collection of data sets imo,
math for obvious reasons, datetime/time for handling timestamps, sklearn for machine learning,
matplotlib for plotting our data on a nice graph, and style for, well, style ;)
'''
import pandas as pd
import quandl, math, datetime, time
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

#user enters stock symbol, and we format it for quandl
stockSymbol = raw_input("Enter a stock symbol you want to predict for (Ex: AAPL): ")
qCode = 'WIKI/{}'.format(stockSymbol)

#how far do we want to forecast out?
print("How far do you want to forecast out? (must be a float value)")
out = float(raw_input("Example: If a stock has been public for 300 days, 0.1 will forecast 30 days out"))

df = quandl.get(qCode)

#keep relevant columns for testing
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

#making new column for High/Low percent which is essentially just the day-to-day pct change
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0

#variable for re-use
forecast_column = 'Adj. Close'

#fill nandata making it a ridiculous outlier
df.fillna(-99999, inplace=True)

#using our variable we got from user for forecast_out
forecast_out = int(math.ceil(out*len(df)))
print("Forecast out: {} days".format(forecast_out))
