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
stockSymbol = input("Enter a stock symbol you want to predict for (Ex: AAPL): ")
qCode = 'WIKI/{}'.format(stockSymbol)

df = qandl.get(qCode)

#keep relevant columns for testing
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]


