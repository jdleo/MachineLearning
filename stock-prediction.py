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
print("###############################\nHow far do you want to forecast out? (must be a float value) (recommended setting: 0.01)")
out = float(raw_input("Example: If a stock has been public for 3000 days, 0.01 will forecast 30 days out: "))

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

df['label'] = df[forecast_column].shift(-forecast_out)

#X for features, y for label. in this case, features will be all data besides our label
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

#drop nan
df.dropna(inplace=True)

#y = label column
y = np.array(df['label'])

#train/test using cross validation with a test size of 0.2
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

#using the Linear Regression classifier. might test with other settings later, but right now were using default params
clf = LinearRegression()
clf.fit(X_train, y_train)

#confidence score
confidence = clf.score(X_test, y_test)
print("Confidence score for this stock prediction: {}".format(confidence))

forecast_set = clf.predict(X_lately)
#print(forecast_set)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.to_datetime()
one_day = 86400
next_unix = time.mktime(last_unix.timetuple()) + one_day

#just making a date for each close price
for i in forecast_set:
  next_date = datetime.datetime.fromtimestamp(next_unix)
  next_unix += one_day
  df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]


#PLOT EVERYTHING!
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

