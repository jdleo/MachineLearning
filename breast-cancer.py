import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors

#bring in breast cancer data
df = pd.read_csv('breastcancerdata.txt')

#replace question marks in data set with outlier values. It's irresponsible to dump nan data,
#but the outlier is so significant that our algorithm will ignore it
df.replace('?', -99999, inplace=True)

#id column isn't relevant for testing
df.drop(['id'], 1, inplace=True)

#x is features, y is class
x = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

#train/test
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y,test_size=0.2)

