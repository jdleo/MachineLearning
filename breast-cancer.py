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

clf = neighbors.KNeighborsClassifier()
clf.fit(x_train,y_train)

confidence = clf.score(x_test,y_test)
#print("Confidence score: {}".format(confidence))

print("Enter your patients tumor classifications")
print("Score 1-10. Clump thickness, Uniformity of Cell Size, Uniformity of Cell Shape,  Marginal Adhesion,")
print("Single Epithelial Cell Size, Bare Nuclei count, Bland Chromatin count, Normal Nucleoli count, Mitoses")

#user input
values = str(raw_input("Enter 1-10 for each above back-to-back. Example 123221234:\n"))

arr = []

for i in values:
  arr.append(int(i))

prediction_measures = np.array(arr)
prediction_measures = prediction_measures.reshape(1,-1)

prediction = clf.predict(prediction_measures)

#decision
if prediction == [2]:
	print("The tumor is benign with a confidence score of {}".format(confidence))
elif prediction == [4]:
	print("The tumor is malignant with a confidence score of {}".format(confidence))



