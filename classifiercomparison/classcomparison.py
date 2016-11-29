import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import linear_model, cross_validation, svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

h = 0.02 #step size in mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

#bring in data
df = pd.read_csv('data.txt')

#as always, X is features, y is label
X = np.array(df.drop(['class'],1))
y = np.array(df['class'])

classifiers = [
    KNeighborsClassifier(3),
    KNeighborsClassifier(),
    BaggingClassifier(),
    GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls'),
    svm.SVR(),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    linear_model.SGDClassifier(),
    linear_model.LogisticRegression(),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

for clf in classifiers:
	X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4)

	clf.fit(X_train, y_train)
	confidence = clf.score(X_test, y_test)
	example_measures = np.array([3,4,3,2,3]) #should be class [1]
	example_measures = example_measures.reshape(1,-1)
	prediction = clf.predict(example_measures)
	print("Confidence score for {} is {} with a prediction of {}".format(clf, confidence, prediction))
