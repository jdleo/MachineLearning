# Machine Learning
In this repository, I will be uploading all things ML-related. This is mostly to showcase some useful AI programs, but also for my own practice. I will be posting programs and deeplearning-related research.  
  
  
  
## Stock Predictions
The first program. This program uses linear regression and cross validation to train/test/predict what a stock will be in the future. Whether you believe in stock pattern-recognition or not, this program has surprising accuracy.  
### Required Frameworks
pandas  
quandl  
sklearn  
matplotlib 
```
sudo pip install library_name
```  
### Usage  
download/copy stock-prediction.py to your computer  
```
python stock-prediction.py
```  
Then enter a forecast_out float. Ex: If a stock has been public for 2700 days, 0.01 will forecast 27 days out.  
It will pop open a graph. The blue lines are historic stocks, green lines are prediction graph.  
You can pan and zoom on matplotlib as you wish.  
  
  
  
## Breast Cancer Diagnosis  
My second ML program. This program is very short and simple. It uses a historic breast cancer data set, with a KNeighbors classifier, trains, tests, and predicts if your patient has a benign or malignant tumor with 96+% confidence. Obviously the 4% error wouldn't be very comforting as a patient, but surprisingly, this works.  
### Required Frameworks  
numpy  
sklearn  
pandas  
```  
sudo pip install library_name  
```  
Data Source:  
> Holberg, William H., Dr. "Breast Cancer Wisconsin (Diagnosis) Data Set." UCI Machine Learning Repository. November 1, 1995. https://archive.ics.uci.edu/ml/datasets/Breast Cancer Wisconsin (Diagnostic).  
  
### Usage  
download/copy breast-cancer.py AND breastcancerdata.txt in the same dir  
```
python breast-cancer.py  
```  
Then enter the tumor classifiers back-to-back. There should be 9 digits scoring 1-10 each.
For example: You can enter 123222343  

