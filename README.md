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
