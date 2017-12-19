This example uses the public dataset -- House Sales in King County, USA to demonstrate regression techniques. 
Details about this dataset can be found in https://www.kaggle.com/harlfoxem/housesalesprediction/kernels

We setup a experiment protocol with 5-fold cross-validation by splitting all examples even into 5 sets and hold one out as testing and the rest as training. Simple algorithms including linear regression (with L2 regularization, aka. ridge regression) and Lasso regression are experimented. Performance are reported as RMSE and MAE.