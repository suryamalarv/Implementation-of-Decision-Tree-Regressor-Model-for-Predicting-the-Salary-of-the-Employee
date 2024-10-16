# EXPERIMENT NO: 9
# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.
2.Upload the dataset and check for any null values using .isnull() function.
3.Import LabelEncoder and encode the dataset.
4.Import DecisionTreeRegressor from sklearn and apply the model on the dataset.
5.Predict the values of arrays.
6.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.
7.Predict the values of array.
8.Apply to new unknown values.
## Program & Output:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SURYAMALARV
RegisterNumber:  212223230224
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
data = pd.read_csv("/content/Salary (2).csv")
data.head()
```
![image](https://github.com/user-attachments/assets/52308b43-0c7d-4370-af33-9ba24831850f)
```
data.info()
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/4b379c8c-d99f-46e1-8802-55b29b3ceb06)
```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
```
![image](https://github.com/user-attachments/assets/3ef7b928-3541-459e-9ab7-501cdc4c734d)
```
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor,plot_tree
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse
```
![image](https://github.com/user-attachments/assets/e0de1eb1-fa3e-4aaf-9643-08d5a2d3c442)
```
r2=metrics.r2_score(y_test,y_pred)
r2
```
![image](https://github.com/user-attachments/assets/1c4b154c-4298-4294-bf65-8a5c557d2327)
```
dt.predict([[5,6]])
```
![image](https://github.com/user-attachments/assets/846066f2-4a14-47c1-9342-81d37a6d3016)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
