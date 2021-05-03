import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
data=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
X=data.iloc[:, :-1].values
y=data.iloc[:, 4].values
from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
print("The training data is\n",X_train)
print("The testing data is\n",X_test)
print("the expected result is\n",y_test)

from sklearn.svm import SVC
clf=SVC()


clf=clf.fit(X_train,y_train)


prediction=clf.predict(X_test)
print("The prediction by the machine learning model  is\n",prediction)
from sklearn.metrics import accuracy_score

a=accuracy_score(y_test,prediction)
print("The accuracy of this model is: ", a)









