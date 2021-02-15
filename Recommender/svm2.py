#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, normalize

from pandas.plotting import scatter_matrix
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score


# In[2]:


df= pd.read_csv("my_final_dataset.csv")
X = df.drop('target',  axis=1)
y = df['target'] 


# In[3]:


X_train, X_test, y_train, y_test, = train_test_split(X,y,test_size=.10, random_state=1)
y_expect = y_test;

def Test10(type, name):
    type.fit(X_train,y_train)
    print("%10 Test",name);
    y_pred = type.predict(X_test);
    print("Precision : ", precision_score(y_expect, y_pred, average='weighted'))
    print("Accuracy : ", accuracy_score(y_expect, y_pred));
    print("Recall : ", recall_score(y_expect, y_pred, average='macro'))
    print("F1 Score : ", f1_score(y_expect, y_pred, average='weighted'))
    print(confusion_matrix(y_expect, y_pred))

def Cross(type,name):
    print("Cross Validation ",name)
    y_pred = cross_val_predict(type, X, y, cv=10)
    print("Precision : ", precision_score(y, y_pred, average='weighted'))
    print("Accuracy : ", accuracy_score(y, y_pred));
    print("Recall : ", recall_score(y, y_pred, average='macro'))
    print("F1 Score : ", f1_score(y, y_pred, average='weighted'))
    print(confusion_matrix(y, y_pred))


# In[ ]:


#vectormachine = svm.SVC(kernel='sigmoid')
#vectormachine = svm.SVC(kernel='rbf')
#vectormachine = svm.SVC(kernel='poly', degree=8)
vectormachine = svm.SVC(gamma= 'scale', kernel='linear')
Test10(vectormachine,"SVM");
Cross(vectormachine,"SVM")


# In[ ]:




