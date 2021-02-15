#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, normalize
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[2]:


full_df= pd.read_csv("full_df.csv")
full_df


# In[3]:


columnsToDelete = ['generation','decade']
full_df.drop(columnsToDelete, inplace = True, axis=1)
full_df


# In[4]:


x = full_df[['age']].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
full_df['age'] = pd.DataFrame(x_scaled)
full_df


# In[5]:


onehotencoder = preprocessing.OneHotEncoder()
full_df['occupation'] = pd.Categorical(full_df['occupation'])
dfDummies = pd.get_dummies(full_df['occupation'])
dfDummies


# In[6]:


onehotencoder = preprocessing.OneHotEncoder()
full_df['gender'] = pd.Categorical(full_df['gender'])
dfDummies2 = pd.get_dummies(full_df['gender'])
dfDummies2


# In[7]:


df_new = pd.concat([full_df, dfDummies, dfDummies2], axis=1)
df_new


# In[8]:


columnsToDelete = ['gender','occupation']
df_new.drop(columnsToDelete, inplace = True, axis=1)
df_new


# In[9]:


df_new.loc[full_df['rating']>=3, 'target']=1

df_new.loc[full_df['rating']<3, 'target']=0
df_new


# In[10]:


df_new.drop('rating', inplace = True, axis=1)
df_new


# In[11]:


columnsToDelete = ['movie_title','imbd_url']
df_new.drop(columnsToDelete, inplace = True, axis=1)
df_new


# In[12]:


df_new.fillna(df_new.mean(), inplace=True)


# In[13]:




X = df_new.drop('target',  axis=1)
y = df_new['target'] 
#X = df_new.iloc[:,:'target']
#y = df_new.iloc[:,'target']

#y = dataframeCsvFile.iloc[:,14]#Target attribute is the 15th column attribute >50K or <=50K
#X = dataframeCsvFile.iloc[:,:14]


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.10, random_state=1)


# In[15]:


knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)


# In[16]:


print("Precision : ", metrics.precision_score(y_test, y_pred, average='weighted'))
print("Accuracy : ", metrics.accuracy_score(y_test, y_pred))
print("Recall : ", metrics.recall_score(y_test, y_pred, average='macro'))
print("F1 Score : ", metrics.f1_score(y_test, y_pred, average='weighted'))


# In[ ]:





# In[ ]:




