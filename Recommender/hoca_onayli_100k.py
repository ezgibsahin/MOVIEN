#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


item = pd.read_csv("u.item", sep="|",encoding="latin-1", 
                      names=["movie_id", "movie_title", "release_date", "video_release_date",
                             "imbd_url", "unknown", "action", "adventure", "animation",
                             "childrens", "comedy", "crime", "documentary", "drama", "fantasy", 
                             "film_noir", "horror", "musical", "mystery", "romance", 
                             "sci-fi", "thriller", "war", "western"])
item


# In[2]:


columnsToDelete = ['video_release_date']
item.drop(columnsToDelete, inplace = True, axis=1)


# In[3]:


pd.isnull(item).sum()


# In[4]:


item = item.dropna()
item


# In[5]:


item = item.reset_index()
item


# In[6]:


item[['day','month','year']] = item.release_date.str.split("-",expand=True,)


# In[7]:


item


# In[8]:


columnsToDelete = ['index','release_date','day','month']
item.drop(columnsToDelete, inplace = True, axis=1)

# item.groupby((item.index.year//10)*10).sum()


# In[9]:


item


# In[10]:


s = item['year']


# In[11]:


s


# In[12]:


decade=[]
np.zeros(decade)
for val in s: 
    valToadd = int(val)%1920
    if(valToadd<=79 and valToadd>=70):
        decade.append('90s')
    elif(valToadd<=69 and valToadd>=60):
        decade.append('80s')
    elif(valToadd<=59 and valToadd>=50):
        decade.append('70s')
    elif(valToadd<=49 and valToadd>=40):
        decade.append('60s')
    elif(valToadd<=39 and valToadd>=30):
        decade.append('50s')   
    elif(valToadd<=29 and valToadd>=20):
        decade.append('40s')
    elif(valToadd<=19 and valToadd>=10):
        decade.append('30s')
    elif(valToadd<=9):
        decade.append('20s')    
    


# In[13]:


decade


# In[14]:


item['decade'] = decade


# In[15]:


item


# In[16]:


item.drop(['year'], inplace = True, axis=1)


# In[17]:


item


# In[18]:


#u.item done
#starting u.user 
user = pd.read_csv("u.user", sep="|", encoding="latin-1", names=["user_id", "age", "gender","occupation", "zip_code"])


# In[19]:


user


# In[ ]:





# In[ ]:





# In[ ]:





# In[20]:


columnsToDelete = ['zip_code']
user.drop(columnsToDelete, inplace = True, axis=1)


# In[21]:


user.age.unique()


# In[22]:


pd.isnull(user).sum()


# In[23]:


s = user['age']
generation=[]
np.zeros(generation)
for val in s:
    if(val<13):
        generation.append('kids')
    elif(val<=19 and val>=13):
        generation.append('teens')
    elif(val<=29 and val>=20):
        generation.append('20-29')
    elif(val<=39 and val>=30):
        generation.append('30-39')
    elif(val<=49 and val>=40):
        generation.append('40-49')
    elif(val<=59 and val>=50):
        generation.append('50-59')
    elif(val<=69 and val>=60):
        generation.append('60-69')
    elif(val<=79 and val>=70):
        generation.append('70-79')
    else:
        generation.append('80&older')
    


# In[24]:


user['generation'] = generation


# In[25]:


user


# In[26]:


#u.user done
#u.data
data = pd.read_csv("u.data")


# In[27]:


data


# In[28]:


data.rating.hist()
plt.title("Rating Distribution", fontsize=16)
plt.ylabel("Count", fontsize=14)
plt.xlabel("Rating", fontsize=14)


# In[ ]:





# In[29]:


import time
data['timestamp'] = data['timestamp'].apply(lambda x: time.strftime('%Y', time.localtime(x)))


# In[30]:


data


# In[31]:


data['timestamp'].unique()


# In[32]:


data.drop('timestamp', inplace = True, axis=1)


# In[33]:


data


# In[34]:


data['rating'].unique()


# In[35]:


rating_values = data['rating']


# In[36]:


data.groupby(['rating']).sum().plot(kind='pie', y='user_id')


# In[37]:


data.groupby(['rating']).sum().plot(kind='pie', subplots=True, shadow = True,startangle=90,
figsize=(15,10), autopct='%1.1f%%')


# In[38]:


#u.data done
#other tables are info
#merging tables
item['unknown']


# In[39]:


item['unknown'].value_counts()


# In[40]:


item = item[item.unknown == 0]


# In[41]:


item


# In[42]:


item['unknown'].value_counts()


# In[43]:


columnsToDelete = ['unknown']
item.drop(columnsToDelete, inplace = True, axis=1)


# In[44]:


item


# In[45]:


user


# In[46]:


data


# In[47]:


full_df = pd.merge(user, data, how="left", on="user_id")
full_df = pd.merge(full_df, item, how="left", right_on="movie_id", left_on="item_id")


# In[48]:


full_df


# In[49]:


top_ten_movies = full_df.groupby("movie_title").size().sort_values(ascending=False)[:10]
# plot the counts
plt.figure(figsize=(12, 5))
plt.barh(y= top_ten_movies.index,width= top_ten_movies.values)
plt.title("10 Most Rated Movies in the Data", fontsize=16)
plt.ylabel("Moive", fontsize=14)
plt.xlabel("Count", fontsize=14)
plt.show()


# In[50]:


least_10_movies = full_df.groupby("movie_title").size().sort_values(ascending=False)[-10:]
# plot the counts
plt.figure(figsize=(12, 5))
plt.barh(y= least_10_movies.index,width= least_10_movies.values)
plt.title("10 Least Rated Movies in the Data", fontsize=16)
plt.ylabel("Moive", fontsize=14)
plt.xlabel("Count", fontsize=14)
plt.show()


# In[51]:


genres= ["action", "adventure", "animation", "childrens", "comedy", "crime", "documentary", "drama", "fantasy", "film_noir", "horror", "musical", "mystery", "romance", "sci-fi", "thriller", "war", "western"]
full_df[full_df.movie_title == "Star Wars (1977)"][genres].iloc[0].sort_values(ascending=False)


# In[52]:


gender_counts = user.gender.value_counts()

# plot the counts 
plt.figure(figsize=(12, 5))
plt.bar(x= gender_counts.index[0], height=gender_counts.values[0], color="blue")
plt.bar(x= gender_counts.index[1], height=gender_counts.values[1], color="orange")
plt.title("Number of Male and Female Participants", fontsize=16)
plt.xlabel("Gender", fontsize=14)
plt.ylabel("Counts", fontsize=14)
plt.show()


# In[53]:


user.groupby(['gender']).sum().plot(kind='pie', y='user_id',autopct='%1.1f%%')


# In[54]:


full_df[genres+["gender"]].groupby("gender").sum().T.plot(kind="barh",  figsize=(15,12), color=["violet", "royalblue"],width = 0.70 )
plt.xlabel("Counts",fontsize=14)
plt.ylabel("Genre", fontsize=14)
plt.title("Popular Genres Among Genders", fontsize=16)
plt.show()


# In[55]:


full_df[genres+["generation"]].groupby("generation").sum().T.plot(kind="barh", stacked=True, figsize=(30,15), color=["limegreen","cornflowerblue","cyan","gold","yellow","orange","maroon","violet"] ,width = 0.70 )
plt.xlabel("Counts",fontsize=20)
plt.ylabel("Genre", fontsize=20)
plt.title("Popular Genres Among Generations", fontsize=25)
plt.legend(title="Generations", title_fontsize=30,prop={'size': 25})
plt.show()


# In[56]:


user.groupby(['occupation']).sum().plot(kind='pie', y='user_id',autopct='%1.1f%%',figsize=(20, 20))


# In[57]:


ys = [i+21+(i*21)**2 for i in range(21)]
user['occupation'].value_counts().plot.bar(title = "Occupation",figsize=(15, 8),color = cm.rainbow(np.linspace(0, 1, len(ys))) )


# In[58]:





ys = [i+8+(i*8)**2 for i in range(8)]
user['generation'].value_counts().plot.bar(title = "Generation",figsize=(15, 8),color = cm.rainbow(np.linspace(0, 1, len(ys))))



# In[59]:


user['generation'].value_counts().plot.pie(title = "Generation",figsize=(10, 10),autopct='%1.1f%%',fontsize=10)


# In[60]:



item['decade'].value_counts().plot.pie(title = "Decades",figsize=(15, 15),autopct='%1.1f%%',fontsize=10,colors=["darkorchid","cornflowerblue","cyan","lime","gold","orange","red","navy"])




# In[61]:


full_df


# In[62]:


full_df.to_csv(r'full_df.csv', index = False, header = True)

