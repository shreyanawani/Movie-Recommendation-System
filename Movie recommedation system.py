#!/usr/bin/env python
# coding: utf-8

# In[62]:


import numpy as np
import pandas as pd
import warnings


# In[63]:


warnings.filterwarnings('ignore')


# In[64]:


columns=['user_id','item-id','rating','timestamp']

df=pd.read_csv(r"C:\Users\Shreya\Downloads\ml-100k\u.data",sep='\t',names=columns)
#df.head()


# In[65]:


df['item-id'].nunique()


# In[66]:


movies_title=pd.read_csv(r"C:\Users\Shreya\Downloads\ml-100k\u.item",sep='\|',header=None)


# In[67]:


#movies_title.head()


# In[68]:


movies_titles=movies_title[[0,1]]


# In[69]:


movies_titles.columns=["item-id",'title']


# In[70]:


#movies_titles.head()


# In[71]:


df=pd.merge(df,movies_titles,on='item-id')


# In[72]:


#df.tail()


# In[228]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[229]:


ratings=pd.DataFrame(df.groupby('title').mean()['rating'])


# In[231]:


ratings['no of ratings']=pd.DataFrame(df.groupby('title').count()['rating'])


# In[232]:


#ratings.head()


# In[233]:


ratings.sort_values(by='rating',ascending=False)


# In[234]:


plt.figure(figsize=(10,6))
plt.hist(ratings['no of ratings'],bins=70)
plt.show()


# In[235]:


plt.hist(ratings['rating'],bins=70)
plt.show()


# In[236]:


sns.jointplot(x='rating',y='no of ratings',data=ratings,alpha=0.5)


# # Creating project

# In[237]:


df.head()


# In[238]:


moviemat=df.pivot_table(index='user_id',columns='title',values='rating')
moviemat


# In[239]:


ratings.sort_values('no of ratings',ascending=False).head()


# In[240]:


starwars_user_rating=moviemat['Star Wars (1977)']
starwars_user_rating.head()


# In[241]:


similar=moviemat.corrwith(starwars_user_rating)
similar


# In[242]:


corr_starwars=pd.DataFrame(similar,columns=['Correlation'])
corr_starwars


# In[247]:


corr_starwars.dropna(inplace=True)


# In[248]:


corr_starwars.head()
corr_starwars.sort_values('Correlation',ascending=False).head(10)


# In[249]:


ratings


# In[250]:


corr_starwars=corr_starwars.join(ratings['no of ratings'])
corr_starwars.head()


# In[251]:


corr_starwars[corr_starwars['no of ratings']>100].sort_values('Correlation',ascending=False)


# # FUNCTION

# In[225]:


def predict_movies(movie_name):
    movie_user_ratings=moviemat[movie_name]
    similar_to_movie=moviemat.corrwith(movie_user_ratings)
    
    corr_movie=pd.DataFrame(similar_to_movie,columns=['Correlation'])
    corr_movie.dropna(inplace=True)
    
    corr_movie=corr_movie.join(ratings['no of ratings'])
    predictions=corr_movie[corr_movie['no of ratings']>100].sort_values('Correlation',ascending=False)
    
    return predictions
predicitons=predict_movies("Titanic (1997)")
predicitons.head()
    

