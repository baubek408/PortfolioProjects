#!/usr/bin/env python
# coding: utf-8

# In[80]:


import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib 
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (12,8) # Adjust a configuration of the plots we will create

# Read in the data
df = pd.read_csv('/Users/mynbayevbaubek/Downloads/movies.csv')


# In[7]:


# Let's a look at the data

df.head()


# In[12]:


# Let's see if there is a any missing data

for col in df.columns:
    pcnt_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, pcnt_missing))


# In[18]:


# Data types of columns

df.dtypes


# In[81]:


# Remove rows with Nan values in any columns

df = df.dropna(how='any') 


# In[82]:


# change a dtype of some columns from float to int

df['budget'] = df['budget'].apply(np.int64)
df['votes'] = df['votes'].apply(np.int64)
df['gross'] = df['gross'].apply(np.int64)
df['runtime'] = df['runtime'].apply(np.int64)


# In[26]:


df.head()


# In[83]:


#Split a column 'released' by comma to create new column 'released_year' and fill it only year of release
df['released_year'] = df['released'].str.split(',').str[1]
df['released_year'] = df['released_year'].astype(str).str[:5]


# In[84]:


# Remove spaces before and after the string in column 'released_year'
df['released_year'] = df['released_year'].str.strip()


# In[85]:


# Rename column 'released_year' to 'yearcorrect'
df = df.rename({'released_year': 'yearcorrect'}, axis=1)


# In[53]:


df.head()


# In[57]:


# Checing the release date correction because of two simmilar info columns named 'released' and 'year'
pd.set_option('display.max_rows', None) 


# In[101]:


# Drop duplicates

df.drop_duplicates().head()


# In[66]:


# Budget high correlation
# company high correlation

# Scatter plot with budget vs gross

plt.scatter(x=df['budget'], y=df['gross'])

plt.title("Budget vs Gross Earnings")
plt.xlabel("Budget for Film")
plt.ylabel("Gross")
plt.show()


# In[102]:


df.sort_values(by=['gross'], inplace=False, ascending=False).head()


# In[69]:


# Plot budget vs gross using seaborn

sns.regplot(x='budget', y='gross', data=df, scatter_kws={'color':'red'}, line_kws={'color':'blue'})


# In[73]:


# Looking at correlation

df.corr(method='pearson') #pearson, kendall, spearman


# In[75]:


# High corrrelation between Budget and Gross as we assumed

correlation_matrix = df.corr(method='pearson')

sns.heatmap(correlation_matrix, annot=True)
plt.title("Correlation Matrix for Numeric Features")
plt.xlabel("Movie Features")
plt.ylabel("Movie Features")
plt.show


# In[87]:


# Looks at company column
df = df.sort_values(by=['gross'], inplace=False, ascending=False)
df.head()


# In[103]:


# Create a new df(df_numarized)
df_numarized = df
for col_names in df_numarized.columns:
    if(df_numarized[col_names].dtype == 'object'):
        df_numarized[col_names] = df_numarized[col_names].astype('category')
        df_numarized[col_names] = df_numarized[col_names].cat.codes
        
df_numarized.head()


# In[88]:


df.head()


# In[89]:


df_numarized.head()


# In[90]:


# Compare other features for valuable correlation
# Crete Correlation matrix for other features changed for numeric by object types
correlation_matrix = df_numarized.corr(method='pearson')

sns.heatmap(correlation_matrix, annot=True)
plt.title("Correlation Matrix for Numeric Features")
plt.xlabel("Movie Features")
plt.ylabel("Movie Features")
plt.show


# In[93]:


# Another way to look at corr
correlation_mat = df_numarized.corr()
corr_pairs = correlation_mat.unstack()
sorted_pairs = corr_pairs.sort_values()
sorted_pairs


# In[97]:


# Sorted corr pairs of features
high_corr = sorted_pairs[(sorted_pairs) > 0.5] 
high_corr


# In[99]:


# Votes and Budget have a highest correlation to gross earnings
# Column 'company' has low correlation than we thought


# In[ ]:




