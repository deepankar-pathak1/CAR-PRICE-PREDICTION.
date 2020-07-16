#!/usr/bin/env python
# coding: utf-8

# This dataset contains information about used cars listed on www.cardekho.com
# This data can be used for a lot of purposes such as price prediction to exemplify the use of linear regression in Machine Learning.
# The columns in the given dataset is as follows:
# 
# 1)Car_Name
# 2)Year
# 3)Selling_Price
# 4)Present_Price
# 5)Kms_Driven
# 6)Fuel_Type
# 7)Seller_Type
# 8)Transmission
# 9)Owner
# 
# Here we going to find out "CAR-PRICE".

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("datasets_33080_1320127_car data.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[22]:


print(df["Seller_Type"].unique())
print(df["Transmission"].unique())
print(df["Owner"].unique())
print(df["Fuel_Type"].unique())


# In[6]:


## Check missing value and null value
df.isnull().sum()


# In[7]:


df.describe()


# In[8]:


df.columns


# In[9]:


final_dataset = df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]


# In[10]:


final_dataset.head()


# In[11]:


final_dataset["Current_Year"] = 2020


# In[12]:


final_dataset.head()


# In[16]:


final_dataset["no_year"] = final_dataset["Current_Year"] - final_dataset["Year"]


# In[17]:


final_dataset.head()


# In[18]:


final_dataset.drop(["Year"], axis=1, inplace = True)


# In[19]:


final_dataset.head()


# In[20]:


final_dataset.drop(["Current_Year"], axis=1, inplace = True)


# In[21]:


final_dataset.head()


# In[23]:


## Now convert categorical feautres.
final_dataset = pd.get_dummies(final_dataset, drop_first = True)


# In[24]:


final_dataset.head()


# In[25]:


# How can we find out correlation
final_dataset.corr()


# In[26]:


import seaborn as sns


# In[27]:


# to visualize correlation 
sns.pairplot(final_dataset)


# In[29]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[31]:


# heat map
corrmat = final_dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
# plot the heat map
g = sns.heatmap(final_dataset[top_corr_features].corr(),annot = True, cmap = "RdYlGn")


# In[32]:


final_dataset.head()


# In[33]:


# independent and dependent features
x = final_dataset.iloc[:,1:] #independent
y = final_dataset.iloc[:,0] # dependent


# In[34]:


x.head()


# In[35]:


y.head()


# In[36]:


## Feature importance
from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(x,y)


# In[37]:


print(model.feature_importances_)


# In[40]:


# Plot feature of importance for better visulaiztion
feat_importance = pd.Series(model.feature_importances_, index = x.columns)
feat_importance.nlargest(5).plot(kind = "barh")
plt.show()


# In[42]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state = 12)


# In[45]:


X_train.shape, X_test.shape


# In[46]:


from sklearn.ensemble import RandomForestRegressor
rf_random = RandomForestRegressor()


# In[49]:


## Randomized search cv
## Hyperparameter
import numpy as np
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
print(n_estimators)
# Number of features to consider at every split
max_features = ["auto", "sqrt"]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5,30, num = (6))]
#max_depth.append(None)
# Minimum number of samples required to split a node.
min_samples_split = [2,5,10,15,100]
# Minimum number of samples required at each leaf.
min_samples_leaf = [1,2,5,10]


# In[52]:


# used for finding best parameters
from sklearn.model_selection import RandomizedSearchCV


# In[53]:


# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)


# In[54]:


# Use random grid for best hyperparameters
# First create the base model to tune.
rf = RandomForestRegressor()


# In[56]:


# Random search of parameters using 3 folds cross validation.
# Search across 100 different combinations.
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)


# In[57]:


rf_random.fit(X_train,Y_train)


# In[58]:


prediction = rf_random.predict(X_test)


# In[59]:


prediction


# In[62]:


sns.distplot(Y_test-prediction)
# as here we are getting minmum value therefore graph is like closed gaussion distribution.


# In[64]:


plt.scatter(Y_test, prediction);


# In[65]:


import pickle 
# open a file where you can store the data.
file = open("radom_forest_regression_model.pk1", "wb")

# dump information into the file
pickle.dump(rf_random, file)


# In[ ]:




