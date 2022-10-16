#!/usr/bin/env python
# coding: utf-8

# #### OVERVIEW
# 
# we're trying to decide whether or not companies should focus their efforts and capital more on either one of TV, radio, their newspaper ads to increase sale

# import the neccessary library

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Import the dataset

# In[2]:


df = pd.read_csv("Advertising.csv")


# In[3]:


#view the first 5 rows in the dataset
df.head()


# In[4]:


#Get the shape of the dataset
df.shape


# The dataset has 200 rows and 4 columns

# In[5]:


#Get the information of the data
df.info()


# All the columns have same datatypes which is float.

# In[6]:


#Decribe the data
df.describe()


# In[10]:


df.columns


# In[11]:


#Check  for missing values
df.isnull().sum()


# There are no null values in the data

# In[17]:


#Check for duplicate values
df.duplicated().sum()


# There are no duplicated values in the data

# #### Exploratory Data Analysis

# In[18]:


#Check for ouliers
for i in df:
    plt.figure(figsize = (4,1))
    sns.boxplot(x = df[i])
    plt.show()


# In[22]:


#Create joitplot to compare the TV and Sales columns
sns.jointplot(x='TV',y='sales', data=df)


# In[25]:


sns.jointplot(x='radio',y='sales', data=df)


# In[28]:


sns.jointplot(x='newspaper',y='sales', data=df)


# #### Lets explore these features  and relationships across the entire dataset using pairplot

# In[29]:


sns.pairplot(df)


# #### Plotting a univariate distribution of sale made

# In[35]:


sns.distplot(df['sales'], bins=50, kde = True)


# In[40]:


corr = df.corr()


# In[43]:


plt.figure(figsize = (10,7))
sns.heatmap(corr, cmap='Spectral', linewidths= 0.2, annot=True)


# Comment: With the above plots, the most correlated feature with sale is TV

# In[44]:


#Creating a linear model plot (using seaborn's lmplot method) of Sale vs TV
sns.lmplot(x='TV', y='sales', data=df)


# ### Model

# #### Dependent and Independent variables

# Set a variable X equal to the amounts spent on different adverts (features) and a variable y equal to the 'sale' column.

# In[46]:


X = df.iloc[:,0:3]   #independent variables
y = df['sales'] #target variable


# In[48]:


# Let's look at the correlation between each variables
X.iloc[:,0:].corr()


# Comment: No exceptionally high correlation is observed, so we wont be dropping any columns

# #### Viewing the OLS model
# 

# import the api for statistical models

# In[50]:


import statsmodels.api as sm


# In[51]:


X_sm = sm.add_constant(X)
ols_model = sm.OLS(y, X_sm).fit()
ols_model.summary()


# ### observations: 

# A good R-squared value is observed.

# The standard error values are low which indicates the absence of a multi-colinearity relationship between the value

# ### Training the model
# Now that we've eplored the data a bit, we can go ahead and split the data into training and testing sets.
# 
# importing model_selection.train_test_split from sklearn to split the data into training and testing sets.

# In[52]:


from sklearn.model_selection import train_test_split


# In[53]:


#Setting the test size as 30%
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=101)


# In[54]:


#Importing LinearRegression from sklearn.linear_model
from sklearn.linear_model import LinearRegression


# In[55]:


#Creating and instance of a LinearRegression() model namedn lm.
lr = LinearRegression()


# In[56]:


#Train/Fit lr on the training data.
lr.fit(X_train,y_train)


# In[57]:


coeff_df = pd.DataFrame(lr.coef_, X.columns, columns=['Coefficient'])
coeff_df


# #### Predicting Test Data
# Now that we have fit/trained our model, we can evaluate it's performance by predicting off the test values.
# 
# 
# Use lr.predict() to predict off the X_test set of the data.

# In[60]:


predictions = lr.predict(X_test)


# Let's have some fun and try to predict the sales of a company with their expenditure on ads

# In[61]:


lr_model = lr.fit(X,y)
#company A spends the following values on advertisement on different platforms; 20,10,11
lr_model.predict([[20,10,11]])


# In[62]:


#Create a scatterplot of the real test values versus the predicteed values
plt.scatter(y_test, predictions)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')


# #### Evaluating the Model

# The model performance will be evaluated by calculating the residual sum of squares and the explained variable score (R-squared). 

# In[63]:


from sklearn import metrics
print('MAE  :', metrics.mean_absolute_error(y_test,predictions))
print('MSE  :', metrics.mean_squared_error(y_test,predictions))
print('RMSE  :', np.sqrt(metrics.mean_squared_error(y_test,predictions)))
print('R-squared  :', ols_model.rsquared)


# #### Residuals

# Now we'll explore the residuals to make sure everything was okay with our data.
# 
# Plot a histogram of the residuals, if it looks normally distribted, we're on the right track

# In[67]:


residuals = y_test - predictions


# In[68]:


sns.distplot(residuals, bins=50)


# In[69]:


coeff_df


# #### The interpretation of the coefficients goes thus:

# - a 1 unit increase in TV ads is associated with an 0.044696 increase in sales
# - a 1 unit increase in radio ads is associated with an 0.187566 increase in sales
# - a 1 unit increase in newspapper ads is associated with an -0.000323 increase in sales

# #### Conclusion
# 
# The answer to the business problem (Should companies focus more on TV, radio, newspapper ads?) goes thus: The companies should focus more on promoting radio ads to see a larger increase in sales.

# In[ ]:




