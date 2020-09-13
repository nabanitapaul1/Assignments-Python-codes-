# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 09:17:37 2020

@author: Nabanita
"""

import pandas as pd # Importing data sets
import seaborn as sb # For visualization  
import matplotlib.pyplot as plt # For Visulization
from sklearn.model_selection import train_test_split # for data splitting into train and test
from sklearn.linear_model import LogisticRegression

# 1.) Whether the clinet has suscribed a term deposit or not

# Import datasets
bank_data =  pd.read_csv("C:\\EXCELR\\ASSIGNMENTS\\LogisticsRegression\\bank-full.csv", sep=';')

# EDA
bank_data_colnames= list(bank_data.columns)
bank_data_colnames

# This will give  the datatypes informations
bank_data.dtypes
print(bank_data.info())

bank_data.head()
bank_data

bank_data.describe # This will give the records
bank_data.describe() # This will give the statistical information of numerical variables

bank_data.shape # This will give no. of rows and columns

# Analyisis of NA values

bank_data.isnull().sum()

# There is no  null values

# Graphical Representations

# For categorical variables

bank_data['job'].value_counts()
chart = sb.countplot(x="job", data=bank_data, palette="Set1" )
#chart.set_xticklabels(rotation=30,horizontalalignment='right')
#chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='right')

bank_data['marital'].value_counts()
chart = sb.countplot(x="marital", data=bank_data, palette="Set2" )
#chart.set_xticklabels(rotation=30,horizontalalignment='right')
#chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='right')

bank_data['education'].value_counts()
chart = sb.countplot(x="education", data=bank_data, palette="Set3" )
#chart.set_xticklabels(rotation=30,horizontalalignment='right')
#chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='right')

bank_data['default'].value_counts()
chart = sb.countplot(x="default", data=bank_data, palette="Set1" )
#chart.set_xticklabels(rotation=30,horizontalalignment='right')
#chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='right')

bank_data['housing'].value_counts()
chart = sb.countplot(x="housing", data=bank_data, palette="Set2" )
#chart.set_xticklabels(rotation=30,horizontalalignment='right')
#chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='right')

bank_data['loan'].value_counts()
chart = sb.countplot(x="loan", data=bank_data, palette="Set3" )
#chart.set_xticklabels(rotation=30,horizontalalignment='right')
#chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='right')

bank_data['contact'].value_counts()
chart = sb.countplot(x="contact", data=bank_data, palette="Set1" )
#chart.set_xticklabels(rotation=30,horizontalalignment='right')
#chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='right')

bank_data['month'].value_counts()
chart = sb.countplot(x="month", data=bank_data, palette="Set2" )
#chart.set_xticklabels(rotation=30,horizontalalignment='right')
#chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='right')

bank_data['poutcome'].value_counts()
chart = sb.countplot(x="poutcome", data=bank_data, palette="Set3" )
#chart.set_xticklabels(rotation=30,horizontalalignment='right')
#chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='right')

bank_data['y'].value_counts()
chart = sb.countplot(x="y", data=bank_data, palette="Set1" )
#chart.set_xticklabels(rotation=30,horizontalalignment='right')
#chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='right')

# Scatter plot and co-relations of numerical variables

sb.pairplot(bank_data)
bank_data.corr()

# Creating dummies
column_names = list(bank_data.columns)
column_names
bank_data.dtypes

bank_data_dummies =pd.get_dummies(bank_data[["job","marital","education","default","housing","loan","contact","month","poutcome"]])
column_names_d = list(bank_data_dummies.columns)
column_names_d 

# Dropping the columns for which we have created dummies
list(bank_data.columns)

bank_data.drop(["job","marital","education","default","housing","loan","contact","month","poutcome"],inplace=True,axis=1)
bank_data.info()
list(bank_data.columns)

# Adding the dummy variables created to the original data prame
bank_data_new =pd.concat([bank_data,bank_data_dummies],axis=1)
column_names = list(bank_data_new.columns)
column_names
bank_data_new.info()

# Changing  resoponse variable y into binary format
bank_data_new.y.head(500)
bank_data_new["y_new"] = 0
bank_data_new["y_new"] 
bank_data_new["y_new"].value_counts()

bank_data_new.loc[bank_data_new.y==" yes","y_new"] 

bank_data_new.loc[bank_data_new.y=="yes","y_new"]=1 
bank_data_new.loc[bank_data_new.y=="yes","y_new"] 
bank_data_new["y_new"].value_counts()
bank_data_new.y.value_counts()
bank_data_new.drop(["y"],axis=1,inplace=True)
list(bank_data_new.columns)
bank_data_new.y_new.head(500)


 # Data Partitoning
y =  bank_data_new.y_new
x=  bank_data_new.drop(["y_new"], axis=1) 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

# For x

x_train.head()
x_test.head()
len(x_train)
len(x_test)
x_train.shape
x_test.shape

# For y

y_train.head()
y_test.head()
len(y_train)
len(y_test)
y_train.shape
y_test.shape

# Model Building

classifier = LogisticRegression()
classifier.fit(x_train,y_train)

classifier.coef_ # coefficients of features 
classifier.predict_proba (x_train) # Probability values 

y_pred = classifier.predict(x_test)
len(y_pred)

bank_data_test_y = pd.DataFrame(y_test)
list(bank_data_test_y.columns)
bank_data_test_y["y_pred"] = y_pred

bank_data_test_y.head(50)

#y_prob = pd.DataFrame(classifier.predict_proba(X.iloc[:,:]))
#new_df = pd.concat([claimants,y_prob],axis=1)

# Confusion Matrix

from sklearn.metrics import confusion_matrix
#  need to pass predicted value and the test datasets
confusion_matrix = confusion_matrix(y_test,y_pred) 
print (confusion_matrix)

# Accuracy 
from sklearn.metrics import accuracy_score 

Accuracy = accuracy_score(y_test,y_pred)
Accuracy
Error = 1-Accuracy
Error *100

########### ROC curve ###########
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test, y_pred)

auc = roc_auc_score(y_test, y_pred)

plt.plot(fpr, tpr, color='cyan', label='ROC')

# 2.) Classify whether application accepted or not using Logistic regression

# Import datasets
credit =  pd.read_csv("C:\\EXCELR\\ASSIGNMENTS\\LogisticsRegression\\creditcard.csv", sep=',')
credit= credit.drop(["Unnamed: 0"], axis=1)

# EDA
credit_colnames= list(credit.columns)
credit_colnames

# This will give  the datatypes information
credit.dtypes
print(credit.info())

credit.head()
credit

credit.describe # This will give the records
credit.describe() # This will give the statistical information of numerical variables

credit.shape # This will give no. of rows and columns

# Analyisis of NA values

credit.isnull().sum()

# There is no  null values

# Graphical Representations

# For categorical variables

credit['card'].value_counts()
chart = sb.countplot(x="card", data=credit, palette="Set1" )
#chart.set_xticklabels(rotation=30,horizontalalignment='right')
#chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='right')

credit['owner'].value_counts()
chart = sb.countplot(x="owner", data=credit, palette="Set2" )
#chart.set_xticklabels(rotation=30,horizontalalignment='right')
#chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='right')

credit['selfemp'].value_counts()
chart = sb.countplot(x="selfemp", data=credit, palette="Set3" )
#chart.set_xticklabels(rotation=30,horizontalalignment='right')
#chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='right')

# Scatter plot and co-relations of numerical variables

sb.pairplot(credit)
credit.corr()

# Creating dummies
column_names = list(credit.columns)
column_names
credit.dtypes

credit_dummies =pd.get_dummies(credit[["owner","selfemp"]])
column_names_d = list(credit_dummies.columns)
column_names_d 

# Dropping the columns for which we have created dummies
list(credit.columns)

credit.drop(["owner","selfemp"],inplace=True,axis=1)
credit.info()
list(credit.columns)

# Adding the dummy variables created to the original data prame
credit_new =pd.concat([credit,credit_dummies],axis=1)
column_names = list(credit_new.columns)
column_names
credit_new.info()

# Changing  resoponse variable card into binary format
credit_new.card.head(500)
credit_new["card_new"] = 0
credit_new["card_new"] 
credit_new["card_new"].value_counts()

credit_new.loc[credit_new.card==" yes","card_new"] 
credit_new.loc[credit_new.card=="yes","card_new"]=1 
credit_new.loc[credit_new.card=="yes","card_new"] 
credit_new["card_new"].value_counts()
credit_new.card.value_counts()
credit_new.drop(["card"],axis=1,inplace=True)
list(credit_new.columns)
credit_new.card_new.head(500)

 # Data Partitoning
y =  credit_new.card_new
x=  credit_new.drop(["card_new"], axis=1) 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

# For x

x_train.head()
x_test.head()
len(x_train)
len(x_test)
x_train.shape
x_test.shape

# For y

y_train.head()
y_test.head()
len(y_train)
len(y_test)
y_train.shape
y_test.shape

# Model Building

classifier = LogisticRegression()
classifier.fit(x_train,y_train)

classifier.coef_ # coefficients of features 
classifier.predict_proba (x_train) # Probability values 


y_pred = classifier.predict(x_test)
len(y_pred)

credit_test_y = pd.DataFrame(y_test)
list(credit_test_y.columns)
credit_test_y["y_pred"] = y_pred

credit_test_y.head(50)


#y_prob = pd.DataFrame(classifier.predict_proba(X.iloc[:,:]))
#new_df = pd.concat([claimants,y_prob],axis=1)


# Confusion Matrix

from sklearn.metrics import confusion_matrix
#  need to pass predicted value and the test datasets
confusion_matrix = confusion_matrix(y_test,y_pred) 
print (confusion_matrix)

# Accuracy 
from sklearn.metrics import accuracy_score 

Accuracy = accuracy_score(y_test,y_pred)
Accuracy
Error = 1-Accuracy
Error *100

########### ROC curve ###########
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test, y_pred)

auc = roc_auc_score(y_test, y_pred)

plt.plot(fpr, tpr, color='orange', label='ROC')
