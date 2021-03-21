# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 08:07:05 2020

@author: Nabanita
"""

# import library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as  sns

# 1.) Perform Clustering for the crime data and identify the number of clusters formed and draw inferences.

# Import  Data

crime =  pd.read_csv("F:\\EXCELR\\ASSIGNMENTS\\Python\\Clustering\\crime_data.csv")

# EDA

crime.head() # Top five rows

crime.info()
crime.describe
crime.describe()

# Normalization of data
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)
crime_norm = norm_func(crime.iloc[:, 1:])
crime_norm.head() # Top five rows
crime_norm.describe()

# Clustering
# k means
from sklearn.cluster import KMeans # this is to import Kmeans function from sklearn
from scipy.spatial.distance import cdist
# Scree plot or elbow curve

k= list(range(1,15))
k

TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(crime_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(crime_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,crime_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))

TWSS
# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)


# Selecting five clusters from the above scree plot which is optimum number of clusters
model =KMeans(n_clusters=5)
model.fit(crime_norm)

model.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(model.labels_)  # converting numpy array into pandas series object 
crime['clust']=md # creating a  new column and assigning it to new column 
crime
crime = crime.iloc[:,[0,1,2,3,4,5]]

clusters = crime.iloc[:,1:5].groupby(crime.clust).mean()


#2.) Perform clustering (Both hierarchical and K means clustering) for the airlines data to obtain optimum number of clusters. 

# Import data

airlines =  pd.read_excel("F:\\EXCELR\\ASSIGNMENTS\\Python\\Clustering\\EastWestAirlines.xlsx",sheet_name="data")

#EDA
airlines = airlines.drop(["ID#"],axis=1) # Droping ID# column
airlines.head()
airlines.info()
airlines.describe()

# Normalization
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)
airlines_norm = norm_func(airlines)
airlines_norm.head() # Top five rows
airlines_norm.describe()

# Clustering
# Hierarchical Clustering 

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z=  linkage(airlines_norm, method="complete", metric="euclidean")
z
plt.figure(figsize=(15, 10));
plt.title('Hierarchical Clustering Dendrogram');
plt.xlabel('Index');
plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

# Agglomerative Clustering

from sklearn.cluster import AgglomerativeClustering
h_complete	=	AgglomerativeClustering(n_clusters=5,linkage='complete',affinity = "euclidean").fit(airlines_norm) 
h_complete.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(h_complete.labels_)  # converting numpy array into pandas series object 
airlines['clust']=md # creating a  new column and assigning it to new column 
airlines
clusters = airlines.groupby(airlines.clust).mean()
clusters

# Kmeans

from sklearn.cluster import KMeans # this is to import Kmeans function from sklearn
from scipy.spatial.distance import cdist

# Scree plot or elbow curve

k= list(range(2,30))
k

TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(airlines_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(airlines_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,airlines_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))

TWSS
# Scree plot 
plt.plot(k,TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)


# Selecting five clusters from the above scree plot which is optimum number of clusters
model =KMeans(n_clusters=8)
model.fit(airlines_norm)

model.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(model.labels_)  # converting numpy array into pandas series object 
airlines['clust']=md # creating a  new column and assigning it to new column 
airlines
clusters = airlines.groupby(airlines.clust).mean()
clusters.head()

