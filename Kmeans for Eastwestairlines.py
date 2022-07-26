import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist 

# Kmeans on Eastwestairlines Data set 
EWA = pd.read_excel(r"C:\Users\kaval\OneDrive\Desktop\360digit\datatypes\EastWestAirlines.xlsx",sheet_name ="data")

EWA.describe()
EWA.info()
EWA = EWA.drop(["ID#"], axis = 1)

# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
EWA.info()
df_norm = norm_func(EWA.iloc[:, :])


###### scree plot or elbow curve ############
TWSS = []
k = list(range(2,25))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 7)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
EWA['clust'] = mb # creating a  new column and assigning it to new column 

EWA.head()
df_norm.head()

EWA = EWA.iloc[:,[11,3,4,5,6,7,8,9,10]]
EWA.head()

EWA.iloc[:, 2:12].groupby(EWA.clust).mean()

EWA.to_csv("Kmeans_EastWestAirlines.csv", encoding = "utf-8")

import os
os.getcwd()
