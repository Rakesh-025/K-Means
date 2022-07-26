import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import	KMeans

 
# Kmeans on Crime Data set 
Crime_df = pd.read_csv(r"C:\Users\kaval\OneDrive\Desktop\360digit\datatypes\crime_data.csv")

Crime_df.describe()
Crime_df.info()
Crime_df = Crime_df.drop(["Unnamed: 0"], axis = 1)

# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(Crime_df.iloc[:, :])

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 11))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
Crime_df['clust'] = mb # creating a  new column and assigning it to new column 

Crime_df.head()
df_norm.head()

Crime_df = Crime_df.iloc[:,[4,0,1,2,3]]
Crime_df.head()

Crime_df.iloc[:, 2:8].groupby(Crime_df.clust).mean()

Crime_df.to_csv("Kmeans_Crimedata.csv", encoding = "utf-8")

import os
os.getcwd()
                           