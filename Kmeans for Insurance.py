import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import	KMeans

 
# Kmeans on Insurance Data set 
Insurance_df = pd.read_csv(r"C:\Users\kaval\OneDrive\Desktop\360digit\datatypes\Insurance Dataset.csv")

Insurance_df.describe()
Insurance_df.info()

# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(Insurance_df.iloc[:, :])

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

# Selecting 4 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
Insurance_df['clust'] = mb # creating a  new column and assigning it to new column 

Insurance_df.head()
df_norm.head()
Insurance_df.shape
Insurance_df = Insurance_df.iloc[:,[5,0,1,2,3,4]]
Insurance_df.head()

Insurance_df.iloc[:, :].groupby(Insurance_df.clust).mean()

Insurance_df.to_csv("Kmeans_Insurance.csv", encoding = "utf-8")

import os
os.getcwd()
                           