import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import	KMeans

# Kmeans on AutoInsurance Data set 
Auto_DF = pd.read_csv(r"C:\Users\kaval\OneDrive\Desktop\360digit\datatypes\AutoInsurance.csv")

Auto_DF.describe()
Auto_DF.info()
Auto_DF = Auto_DF.drop(["Customer","State","Effective To Date"], axis = 1)

# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)


#@create dummies
df_new=pd.get_dummies(Auto_DF)
df_new1=pd.get_dummies(Auto_DF , drop_first=True)


# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(df_new1.iloc[:, :])

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 50))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 15)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
Auto_DF['clust'] = mb # creating a  new column and assigning it to new column 

Auto_DF.head()
df_norm.head()
Auto_DF.info()
Auto_DF.shape


Auto_DF = Auto_DF.iloc[:,[21,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]
Auto_DF.head()

Auto_DF.iloc[:, 2:].groupby(Auto_DF.clust).mean()

Auto_DF.to_csv("Kmeans_AutoInssurance.csv", encoding = "utf-8")

import os
os.getcwd()
                           