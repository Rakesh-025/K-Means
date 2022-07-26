import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import	KMeans

 
# Kmeans on Telco_customerchurn Data set 
TCC = pd.read_excel(r"C:\Users\kaval\OneDrive\Desktop\360digit\datatypes\Telco_customer_churn.xlsx")

TCC.describe()
TCC.info()
TCC = TCC.drop(["Customer ID","Count","Quarter"], axis = 1)

# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)
# Creating dummies for non numerical data
dff = pd.get_dummies(TCC)
dff = pd.get_dummies(dff,drop_first = True)
# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(dff.iloc[:, :])

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 25))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters =7)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
TCC['clust'] = mb # creating a  new column and assigning it to new column 

TCC.head()
df_norm.head()
TCC.shape

TCC = TCC.iloc[:,[27,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]]
TCC.head()

TCC.iloc[:, 2:].groupby(TCC.clust).mean()

TCC.to_csv("Kmeans_TCC.csv", encoding = "utf-8")

import os
os.getcwd()
                           