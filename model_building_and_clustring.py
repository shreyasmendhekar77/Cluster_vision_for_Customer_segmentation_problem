
#  Input should be the the cleaned data frame 
import pandas as pd
from sklearn.cluster import KMeans



#  apply the model on the data 

def Elbow_method(data, n_clusters):
    wcss=[]
    for i in range(2,n_clusters):
        model=KMeans(n_clusters=i)
        model.fit(data)
        wcss.append(model.inertia_)
    
    return wcss


    









#  plot the value of k using the elbow mwthod 

# Visualize the clusters 2d and 3d...