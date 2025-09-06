
#  Input should be the the cleaned data frame 
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


#  apply the model on the data 

def Elbow_method(data, n_clusters):
    wcss=[]
    for i in range(2,n_clusters):
        model=KMeans(n_clusters=i)
        model.fit(data)
        wcss.append(model.inertia_)
    
    return wcss

def sillhouette_score(data, n_clusters):
    silhouette_scores = []
    for i in range(2, n_clusters):
        model = KMeans(n_clusters=i)
        labels = model.fit_predict(data)
        score = silhouette_score(data, labels)
        silhouette_scores.append(score)
    
    return silhouette_scores

def k_means_cluster(df,data,k):
    model=KMeans(n_clusters=k)
    df['clusters']=model.fit_predict(data)

    return df



#  plot the value of k using the elbow mwthod 

# Visualize the clusters 2d and 3d...