
#  Input should be the the cleaned data frame 
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


#  apply the model on the data 

def Elbow_method(data, n_clusters):
    wcss=[]
    for i in range(2,n_clusters):
        model=KMeans(
        n_clusters=i,             
        init='k-means++',        
        n_init=20,               
        max_iter=500,                  
        )
        model.fit(data)
        wcss.append(model.inertia_)
    
    return wcss

def sillhouette_score(data, n_clusters):
    silhouette_scores = []
    for i in range(2, n_clusters):
        model = KMeans(
        n_clusters=i,             
        init='k-means++',        
        n_init=20,               
        max_iter=500,            
        )
        labels = model.fit_predict(data)
        score = silhouette_score(data, labels)
        silhouette_scores.append(score)
    
    return silhouette_scores

def k_means_cluster(df,data,k):
    model=KMeans(
    n_clusters=k,             # Use Elbow Method to find optimal k
    init='k-means++',        # Smart initialization
    n_init=20,               # Multiple initializations for robustness
    max_iter=500,            # Sufficient iterations for convergence
    tol=1e-4,                # Default convergence tolerance
    )
    df['clusters']=model.fit_predict(data)

    return df



#  plot the value of k using the elbow mwthod 

# Visualize the clusters 2d and 3d...