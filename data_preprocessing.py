# First clean the dataset with 

# Check the features are  text or numeric

# If they are text then apply the feature transformation - 

#  If neumeric then  check the data type 

#  Make the dataset float...


#  Outpur should be a dataframe and input should be the columns and input dataframe...
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
le=LabelEncoder()

def clean_data(colum_list:list,data):
    for i in colum_list:
        if int(data[i].isna().sum())!=0 and not pd.api.types.is_numeric_dtype(data[i]):
            data[i]=data[i].fillna(data[i].mode)
        elif int(data[i].isna().sum())!=0:
            data[i]=data[i].fillna(data[i].mean())
    
    return data


def columns_encode(column_list:list,data):
    for i in column_list:
        if not pd.api.types.is_numeric_dtype(data[i]):
            le = LabelEncoder()
            data[i] = le.fit_transform(data[i].astype(str))
            print(f"Encoded column: {i}")
    
    return data



def standard_data(data):
    std=StandardScaler()
    # data=data[column_list]
    x_std=std.fit_transform(data)
    return x_std


def pca(data,component):
    feature=PCA(n_components=component)
    return feature.fit_transform(data)
    



