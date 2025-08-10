import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from  data_preprocessing import clean_data, columns_encode ,standard_data
from model_building_and_clustring import Elbow_method
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
# Title
st.title("CSV Upload and Graph Display")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    # Read file
    df = pd.read_csv(uploaded_file)
    
    st.write("### Preview of the Data")
    st.dataframe(df.head())

    Features=df.columns.to_list()
    st.write("Select the Features for the clustring")
    feature_list=st.multiselect("Select the features for performing the clustring ",Features)
    # st.write("Your selected features are - ")
    # for feature in feature_list:
    #     st.write(feature)
    df=df[feature_list]
    print(feature_list)
    if st.button("Clean and Encode Data"):
        # df=clean_data(feature_list,df)
        # df=columns_encode(feature_list,df)
        # st.dataframe(df.head())
        # st.write(X_std[0])
        pass

    st.dataframe(df.head())
    
    # if st.button("Find k with elbow method"):s
    k = st.slider("Select Number of Clusters (k)", min_value=1, max_value=20, value=3)
    st.write("Selected number of clusters:", k)
    if st.button("Elbow Method"):
        df=clean_data(feature_list,df)
        df=columns_encode(feature_list,df)
        st.dataframe(df.head())
        # st.dataframe(df.head())
        X_std=standard_data(df)
        wcss=Elbow_method(X_std,k)
        # Plot the graph
        plt.plot(range(2,k),wcss,marker='o')
        plt.xlabel("Number of clusters")
        plt.ylabel("Within Cluster Sum of Square")
        plt.title("K-value vs WCSS")
        # plt.show()
        st.pyplot(plt)




    if st.button("Apply clustring "):
        st.write("Applying clustring")



    if st.button("Visualize Clusters"):
        st.write("Part to be coded")






    # Plot the clusters in PCA-reduced space
    # plt.figure(figsize=(8,6))
    # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='viridis', s=50)
    # plt.xlabel('PCA Component 1')
    # plt.ylabel('PCA Component 2')
    # plt.title('KMeans Clusters (PCA Reduced)')
    # plt.colorbar(label='Cluster')
    # # plt.show()
    # st.pyplot(plt)


    # # Dropdowns for selecting columns
    # columns = df.columns.tolist()
    # x_col = st.selectbox("Select X-axis column", columns)
    # y_col = st.selectbox("Select Y-axis column", columns)

    # # Plot using matplotlib
    # st.write("### Line Chart")
    # fig, ax = plt.subplots()
    # ax.plot(df[x_col], df[y_col])
    # ax.set_xlabel(x_col)
    # ax.set_ylabel(y_col)
    # ax.set_title(f"{y_col} vs {x_col}")
    # st.pyplot(fig)




# 
