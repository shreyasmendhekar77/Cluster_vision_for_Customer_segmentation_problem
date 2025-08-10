import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from  data_preprocessing import clean_data, columns_encode ,standard_data,pca
from model_building_and_clustring import Elbow_method,k_means_cluster
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
# Title
st.title("CSV Upload and Graph Display")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    # Read file
    df = pd.read_csv(uploaded_file)
    final_data=df
    
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



    k_value = st.text_input("The K- value you want to try - ")
    if st.button("Apply clustering"):
        df = clean_data(feature_list, df)
        df = columns_encode(feature_list, df)
        X_std = standard_data(df)
        df = k_means_cluster(df, X_std, int(k_value))
        st.session_state.df = df
        st.session_state.X_std = X_std
        final_data['clusters']=st.session_state.df['clusters']
        st.session_state.final_data = final_data
        st.dataframe(df.head())

    if st.button("Visualize Clusters"):
        if "df" in st.session_state and "X_std" in st.session_state:
            x_pca = pca(st.session_state.X_std, 3)

            Features = pd.DataFrame(x_pca, columns=['f1', 'f2', 'f3'])
            Features['cluster'] = st.session_state.df['clusters']
            st.dataframe(Features.head(2))

            fig = px.scatter_3d(
                Features,
                x='f1', y='f2', z='f3',
                color=Features['cluster'].astype(str),
                title='KMeans Clusters (3D PCA Reduced)',
                opacity=0.75,
                symbol=Features['cluster'].astype(str),
                hover_data=['f1', 'f2', 'f3', 'cluster']
            )
            fig.update_traces(marker=dict(size=5))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please apply clustering first!")
    
   
    csv =st.session_state.final_data.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Clustered Data", csv, "clusters.csv", "text/csv")
    
    # final=df['cluster']

    



 
        
         
        





        






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
