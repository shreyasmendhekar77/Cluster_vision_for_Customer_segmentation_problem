import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from  data_preprocessing import clean_data, columns_encode ,standard_data,pca
from model_building_and_clustring import Elbow_method,k_means_cluster,sillhouette_score
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
    
    if st.button("Silhouette Score"):
        df=clean_data(feature_list,df)
        df=columns_encode(feature_list,df)
        st.dataframe(df.head())
        # st.dataframe(df.head())
        X_std=standard_data(df)
        Sill_Score=sillhouette_score(X_std,k)
        # Plot the graph
        plt.plot(range(2,k),Sill_Score,marker='o')
        plt.xlabel("Number of clusters")
        plt.ylabel("Sillhoutte Score")
        plt.title("K-value vs Sillhoutte Score")
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

    



 
        
         
        





        






#     # Plot the clusters in PCA-reduced space
#     # plt.figure(figsize=(8,6))
#     # plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Cluster'], cmap='viridis', s=50)
#     # plt.xlabel('PCA Component 1')
#     # plt.ylabel('PCA Component 2')
#     # plt.title('KMeans Clusters (PCA Reduced)')
#     # plt.colorbar(label='Cluster')
#     # # plt.show()
#     # st.pyplot(plt)


#     # # Dropdowns for selecting columns
#     # columns = df.columns.tolist()
#     # x_col = st.selectbox("Select X-axis column", columns)
#     # y_col = st.selectbox("Select Y-axis column", columns)

#     # # Plot using matplotlib
#     # st.write("### Line Chart")
#     # fig, ax = plt.subplots()
#     # ax.plot(df[x_col], df[y_col])
#     # ax.set_xlabel(x_col)
#     # ax.set_ylabel(y_col)
#     # ax.set_title(f"{y_col} vs {x_col}")
#     # st.pyplot(fig)




# # # 

# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import plotly.express as px
# from data_preprocessing import clean_data, columns_encode, standard_data, pca
# from model_building_and_clustring import Elbow_method, k_means_cluster

# # Page configuration: wide layout helps centering in the middle column
# st.set_page_config(page_title="CSV Upload and Graph Display", layout="wide")

# # Initialize session state
# if "df_orig" not in st.session_state: st.session_state.df_orig = None
# if "df_features" not in st.session_state: st.session_state.df_features = None
# if "X_std" not in st.session_state: st.session_state.X_std = None
# if "final_data" not in st.session_state: st.session_state.final_data = None
# if "clustered" not in st.session_state: st.session_state.clustered = None
# if "k" not in st.session_state: st.session_state.k = 3
# if "feature_list" not in st.session_state: st.session_state.feature_list = None
# if "uploader_used" not in st.session_state: st.session_state.uploader_used = False

# # Left UI (controls)
# with st.sidebar:
#     st.header("Controls")

#     uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'], key="uploader")
#     if uploaded_file is not None:
#         st.session_state.uploader_used = True
#         if st.session_state.df_orig is None:
#             st.session_state.df_orig = pd.read_csv(uploaded_file)

#     if st.session_state.df_orig is not None:
#         all_features = st.session_state.df_orig.columns.tolist()
#         feature_list = st.multiselect("Features for clustering", all_features, key="feature_list")
#         if feature_list:
#             st.session_state.feature_list = feature_list

#     # Prepare data
#     if (st.session_state.df_orig is not None) and (st.session_state.feature_list is not None):
#         if st.button("Prepare Data"):
#             df = st.session_state.df_orig[st.session_state.feature_list]
#             df_clean = clean_data(st.session_state.feature_list, df)
#             df_enc = columns_encode(st.session_state.feature_list, df_clean)
#             X_std = standard_data(df_enc)

#             st.session_state.df_features = df_enc
#             st.session_state.X_std = X_std
#             st.session_state.final_data = None
#             st.session_state.clustered = None
#             st.success("Data prepared. You can run Elbow Method or clustering.")

#     st.markdown("---")

#     # Elbow Method options
#     if st.session_state.X_std is not None:
#         k = st.slider("Number of Clusters (k)", 1, 20, value=st.session_state.k, key="k_slider")
#         st.session_state.k = k

#         if st.button("Elbow Method"):
#             wcss = Elbow_method(st.session_state.X_std, int(st.session_state.k))
#             fig, ax = plt.subplots()
#             ax.plot(range(2, int(st.session_state.k) + 1), wcss, marker='o')
#             ax.set_xlabel("Number of clusters")
#             ax.set_ylabel("WCSS")
#             ax.set_title("K-value vs WCSS")
#             st.pyplot(fig)

#         # Apply clustering
#         st.markdown("---")
#         if st.button("Apply clustering"):
#             if st.session_state.df_features is not None and st.session_state.X_std is not None:
#                 df_clustered = k_means_cluster(st.session_state.df_features, st.session_state.X_std, int(st.session_state.k))
#                 st.session_state.clustered = df_clustered
#                 # Attach clusters to original data (if possible)
#                 if isinstance(st.session_state.df_orig, pd.DataFrame) and "clusters" in df_clustered.columns:
#                     final = st.session_state.df_orig.copy()
#                     final = final.assign(clusters=df_clustered["clusters"].values)
#                     st.session_state.final_data = final
#                 else:
#                     st.session_state.final_data = df_clustered
#                 st.success("Clustering applied.")

#         st.markdown("---")

#         # Visualization (optional, if clustering done)
#         if st.session_state.clustered is not None:
#             st.write("Visualization will appear in the main area.")

#     # Download
#     if st.session_state.final_data is not None:
#         csv = st.session_state.final_data.to_csv(index=False).encode('utf-8')
#         st.download_button("Download Clusters", csv, "clusters.csv", "text/csv")

# # Center/main area
# center_col1, center_col2, _ = st.columns([1, 2, 1])  # center content
# with center_col2:
#     st.title("Output")

#     if not st.session_state.uploader_used or st.session_state.df_orig is None:
#         st.info("Please upload a CSV on the left to begin.")
#         st.stop()

#     # Show prepared data preview
#     if st.session_state.df_features is None:
#         st.info("Select features and click 'Prepare Data' to transform the data.")
#         st.stop()

#     st.subheader("Preview of prepared data (features-only)")
#     st.dataframe(st.session_state.df_features.head())

#     # Show clustering results if available
#     if st.session_state.clustered is not None:
#         st.subheader("Clustering result (example view)")
#         st.dataframe(st.session_state.clustered.head())

#     # Show 3D PCA visualization if clustering done
#     if (st.session_state.X_std is not None) and (st.session_state.clustered is not None):
#         x_pca = pca(st.session_state.X_std, 3)
#         Features = pd.DataFrame(x_pca, columns=["f1", "f2", "f3"])
#         if "clusters" in st.session_state.clustered.columns:
#             Features["cluster"] = st.session_state.clustered["clusters"].astype(str)
#             fig = px.scatter_3d(
#                 Features, x="f1", y="f2", z="f3",
#                 color=Features["cluster"],
#                 title="KMeans Clusters (3D PCA Reduced)",
#                 opacity=0.75
#             )
#             fig.update_traces(marker=dict(size=5))
#             st.plotly_chart(fig, use_container_width=True)

#     # Show final data (with clusters) if available
#     if st.session_state.final_data is not None:
#         st.subheader("Final data with clusters (preview)")
#         st.dataframe(st.session_state.final_data.head())
