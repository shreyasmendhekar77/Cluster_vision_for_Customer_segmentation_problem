# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# from data_preprocessing import clean_data, columns_encode, standard_data, pca
# from model_building_and_clustring import Elbow_method, k_means_cluster, sillhouette_score
# import plotly.express as px

# # Title
# st.title("📊 CSV Upload and Clustering Visualization")

# # ---------------- Sidebar ----------------
# st.sidebar.header("⚙️ Clustering Controls")

# # Upload CSV
# uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=['csv'])

# # Global variables
# df = None
# final_data = None

# if uploaded_file is not None:
#     df = pd.read_csv(uploaded_file)
#     final_data = df.copy()

#     st.sidebar.subheader("🔎 Feature Selection")
#     Features = df.columns.to_list()
#     feature_list = st.sidebar.multiselect("Select features for clustering", Features)

#     st.sidebar.subheader("📌 Clustering Parameters")
#     k = st.sidebar.slider("Select Max Number of Clusters (for evaluation)", 2, 20, 5)

#     # ---------------- Main Area ----------------
#     st.subheader("👀 Data Preview")
#     st.dataframe(df.head())

#     if feature_list:
#         df = df[feature_list]

#         # Buttons in sidebar
#         if st.sidebar.button("🔹 Elbow Method"):
#             df_clean = clean_data(feature_list, df)
#             df_encoded = columns_encode(feature_list, df_clean)
#             X_std = standard_data(df_encoded)

#             wcss = Elbow_method(X_std, k)
#             fig, ax = plt.subplots()
#             ax.plot(range(2, k), wcss, marker='o')
#             ax.set_xlabel("Number of clusters")
#             ax.set_ylabel("Within Cluster Sum of Squares (WCSS)")
#             ax.set_title("Elbow Method")
#             st.subheader("📉 Elbow Method Plot")
#             st.pyplot(fig)
        


#         if st.sidebar.button("🔹 Silhouette Score"):
#             df_clean = clean_data(feature_list, df)
#             df_encoded = columns_encode(feature_list, df_clean)
#             X_std = standard_data(df_encoded)

#             sill_scores = sillhouette_score(X_std, k)
#             fig, ax = plt.subplots()
#             ax.plot(range(2, k), sill_scores, marker='o')
#             ax.set_xlabel("Number of clusters")
#             ax.set_ylabel("Silhouette Score")
#             ax.set_title("Silhouette Score Analysis")
#             st.subheader("📈 Silhouette Score Plot")
#             st.pyplot(fig)


#         k_value = st.sidebar.text_input("K-value for clustering", "3")

#         if st.sidebar.button("🚀 Apply Clustering"):
#             df_clean = clean_data(feature_list, df)
#             df_encoded = columns_encode(feature_list, df_clean)
#             X_std = standard_data(df_encoded)

#             clustered_df = k_means_cluster(df_encoded, X_std, int(k_value))
#             final_data['clusters'] = clustered_df['clusters']

#             st.session_state.df = clustered_df
#             st.session_state.X_std = X_std
#             st.session_state.final_data = final_data

#             st.subheader("✅ Clustered Data Preview")
#             st.dataframe(clustered_df.head())

#         if st.sidebar.button("🎨 Visualize Clusters"):
#             if "df" in st.session_state and "X_std" in st.session_state:
#                 x_pca = pca(st.session_state.X_std, 3)
#                 Features_pca = pd.DataFrame(x_pca, columns=['f1', 'f2', 'f3'])
#                 Features_pca['cluster'] = st.session_state.df['clusters']

#                 st.subheader("3D Cluster Visualization")
#                 fig = px.scatter_3d(
#                     Features_pca,
#                     x='f1', y='f2', z='f3',
#                     color=Features_pca['cluster'].astype(str),
#                     symbol=Features_pca['cluster'].astype(str),
#                     title='KMeans Clusters (3D PCA Reduced)',
#                     opacity=0.8,
#                     hover_data=['f1', 'f2', 'f3', 'cluster']
#                 )
#                 fig.update_traces(marker=dict(size=5))
#                 st.plotly_chart(fig, use_container_width=True)
#             else:
#                 st.warning("⚠️ Please apply clustering first.")

#         if "final_data" in st.session_state:
#             csv = st.session_state.final_data.to_csv(index=False).encode('utf-8')
#             st.subheader("⬇️ Download Clustered Data")
#             st.download_button("📥 Download CSV", csv, "clusters.csv", "text/csv")

# else:
#     st.info("👈 Upload a CSV file from the sidebar to begin.")


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import plotly.express as px
from data_preprocessing import clean_data, columns_encode, standard_data, pca
from model_building_and_clustring import Elbow_method, k_means_cluster, sillhouette_score



# Title
st.title("📊 CSV Upload and Clustering Visualization")

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Clustering Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=['csv'])

# Init session flags
for key in ["show_elbow", "show_silhouette", "show_clusters"]:
    if key not in st.session_state:
        st.session_state[key] = False

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    final_data = df.copy()

    st.sidebar.subheader("🔎 Feature Selection")
    Features = df.columns.to_list()
    feature_list = st.sidebar.multiselect("Select features for clustering", Features)

    st.sidebar.subheader("📌 Clustering Parameters")
    k = st.sidebar.slider("Select Max Number of Clusters (for evaluation)", 2, 20, 5)
    k_value = st.sidebar.text_input("K-value for clustering", "3")

    # ---------------- Main Area ----------------
    st.subheader("👀 Data Preview")
    st.dataframe(df.head())

    if feature_list:
        df = df[feature_list]

        # --------- Elbow Method ---------
        if st.sidebar.button("📉 Show Elbow Method"):
            st.session_state.show_elbow = True
        if st.session_state.show_elbow:
            df_clean = clean_data(feature_list, df)
            df_encoded = columns_encode(feature_list, df_clean)
            X_std = standard_data(df_encoded)
            wcss = Elbow_method(X_std, k)

            fig, ax = plt.subplots()
            ax.plot(range(2, k), wcss, marker='o')
            ax.set_xlabel("Number of clusters")
            ax.set_ylabel("Within Cluster Sum of Squares (WCSS)")
            ax.set_title("Elbow Method")

            st.subheader("📉 Elbow Method Plot")
            st.pyplot(fig)

            # Download button
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            st.download_button("⬇️ Download Elbow Plot", buf.getvalue(),
                               file_name="elbow_method.png", mime="image/png")

            if st.button("❌ Hide Elbow Plot"):
                st.session_state.show_elbow = False

        # --------- Silhouette Score ---------
        if st.sidebar.button("📈 Show Silhouette Score"):
            st.session_state.show_silhouette = True
        if st.session_state.show_silhouette:
            df_clean = clean_data(feature_list, df)
            df_encoded = columns_encode(feature_list, df_clean)
            X_std = standard_data(df_encoded)
            sill_scores = sillhouette_score(X_std, k)

            fig, ax = plt.subplots()
            ax.plot(range(2, k), sill_scores, marker='o')
            ax.set_xlabel("Number of clusters")
            ax.set_ylabel("Silhouette Score")
            ax.set_title("Silhouette Score Analysis")

            st.subheader("📈 Silhouette Score Plot")
            st.pyplot(fig)

            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            st.download_button("⬇️ Download Silhouette Plot", buf.getvalue(),
                               file_name="silhouette_score.png", mime="image/png")

            if st.button("❌ Hide Silhouette Plot"):
                st.session_state.show_silhouette = False

        # --------- Clustering ---------
        if st.sidebar.button("🚀 Apply Clustering"):
            df_clean = clean_data(feature_list, df)
            df_encoded = columns_encode(feature_list, df_clean)
            X_std = standard_data(df_encoded)

            clustered_df = k_means_cluster(df_encoded, X_std, int(k_value))
            final_data['clusters'] = clustered_df['clusters']

            st.session_state.df = clustered_df
            st.session_state.X_std = X_std
            st.session_state.final_data = final_data

            st.subheader("✅ Clustered Data Preview")
            st.dataframe(clustered_df.head())

        # --------- Cluster Visualization ---------
        if st.sidebar.button("🎨 Show 3D Clusters"):
            st.session_state.show_clusters = True
        if st.session_state.show_clusters:
            if "df" in st.session_state and "X_std" in st.session_state:
                x_pca = pca(st.session_state.X_std, 3)
                Features_pca = pd.DataFrame(x_pca, columns=['f1', 'f2', 'f3'])
                Features_pca['cluster'] = st.session_state.df['clusters']

                st.subheader("🎨 3D Cluster Visualization")
                fig = px.scatter_3d(
                    Features_pca,
                    x='f1', y='f2', z='f3',
                    color=Features_pca['cluster'].astype(str),
                    symbol=Features_pca['cluster'].astype(str),
                    title='KMeans Clusters (3D PCA Reduced)',
                    opacity=0.8
                )
                fig.update_traces(marker=dict(size=5))
                st.plotly_chart(fig, use_container_width=True)

                # # Download plotly as HTML
                # buf = io.StringIO()
                # fig.write_html(buf)
                # st.download_button("⬇️ Download 3D Plot", buf.getvalue(),
                #                    file_name="3D cluster image.png", mime="image/png")

                if st.button("❌ Hide Cluster Plot"):
                    st.session_state.show_clusters = False
            else:
                st.warning("⚠️ Please apply clustering first.")

        # --------- Download Data ---------
        if "final_data" in st.session_state:
            csv = st.session_state.final_data.to_csv(index=False).encode('utf-8')
            st.subheader("⬇️ Download Clustered Data")
            st.download_button("📥 Download CSV", csv, "clusters.csv", "text/csv")

else:
    st.info("👈 Upload a CSV file from the sidebar to begin.")
