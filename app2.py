import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import plotly.express as px

from Preprocessing.data_preprocessing import clean_data, columns_encode, standard_data, pca
from Model_building_Evaluation.model_building_and_clustring import Elbow_method,sillhouette_score,k_means_cluster
from sklearn.exceptions import ConvergenceWarning
import warnings

# Title
st.title("Cluster Vision")

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Clustering Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=['csv'])

# Init session flags
for key in ["show_elbow", "show_silhouette", "show_clusters"]:
    if key not in st.session_state:
        st.session_state[key] = False

# Initialize session state for data preview visibility
if "show_data_preview" not in st.session_state:
    st.session_state.show_data_preview = True

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.error("❌ The uploaded CSV file is empty. Please upload a valid non-empty file.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")
        st.stop()
    
    if st.sidebar.button("👀 Show Data Preview"):
        st.session_state.show_data_preview = True

    if st.session_state.show_data_preview:
        st.subheader("👀 Data Preview")
        st.dataframe(df.head(10))

    if st.sidebar.button("❌ Hide the data preview"):
        st.session_state.show_data_preview = False


    final_data = df.copy()

    st.sidebar.subheader("🔎 Feature Selection")
    Features = df.columns.to_list()
    feature_list = st.sidebar.multiselect("Select features for clustering", Features)



    st.sidebar.subheader("📌 Clustering Parameters")
    k = st.sidebar.slider("Select Max Number of Clusters (for evaluation)", 2, 20, 5)

    # ---------------- Main Area ----------------
    
    if feature_list:
        # Ensure selected features are numeric
        # if not all(pd.api.types.is_numeric_dtype(df[col]) for col in feature_list):
        #     st.error("⚠️ Selected features must be numeric. Please select numeric columns only.")
        #     st.stop()

        df = df[feature_list]

        # --------- Elbow Method ---------
        if st.sidebar.button("📉 Show Elbow Method"):
            st.session_state.show_elbow = True
        if st.session_state.show_elbow:
            try:
                df_clean = clean_data(feature_list, df)
                df_encoded = columns_encode(feature_list, df_clean)
                X_std = standard_data(df_encoded)
                wcss = Elbow_method(X_std, k)

                fig, ax = plt.subplots()
                ax.plot(range(2, k), wcss, marker='o')
                ax.set_xlabel("Number of clusters")
                ax.set_ylabel("WCSS")
                ax.set_title("Elbow Method")

                st.subheader("📉 Elbow Method Plot")
                st.pyplot(fig)

                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                st.download_button("⬇️ Download Elbow Plot", buf.getvalue(),
                                   file_name="elbow_method.png", mime="image/png")

                if st.button("❌ Hide Elbow Plot"):
                    st.session_state.show_elbow = False
            except Exception as e:
                st.error(f"❌ Elbow Method failed: {e}")

        # --------- Silhouette Score ---------
        if st.sidebar.button("📈 Show Silhouette Score"):
            st.session_state.show_silhouette = True
        if st.session_state.show_silhouette:
            try:
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
            except Exception as e:
                st.error(f"❌ Silhouette Score plotting failed: {e}")

        k_value = st.sidebar.text_input("K-value for clustering", "3")

        # --------- Clustering ---------
        if st.sidebar.button("🚀 Apply Clustering"):
            try:
                df_clean = clean_data(feature_list, df)
                df_encoded = columns_encode(feature_list, df_clean)
                X_std = standard_data(df_encoded)

                with warnings.catch_warnings():
                    warnings.filterwarnings('error', category=ConvergenceWarning)
                    clustered_df = k_means_cluster(df_encoded, X_std, int(k_value))

                final_data['clusters'] = clustered_df['clusters']
                st.session_state.df = clustered_df
                st.session_state.X_std = X_std
                st.session_state.final_data = final_data

                st.subheader("✅ Clustered Data Preview")
                st.dataframe(clustered_df.head())

            except ConvergenceWarning:
                st.error("❌ K-Means did not converge. Try adjusting 'k' or preprocessing the data.")
            except Exception as e:
                st.error(f"❌ Clustering failed: {e}")

        # --------- Cluster Visualization ---------
        if st.sidebar.button("🎨 Show 2D/3D Cluster Visualization"):
             st.session_state.show_clusters = True

        if st.session_state.show_clusters:
            try:
                if "df" in st.session_state and "X_std" in st.session_state:
                    n_features = len(feature_list)

                    if n_features == 2:
                        # 2D Scatter plot
                        df_clean = clean_data(feature_list, df)
                        df_encoded = columns_encode(feature_list, df_clean)
                        X_std = standard_data(df_encoded)
                        Features_2d = pd.DataFrame(X_std, columns=['f1', 'f2'])
                        Features_2d['cluster'] = st.session_state.df['clusters']

                        st.subheader("📊 2D Cluster Visualization")
                        fig, ax = plt.subplots()
                        scatter = ax.scatter(
                            Features_2d['f1'], Features_2d['f2'],
                            c=Features_2d['cluster'], cmap='viridis', alpha=0.8
                        )
                        ax.set_xlabel('Feature 1')
                        ax.set_ylabel('Feature 2')
                        ax.set_title('2D KMeans Cluster Visualization')
                        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
                        ax.add_artist(legend1)
                        st.pyplot(fig)

                    elif n_features > 2:
                        # 3D PCA Scatter plot
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

                    else:
                        st.warning("⚠️ At least 2 features are needed for visualization.")

                    if st.button("❌ Hide Cluster Plot"):
                        st.session_state.show_clusters = False
                else:
                    st.warning("⚠️ Please apply clustering first.")
            except Exception as e:
                st.error(f"❌ Visualization failed: {e}")

        # --------- Download Data ---------
        try:
            if "final_data" in st.session_state:
                csv = st.session_state.final_data.to_csv(index=False).encode('utf-8')
                st.subheader("⬇️ Download Clustered Data")
                st.download_button("📥 Download CSV", csv, "clusters.csv", "text/csv")
        except Exception as e:
            st.error(f"❌ Failed to export data: {e}. Please retry.")

else:
    st.info("👈 Upload a CSV file from the sidebar to begin.")
