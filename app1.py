import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from data_preprocessing import clean_data, columns_encode, standard_data, pca
from model_building_and_clustring import Elbow_method, k_means_cluster
import plotly.express as px

# ---------------- UI Setup ---------------- #
st.set_page_config(page_title="CSV Clustering App", page_icon="📊", layout="wide")

st.title("📊 CSV Upload and Clustering Tool")
st.write("Upload a CSV file, preprocess it, run **KMeans clustering**, and visualize results interactively.")

# ---------------- Tabs ---------------- #
tab_upload, tab_preprocess, tab_cluster, tab_visualize, tab_download = st.tabs(
    ["📂 Upload", "⚙️ Preprocessing", "🎯 Clustering", "📊 Visualization", "📥 Download"]
)

# ---------------- Upload Tab ---------------- #
with tab_upload:
    st.subheader("📂 Upload CSV File")
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        final_data = df.copy()

        st.success("✅ File uploaded successfully!")
        st.write("### Preview of the Data")
        st.dataframe(df.head(), use_container_width=True)

        st.session_state.df_raw = df
        st.session_state.final_data = final_data

# ---------------- Preprocessing Tab ---------------- #
with tab_preprocess:
    if "df_raw" not in st.session_state:
        st.warning("⚠️ Please upload a CSV file first (Go to **Upload** tab).")
    else:
        st.subheader("⚙️ Select Features for Clustering")
        df = st.session_state.df_raw.copy()
        Features = df.columns.to_list()

        feature_list = st.multiselect("Select features to use for clustering:", Features)
        if feature_list:
            df = df[feature_list]
            st.session_state.df_selected = df
            st.success("✅ Features selected!")
            st.dataframe(df.head(), use_container_width=True)
        else:
            st.info("ℹ️ Please select at least one feature.")

# ---------------- Clustering Tab ---------------- #
with tab_cluster:
    if "df_selected" not in st.session_state:
        st.warning("⚠️ Please select features first (Go to **Preprocessing** tab).")
    else:
        df = st.session_state.df_selected.copy()
        st.subheader("🎯 Clustering")

        # Elbow Method
        k = st.slider("Select maximum number of clusters for Elbow Method:", min_value=2, max_value=20, value=6)
        if st.button("📈 Run Elbow Method"):
            df_clean = clean_data(df.columns.to_list(), df)
            df_clean = columns_encode(df.columns.to_list(), df_clean)
            X_std = standard_data(df_clean)

            wcss = Elbow_method(X_std, k)
            fig, ax = plt.subplots()
            ax.plot(range(2, k), wcss, marker='o', linestyle="--")
            ax.set_xlabel("Number of Clusters")
            ax.set_ylabel("WCSS")
            ax.set_title("Elbow Method - Optimal K")
            st.pyplot(fig)

        # Apply KMeans
        k_value = st.number_input("Enter the number of clusters (k):", min_value=2, max_value=20, value=3, step=1)
        if st.button("🚀 Apply Clustering"):
            df_clean = clean_data(df.columns.to_list(), df)
            df_clean = columns_encode(df.columns.to_list(), df_clean)
            X_std = standard_data(df_clean)

            clustered_df = k_means_cluster(df_clean, X_std, int(k_value))
            st.session_state.df = clustered_df
            st.session_state.X_std = X_std
            st.session_state.final_data["clusters"] = clustered_df["clusters"]

            st.success(f"✅ Clustering applied with **k={k_value}**")
            st.dataframe(clustered_df.head(), use_container_width=True)

# ---------------- Visualization Tab ---------------- #
with tab_visualize:
    st.subheader("📊 3D PCA Visualization of Clusters")
    if "df" in st.session_state and "X_std" in st.session_state:
        if st.button("🎨 Show 3D Visualization"):
            x_pca = pca(st.session_state.X_std, 3)

            Features = pd.DataFrame(x_pca, columns=['f1', 'f2', 'f3'])
            Features['cluster'] = st.session_state.df['clusters']

            fig = px.scatter_3d(
                Features,
                x='f1', y='f2', z='f3',
                color=Features['cluster'].astype(str),
                title="KMeans Clusters (3D PCA Reduced)",
                opacity=0.8,
                symbol=Features['cluster'].astype(str),
                hover_data=['f1', 'f2', 'f3', 'cluster']
            )
            fig.update_traces(marker=dict(size=6))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Please apply clustering first (Go to **Clustering** tab).")

# ---------------- Download Tab ---------------- #
with tab_download:
    st.subheader("📥 Download Clustered Data")
    if "final_data" in st.session_state or "clusters" in st.session_state.final_data:
        csv = st.session_state.final_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Clustered Data (CSV)",
            csv,
            "clusters.csv",
            "text/csv",
            use_container_width=True
        )
    else:
        st.info("ℹ️ No clustered data available yet. Run clustering first.")
