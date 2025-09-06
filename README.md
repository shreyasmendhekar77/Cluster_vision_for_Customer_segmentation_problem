# 🚀 Cluster Vision

Cluster Vision is an interactive web application built using **Streamlit** that performs customer segmentation using K-Means clustering. The app enables easy feature selection, model evaluation (Elbow Method & Silhouette Score), clustering, and visualization (2D & 3D PCA). It is ideal for business analysts or data scientists looking to perform customer segmentation and visualize insights intuitively.

---

## 🎯 Project Purpose

The primary goal of Cluster Vision is to help businesses segment their customers based on selected features and visualize cluster patterns for actionable insights. This is useful for targeted marketing, customer profiling, and strategic decision-making.

---

## ⚡ Features

- ✅ Upload and preview CSV datasets  
- ✅ Select features for clustering  
- ✅ Evaluate the optimal number of clusters using:
    - Elbow Method (WCSS plot)  
    - Silhouette Score Analysis  
- ✅ Apply K-Means clustering with customizable cluster count  
- ✅ Visualize clusters in:
    - 2D scatter plot (when 2 features selected)  
    - 3D PCA-reduced scatter plot (when >2 features selected)  
- ✅ Download the Elbow/Silhouette plots and clustered dataset  

---

## 🛠️ Tech Stack

- Python  
- [Streamlit](https://streamlit.io) (for the web UI)  
- Pandas (for data manipulation)  
- Matplotlib & Plotly (for plotting)  
- scikit-learn (for clustering & preprocessing)  

---

## 🚀 Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/cluster-vision.git
    cd cluster-vision
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # (Linux/macOS)
    venv\Scripts\activate     # (Windows)
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## ⚙️ Usage

1. Launch the Streamlit app:
    ```bash
    streamlit run app.py
    ```

2. In the web app:
    - Upload your customer dataset as a `.csv` file  
    - Select numeric features to include in clustering  
    - Evaluate the number of clusters with Elbow Method and Silhouette Score  
    - Apply K-Means clustering  
    - Visualize clusters in 2D or 3D  
    - Download clustered data and plots  

---

## 📊 Example Dataset

Ensure your dataset contains numeric customer features such as:
- Age  
- Income  
- Spending Score  
- Account Balance  

Example dataset columns:
```csv
CustomerID,Age,Income,SpendingScore,AccountBalance
1,25,50000,70,1500
2,45,80000,45,2000
...
