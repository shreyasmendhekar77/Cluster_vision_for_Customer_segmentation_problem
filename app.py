import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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
    st.write("Your selected features are - ")
    for feature in feature_list:
        st.write(feature)
    print(feature_list)
    st.button("Perform Clustering")
    

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
