import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import io

# --- Page Config ---
st.set_page_config(page_title="📦 SmartWarehouse", layout="wide")

# --- Set Background Image ---
def set_bg(image_url):
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("{image_url}");
        background-attachment: fixed;
        background-size: cover;
    }}        
    </style>
    """, unsafe_allow_html=True)

# --- Styled Title ---
st.markdown(
    """
    <div style='background-color: white; padding: 10px; border-radius: 50px; text-align: center;'>
        <h1 style='color: black; font-weight: bold;'>
            📦📦 SmartWarehouse: Real-Time Product Clustering, Insights & Recommendation System
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)
set_bg("https://i.postimg.cc/WbKdjsZK/Untitled-design-2.png")
    
# --- ProductVisualizer Class ---
class ProductVisualizer:
    def __init__(self, category_columns):
        self.category_columns = category_columns

    def reconstruct_category(self, df):
        df = df.copy()
        df['Product Category'] = df[self.category_columns].idxmax(axis=1).str.replace("Product Category_", "")
        return df

    def plot_insights(self, df):
        df = self.reconstruct_category(df)

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle("Product Analysis Visualizations", fontsize=16)

        # 1. Pie Chart - Product Category Distribution
        category_counts = df['Product Category'].value_counts()
        axes[0].pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)
        axes[0].set_title('Product Category Distribution')
        axes[0].axis('equal')

        # 2. Line Chart - Avg Daily Demand by Product Category
        sns.lineplot(data=df, x='Product Category', y='Avg Daily Demand', estimator='mean', ax=axes[1])
        axes[1].set_title('Avg Daily Demand by Product Category')
        axes[1].set_xlabel('Product Category')
        axes[1].set_ylabel('Avg Daily Demand')

        # 3. Histogram - Product Dimensions
        df[['Height (cm)', 'Width (cm)', 'Depth (cm)']].plot(kind='hist', bins=20, alpha=0.6, ax=axes[2])
        axes[2].set_title('Distribution of Product Dimensions')
        axes[2].set_xlabel('Size (cm)')
        axes[2].set_ylabel('Frequency')
        axes[2].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        st.pyplot(plt)

# --- Upload Section ---
st.subheader("📁 Upload your product CSV file")
uploaded_file = st.file_uploader("", type=["csv"])

if uploaded_file:
    set_bg("https://i.postimg.cc/cCbVK9Mx/Untitled-design-1.png")
    # Step 1: Read uploaded data
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Uploaded Data Sample")
    st.dataframe(df.head())

    # Step 2: Clean and preprocess data
    with st.spinner("🔄 Cleaning and preprocessing data..."):
        float_cols = ['Height (cm)', 'Width (cm)', 'Depth (cm)', 'Avg Daily Demand']

        if 'Product Category' not in df.columns:
            st.error("❌ 'Product Category' column is required in uploaded data.")
            st.stop()

        # Fill missing values
        df[float_cols] = df.groupby('Product Category')[float_cols].transform(lambda x: x.fillna(x.mean()))

        # Drop irrelevant columns
        if 'Product ID' in df.columns:
            df = df.drop(columns="Product ID")
        if 'Score' in df.columns:
            df = df.drop(columns="Score")

        # One-hot encode category
        df = pd.get_dummies(df, columns=['Product Category']).astype(float)

        st.success("✅ Data cleaned and preprocessed.")

    # Step 3: Load Pickled Models
    try:
        with open("/Users/somesh-19583/Desktop/Warehouse/Pickle/Scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        with open("/Users/somesh-19583/Desktop/Warehouse/Pickle/kmeans_model.pkl", "rb") as f:
            kmeans = pickle.load(f)

        with open("/Users/somesh-19583/Desktop/Warehouse/Pickle/pca.pkl", "rb") as f:
            pca = pickle.load(f)

        with open("/Users/somesh-19583/Desktop/Warehouse/Pickle/processed_product_analysis_df.pkl", "rb") as f:
            visualizer = pickle.load(f)

        with open("/Users/somesh-19583/Desktop/Warehouse/Pickle/cluster_recommendations_all.pkl", "rb") as f:
            data = pickle.load(f)

    except FileNotFoundError as e:
        st.error(f"❌ Missing pickle file: {e}")
        st.stop()

    # Step 4: Apply Scaler, Clustering & PCA
    with st.spinner("🔍 Applying clustering and dimensionality reduction..."):
        df_scaled = scaler.transform(df)
        cluster_labels = kmeans.predict(df_scaled)
        pca_components = pca.transform(df_scaled)

        df_result = pd.DataFrame(df_scaled, columns=df.columns)
        df_result["Cluster"] = cluster_labels
        df_result["PC1"] = pca_components[:, 0]
        df_result["PC2"] = pca_components[:, 1]

        # st.success("✅ Clustering and PCA applied.")
        # st.subheader("📊 Clustered Data Preview")
        # st.dataframe(df_result.head())

    # Step 5: Visualize and Show Recommendations
    if st.button("📊 Visualize Insights"):
        st.subheader("📈 Product Category Insights")
        try:
            visualizer.plot_insights(df_result)

            st.subheader("🧭 PCA Scatter Plot of Clusters")
            plt.figure(figsize=(12, 4))
            sns.scatterplot(data=df_result, x="PC1", y="PC2", hue="Cluster", palette="viridis")
            plt.title("Cluster Visualization using PCA")
            plt.grid(True)
            st.pyplot(plt)

            st.subheader("💡 Cluster-Based Business Recommendations")
            st.write("Strategic warehouse suggestions derived from product clusters:")
            # Correctly unpack the individual DataFrames
            category_dist_df = data["category_distribution"]
            cluster_profile_df = data["cluster_profiles"]
            recommendation_df = data["recommendations"]

            st.success("✅ Pickle file loaded successfully!")

            # Display Category Distribution
            st.subheader("📊 Category Distribution by Cluster")
            st.dataframe(pd.DataFrame(category_dist_df))

            # Display Cluster Feature Profiles
            st.subheader("📌 Cluster Feature Profiles (Averages)")
            st.dataframe(pd.DataFrame(cluster_profile_df))

            # Display Business Recommendations
            st.subheader("🚀 Business Recommendations")
            st.dataframe(pd.DataFrame(recommendation_df))


            # Download button
            buffer = io.BytesIO()
            df.to_excel(buffer, index=False)
            st.download_button(
                label="📥 Download Recommendations as Excel",
                data=buffer,
                file_name="cluster_recommendations.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e:
            st.error(f"❌ Visualization error: {e}")

else:
    st.info("👆 Please upload a CSV file to begin.")
