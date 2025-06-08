import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


# --- Page Config ---
st.set_page_config(page_title="SmartWarehouse", layout="wide")

# --- Styled Title ---
st.markdown(
    """
    <div style='background-color: white; padding: 10px; border-radius: 50px; text-align: center;'>
        <h1 style='color: black; font-weight: bold;'>
            📦 SmartWarehouse: Real-Time Product Clustering, Insights & Recommendation System
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)


def set_bg(image_url):
    st.markdown(f"""
    <style>
    .stApp {{
            background-image: url("{image_url}");
            background-attachment: fixed;
            background-size: cover;
            }}        
    </style>
    """,
    unsafe_allow_html=True)
image_url = "https://i.postimg.cc/85z6S1wD/Untitled-design-1.png1"
set_bg(image_url)

# --- Define ProductVisualizer class BEFORE unpickling ---
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

        # 3. Histogram - Distribution of Product Dimensions
        df[['Height (cm)', 'Width (cm)', 'Depth (cm)']].plot(kind='hist', bins=20, alpha=0.6, ax=axes[2])
        axes[2].set_title('Distribution of Product Dimensions')
        axes[2].set_xlabel('Size (cm)')
        axes[2].set_ylabel('Frequency')
        axes[2].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        st.pyplot(plt)

# --- Upload section ---
uploaded_file = st.file_uploader("Upload your product CSV file", type=["csv"])

if uploaded_file:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)
    st.subheader("Original Uploaded Data")
    st.dataframe(df.head())

    # --- Data Cleaning ---
    with st.spinner("Cleaning and preprocessing data..."):
        float_cols = ['Height (cm)', 'Width (cm)', 'Depth (cm)', 'Avg Daily Demand']

        if 'Product Category' not in df.columns:
            st.error("The input data must contain a 'Product Category' column.")
            st.stop()

        # 1. Fill missing float values grouped by category
        df[float_cols] = df.groupby('Product Category')[float_cols].transform(
            lambda x: x.fillna(x.mean())
        )

        # 2. Drop 'Product ID' if present
        if 'Product ID' in df.columns:
            df = df.drop(columns="Product ID")

        # 3. One-hot encode 'Product Category'
        df = pd.get_dummies(df, columns=['Product Category']).astype(float)

        # 4. Drop 'Score' column if present
        if 'Score' in df.columns:
            df = df.drop(columns="Score")

        st.success("✅ Data cleaned and preprocessed.")

    # --- Load Pickled Models ---
    try:
        with open("/Users/somesh-19583/Desktop/Warehouse/Pickle/Scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        with open("/Users/somesh-19583/Desktop/Warehouse/Pickle/kmeans_model.pkl", "rb") as f:
            kmeans = pickle.load(f)

        with open("/Users/somesh-19583/Desktop/Warehouse/Pickle/pca.pkl", "rb") as f:
            pca = pickle.load(f)

        with open("/Users/somesh-19583/Desktop/Warehouse/Pickle/product_visualizer.pkl", "rb") as f:
            visualizer = pickle.load(f)

    except FileNotFoundError as e:
        st.error(f"Missing pickle file: {e}")
        st.stop()

    # --- Scaling, Clustering, PCA ---
    with st.spinner("Applying clustering and dimensionality reduction..."):
        df_scaled = scaler.transform(df)
        cluster_labels = kmeans.predict(df_scaled)
        pca_components = pca.transform(df_scaled)

        df_result = pd.DataFrame(df_scaled, columns=df.columns)
        df_result["Cluster"] = cluster_labels
        df_result["PC1"] = pca_components[:, 0]
        df_result["PC2"] = pca_components[:, 1]

        # st.success("✅ Clustering and PCA applied.")
        # st.subheader("Processed DataFrame with Cluster & PCA")
        # st.dataframe(df_result.head())

    # --- Visualize Button ---
    if st.button("📊 Visualize Insights"):
        st.subheader("📈 Product Category Analysis")
        st.write("Based on reconstructed product categories from real-time input.")
        try:
            visualizer.plot_insights(df_result)
            # --- PCA Plot ---
            st.subheader("🧭 PCA Scatter Plot of Clusters")
            plt.figure(figsize=(12,4))
            sns.scatterplot(data=df_result, x="PC1", y="PC2", hue="Cluster", palette="viridis")
            plt.title("Cluster Visualization using PCA")
            plt.grid(True)
            st.pyplot(plt)

        except Exception as e:
            st.error(f"Visualization error: {e}")

    

else:
    st.info("👆 Please upload a CSV file to begin.")
