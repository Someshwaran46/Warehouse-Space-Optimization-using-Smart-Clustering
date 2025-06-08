
# 📦 Warehouse Space Optimization using Smart Clustering

## 🔍 Project Overview

Modern warehouses manage diverse inventories with significant variation in size, shape, and demand frequency. This project addresses inefficiencies in traditional storage systems by using unsupervised machine learning to dynamically group products for optimal warehouse layout and retrieval speed.

---

## 🎯 Problem Statement

Warehouses often struggle with:

* Inefficient use of space due to inconsistent product sizes.
* Increased order retrieval times due to poor layout planning.
* Static storage systems that don't adapt to changes in product demand.

**Objective:**
Use clustering algorithms to group products based on their **physical dimensions** (Height, Width, Depth) and **Average Daily Demand**, enabling:

* Optimal storage zone assignment.
* Improved picking/packing speed.
* Minimized space wastage.

---

## 🧠 Skills Gained

* 🧩 Unsupervised Learning (Clustering)
* 📊 Handling Mixed Data (Numerical + Categorical)
* 🧼 Data Cleaning & Preprocessing
* 🚚 Real-world ML application in Logistics & Supply Chain

---

## 🌐 Domain

* Supply Chain Optimization
* Industrial AI
* IoT in Logistics

---

## 🧰 Business Use Cases

* **Automated Storage Systems:** Efficient shelving of high-demand items.
* **Warehouse Layout Design:** Data-driven space allocation.
* **Inventory Segmentation:** Grouping by movement and shape profiles.
* **Smart Robotics:** Faster navigation and reduced retrieval time.

---

## 🛠️ Approach

1. **Data Cleaning & Imputation:**
   Handle missing values, outliers, and inconsistent types.

2. **Feature Scaling:**
   Normalize product dimensions and demand for clustering.

3. **Clustering:**
   Apply **K-Means** to group similar products.

4. **Dimensionality Reduction:**
   Use **PCA** and **t-SNE** for cluster visualization.

5. **Cluster Profiling:**
   Analyze cluster characteristics to suggest storage zones.

---

## 📈 Results

* ✅ Identified meaningful product clusters.
* 📊 Visualized clusters based on physical size and demand.
* 🗂️ Suggested storage strategies per cluster.
* 🚀 Business-ready recommendations for layout optimization.

---

## 💻 Tech Stack

* Python
* Scikit-learn
* Pandas & NumPy
* Matplotlib & Seaborn
* PCA & t-SNE
* Streamlit *(optional hosting)*

---

## 📂 Repository Structure

```bash
Warehouse-Space-Optimization-using-Smart-Clustering/
├── Data/
│   ├── Source Data.csv
│   ├── Cleaned Data.csv
│   ├── Preprocessed Data.csv
│   └── Final Data.csv
│
├── Notebook/
│   ├── 1. Data Cleaning.ipynb
│   ├── 2. Data Preprocessing.ipynb
│   ├── 3. EDA.ipynb
│   ├── 4. Clustering.ipynb
│   └── app.py # Streamlit application
│
├── Pickle/
│   ├── Scaler.pkl
│   ├── kmeans_model.pkl
│   ├── pca.pkl
│   ├── processed_product_analysis_df.pkl
│   └── product_visualizer.pkl
│                 
├── README.md
└── requirements.txt

```

---

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/Someshwaran46/Warehouse-Space-Optimization-using-Smart-Clustering.git
cd Warehouse-Space-Optimization-using-Smart-Clustering
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3.  Launch Streamlit app:

```bash
streamlit run app/streamlit_app.py
```

---

## 📬 Feedback

- Feel free to open issues or submit pull requests! Improvements, and suggestions are always welcome 🙌.
- For clarifications drop an email to somesh4602@gmail.com.
---
