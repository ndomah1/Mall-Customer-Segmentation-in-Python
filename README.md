# Mall Customer Segmentation in Python

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Overview](#dataset-overview)
  - [Data Source](#data-source)
- [Goals and Key Questions](#goals-and-key-questions)
- [Data Preprocessing & Exploration](#data-preprocessing--exploration)
  - [Loading the Data](#loading-the-data)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Visualizing Customer Attributes](#visualizing-customer-attributes)
- [Feature Engineering & Preprocessing](#feature-engineering--preprocessing)
- [Clustering Using K-Means](#clustering-using-k-means)
  - [Finding the Optimal Number of Clusters (Elbow Method)](#finding-the-optimal-number-of-clusters-elbow-method)
  - [Applying K-Means Clustering](#applying-k-means-clustering)
- [Cluster Visualization](#cluster-visualization)
  - [2D Visualization](#2d-visualization)
  - [3D Visualization](#3d-visualization)
- [Interpreting the Clusters](#interpreting-the-clusters)
- [Results](#results)
- [Limitations](#limitations)
- [Next Steps](#next-steps)


## **Project Overview**

This project applies **K-Means clustering** to segment customers based on **age, annual income, and spending score**. By grouping customers into clusters, businesses can **identify target customer segments** for marketing strategies.

## **Dataset Overview**

The dataset contains the following key attributes:

- **CustomerID**: Unique identifier for each customer.
- **Gender**: Male or Female.
- **Age**: Customer's age.
- **Annual Income (k$)**: Income in thousands of dollars.
- **Spending Score (1-100)**: Score assigned based on purchasing behavior.

### **Data Source**

The dataset is publicly available: [Customer Segmentation Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

## **Goals and Key Questions**

- How to achieve **customer segmentation** using **K-Means clustering**?
- Who are the **target customers** that businesses can focus on for marketing?
- How does **spending behavior** relate to **income and age**?
- What actionable insights can we derive from the customer segments?

## **Data Preprocessing & Exploration**

### **Loading the Data**

```python
import pandas as pd

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Display first few rows
df.head()
```

|  | CustomerID | Gender | Age | Annual Income *k$) | Spending Score (1-100) |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | Male | 19 | 15 | 39 |
| 1 | 2 | Male | 21 | 15 | 81 |
| 2 | 3 | Female | 20 | 16 | 6 |
| 3 | 4 | Female | 23 | 16 | 77 |
| 4 | 5 | Female | 31 | 17 | 40 |

## **Exploratory Data Analysis (EDA)**

### **Visualizing Customer Attributes**

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use("ggplot")

# Distribution of Age, Income, and Spending Score
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(df["Age"], bins=20, kde=True, ax=axes[0])
axes[0].set_title("Age Distribution")

sns.histplot(df["Annual Income (k$)"], bins=20, kde=True, ax=axes[1])
axes[1].set_title("Annual Income Distribution")

sns.histplot(df["Spending Score (1-100)"], bins=20, kde=True, ax=axes[2])
axes[2].set_title("Spending Score Distribution")

plt.tight_layout()
plt.show()
```

![image.png](image.png)

**Key Findings**: The data is well-distributed, but some customers have extreme spending behaviors.

## **Feature Engineering & Preprocessing**

### **Handling Categorical and Numerical Data**

```python
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Drop CustomerID
df = df.drop(columns=["CustomerID"])

# Encode Gender
encoder = LabelEncoder()
df["Gender"] = encoder.fit_transform(df["Gender"])

# Scale numerical data
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[["Age", "Annual Income (k$)", "Spending Score (1-100)"]] = scaler.fit_transform(
    df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
)

df_scaled.head()
```

|  | Gender | Age | Annual Income (k$) | Spending Score (1-100) |
| --- | --- | --- | --- | --- |
| 0 | 1 | -1.424569 | -1.738999 | -0.434801 |
| 1 | 1 | -1.281035 | -1.738999 | 1.195704 |
| 2 | 0 | -1.352802 | -1.700830 | -1.715913 |
| 3 | 0 | -1.137502 | -1.700830 | 1.040418 |
| 4 | 0 | -0.563369 | -1.662660 | -0.395980 |

**Why This Step?**: Standardization ensures features have equal importance in clustering.

## **Clustering Using K-Means**

### **Finding the Optimal Number of Clusters (Elbow Method)**

```python
from sklearn.cluster import KMeans

# Finding optimal clusters using inertia
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

# Plot elbow method
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker="o", linestyle="-")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.show()
```

![image.png](image%201.png)

**Key Finding**: The **optimal number of clusters (K) is 5**, as observed at the elbow point.

### **Applying K-Means Clustering**

```python
# Train K-Means model
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df_scaled["Cluster"] = kmeans.fit_predict(df_scaled[["Annual Income (k$)", "Spending Score (1-100)"]])

# Assign cluster labels
df["Cluster"] = df_scaled["Cluster"]
```

### **Cluster Visualization**

#### 2D

```python
# Visualize the clusters in a 2D scatter plot using Annual Income and Spending Score
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=df["Annual Income (k$)"], 
    y=df["Spending Score (1-100)"], 
    hue=df["Cluster"], 
    palette="viridis", 
    alpha=0.7
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation using K-Means")
plt.legend(title="Cluster")
plt.show()
```

![image.png](image%202.png)

#### 3D

```python
from mpl_toolkits.mplot3d import Axes3D

# 3D Scatter Plot for better visualization of clusters
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

# Scatter plot
scatter = ax.scatter(
    df["Annual Income (k$)"], 
    df["Spending Score (1-100)"], 
    df["Age"], 
    c=df["Cluster"], 
    cmap="viridis", 
    alpha=0.8
)

# Labels and title
ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
ax.set_zlabel("Age")
ax.set_title("3D Customer Segmentation")

# Add legend
plt.legend(*scatter.legend_elements(), title="Cluster")
plt.show()
```

![image.png](image%203.png)

### **Interpreting the Clusters**:

- **Cluster 0**: High income, high spending – Premium customers.
    - **Business Strategy**: Offer loyalty programs, exclusive discounts, and personalized services to maintain engagement.
- **Cluster 1**: Low income, high spending – Budget-conscious frequent shoppers.
    - **Business Strategy**: Implement targeted promotions, installment payment options, and discounts on frequently purchased products.
- **Cluster 2**: Medium income, medium spending – Average customers.
    - **Business Strategy**: Encourage higher spending through bundle deals and seasonal offers.
- **Cluster 3**: High income, low spending – Price-sensitive customers.
    - **Business Strategy**: Use premium product marketing and upscale shopping experiences to increase spending.
- **Cluster 4**: Low income, low spending – Occasional mall visitors.
    - **Business Strategy**: Provide referral discounts and budget-friendly product lines to increase engagement.

## Results

### Addressing Key Goals & Questions:

1. **How to achieve customer segmentation using machine learning?**
    - Successfully applied K-Means clustering to segment customers into five distinct groups based on spending behavior and income levels.
2. **Who are the target customers for marketing strategies?**
    - The high-income, high-spending group (**Cluster 0**) and budget-conscious high-spenders (**Cluster 1**) are key targets for promotional campaigns.
3. **How does spending behavior relate to income and age?**
    - There is a strong correlation between **income and spending scores**, with younger and mid-aged individuals tending to spend more in certain income brackets.
4. **What actionable insights can we derive?**
    - Marketing teams should focus on **premium shoppers** for retention, while engaging **low-spending high-income customers** with personalized experiences.

## **Limitations**

- **Limited features**: Only considers age, income, and spending.
- **Static dataset**: No time-based behavior tracking.
- **Assumption of K-Means**: Spherical, equally-sized clusters may not always be ideal.

## **Next Steps**

- Experiment with **hierarchical clustering** and **DBSCAN**.
- Incorporate additional features like **purchase frequency** and **loyalty scores**.
- Deploy as a **customer segmentation dashboard** for business insights.
