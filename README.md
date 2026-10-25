# Customer Segmentation Using Machine Learning

## Project Overview

Customer Segmentation is a machine learning project that groups customers based on their annual income and spending score using the **K-Means Clustering algorithm**.

The goal is to identify different customer groups so businesses can better understand customer behavior, improve marketing strategies, and create targeted offers.

This project uses the Mall Customers dataset and applies feature scaling before performing clustering.

## Objectives

- Analyze customer demographic and spending data.
- Identify customer groups using K-Means clustering.
- Apply StandardScaler to normalize the selected features.
- Visualize customer segments.
- Generate summary reports for business insights.

## Graph Screenshot
**Final Report Graph**

![](https://github.com/mohamud007/Customer_Segementation_App/blob/main/Screenshot%202026-10-24%20222503.png)


## Technologies Used

- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **Machine Learning Algorithm:** K-Means Clustering
- **Development Environment:** Visual Studio Code

## Dataset

The project uses the Mall Customers dataset, which contains customer information such as:

- Customer ID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1–100)

The model uses two features for clustering:

- Annual Income (k$)
- Spending Score (1–100)

**Dataset setup:** Download the Mall Customers dataset from a trusted source, such as the [Kaggle Mall Customer Segmentation Data dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python).

Create a `data` folder in the project directory and place the CSV file inside it with this exact filename:

`data/Mall_Customers.csv`

The dataset is excluded from this Git repository through `.gitignore`, so you must obtain it separately.

## Project Structure

```text
Customer-Segmentation-ML/
│
├── data/
│   └── Mall_Customers.csv
│
├── outputs/
│   ├── final_customer_segmentation.png
│   ├── final_segment_summary.csv
│   └── final_segmented_customers.csv
│
├── final_report.py
├── requirements.txt
├── README.md
└── .gitignore
```

The `.venv` virtual environment is used locally and is not included in the repository.

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/mohamud007/Customer_Segementation_App.git
cd Customer_Segementation_App
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

**Windows Command Prompt:**

```bash
.venv\Scripts\activate
```

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install the required libraries

```bash
python -m pip install -r requirements.txt
```

### 5. Add the dataset

Place `Mall_Customers.csv` inside the `data` folder as explained in the Dataset section.

### 6. Run the project

```bash
python final_report.py
```

The program analyzes the customers, creates five clusters, displays summary statistics, and generates the output files in the `outputs` directory.

## Machine Learning Workflow

1. Load the customer dataset using Pandas.
2. Select annual income and spending score as input features.
3. Scale the features using StandardScaler.
4. Train the K-Means clustering model with five clusters.
5. Assign each customer to a cluster.
6. Calculate summary statistics for each cluster.
7. Generate a visualization and CSV reports.

## Customer Segments

The model identifies five groups. Their exact characteristics are calculated from the dataset and may vary if the data or model settings change.

| Cluster | General Interpretation |
|---|---|
| Cluster 0 | Customers with moderate income and spending |
| Cluster 1 | Customers with high income and high spending |
| Cluster 2 | Customers with low income and high spending |
| Cluster 3 | Customers with high income and low spending |
| Cluster 4 | Customers with low income and low spending |

These descriptions are approximate interpretations of the current results, not fixed labels produced by K-Means.

## Output Files

### 1. `final_customer_segmentation.png`

A visualization of the five customer clusters based on scaled annual income and spending score.

### 2. `final_segment_summary.csv`

Contains the following summary metrics for each cluster:

- Customer count
- Average age
- Average annual income
- Average spending score

### 3. `final_segmented_customers.csv`

Contains the customer records with their assigned cluster labels.

## Business Applications

- Targeted marketing campaigns
- Customer behavior analysis
- Personalized offers and promotions
- Customer retention strategies
- Identification of high-value customer groups

## Future Improvements

- Compare different clustering algorithms.
- Build an interactive dashboard using Streamlit.
- Allow users to upload their own customer datasets.
- Experiment with additional customer features.
- Evaluate cluster quality using appropriate metrics.


*This project was developed for learning and demonstrating data preprocessing, unsupervised machine learning, clustering, data visualization, and customer analytics.*
