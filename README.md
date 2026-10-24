# Customer Segmentation Using Machine Learning

## Project Overview

This project uses unsupervised machine learning to group mall customers based on their annual income and spending score. The goal is to identify customer segments that can help businesses understand customer behavior and plan targeted marketing strategies.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- K-Means Clustering
- StandardScaler

## Dataset

The Mall Customers dataset contains customer information such as:

- Customer ID
- Gender
- Age
- Annual Income
- Spending Score

The model uses Annual Income and Spending Score as its clustering features.

## Project Workflow

1. Load the customer dataset using Pandas.
2. Inspect the data and check for missing values.
3. Explore the data using visualizations.
4. Select the features used for clustering.
5. Scale the features using StandardScaler.
6. Apply K-Means Clustering.
7. Explore the number of clusters using the Elbow Method.
8. Analyze cluster statistics and customer segments.
9. Export the results and visualizations.

## Project Structure

```text
Customer-Segmentation-ML/
├── data/
│   └── Mall_Customers.csv
├── notebooks/
├── outputs/
├── load_data.py
├── visualize_data.py
├── prepare_data.py
├── customer_segmentation.py
├── analyze_clusters.py
├── find_optimal_clusters.py
├── scaled_segmentation.py
├── segment_insights.py
├── final_report.py
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run the Project

### 1. Install Python

Install Python 3.13 or another compatible Python version.

### 2. Create and activate a virtual environment

On Windows:

```bash
py -3.13 -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the final report

```bash
python final_report.py
```

## Output Files

The project generates these results in the `outputs/` directory:

- `final_segmented_customers.csv` — customer records with assigned cluster numbers.
- `final_segment_summary.csv` — summary statistics for each cluster.
- `final_customer_segmentation.png` — final customer segmentation visualization.

Additional scripts generate exploratory graphs, the Elbow Method graph, and other analysis files.

## Machine Learning Approach

K-Means Clustering groups customers according to similarities in their selected features. StandardScaler standardizes the features before model training.

The current model uses five clusters. The Elbow Method can help evaluate whether this is a reasonable choice.

## Business Applications

Customer segmentation can help businesses:

- Understand different customer spending patterns.
- Identify groups with higher or lower average spending.
- Design targeted marketing campaigns.
- Support customer engagement and retention strategies.

## Future Improvements

- Evaluate clustering quality using the Silhouette Score.
- Compare different numbers of clusters.
- Add an interactive dashboard.
- Experiment with additional customer features.
- Explore alternative clustering algorithms.

## Disclaimer

This project is intended for educational purposes. The identified customer groups are exploratory and should be validated before making real business decisions.
