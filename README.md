# Uber Fare Prediction Using Spatial-Temporal Modeling

This project focuses on predicting Uber fares using various data features, including geographical and temporal information. We started with raw trip data, cleaned it, engineered new features, performed exploratory data analysis, and finally built and evaluated several machine learning models.

## Data Cleaning & Pre-analysis
*   The initial dataset had 200,000 rows with some missing values and outliers.
*   Missing values were handled by dropping rows (very few).
*   Outliers like zero/negative fares and invalid coordinates were removed.
*   Data was filtered to include only trips within NYC boundaries.
*   Approximately 2.11% of the original rows were dropped during cleaning.

## Feature Engineering
*   **Distance**: Calculated using the Haversine formula (distance_km).
*   **Time-based features**: Extracted hour, day, month, weekday, and year from `pickup_datetime`.
*   **Rush Hour**: A binary indicator for peak commuting times.
*   **Geographical Clusters**: Created `pickup_cluster` and `dropoff_cluster` using K-Means clustering (4 clusters) on coordinates to represent 'neighborhoods'.

## Exploratory Data Analysis (EDA)
*   Most fares are under $20, and most trips are under 5 km (right-skewed distributions).
*   Trip volume peaks during morning (8-10 AM) and evening (5-7 PM) hours.
*   A strong positive correlation exists between `distance_km` and `fare_amount` (~0.82).
*   New `pickup_cluster` and `dropoff_cluster` features helped capture geographical patterns effectively.

## Modeling & Evaluation
We trained and evaluated three models: Linear Regression, Random Forest, and XGBoost. All models were evaluated before and after adding the geographical cluster features.

### Model Performance Comparison

| Model                               | RMSE   | R²     |
| :---------------------------------- | :----- | :----- |
| Linear Regression                   | 5.0760 | 0.7201 |
| Random Forest                       | 4.4892 | 0.7811 |
| XGBoost                             | 4.6103 | 0.7691 |
| **Linear Regression (with clusters)** | **5.0602** | **0.7218** |
| **Random Forest (with clusters)**   | **4.2672** | **0.8022** |
| **XGBoost (with clusters)**         | **4.3653** | **0.7930** |

## Key Takeaways
*   **Distance** is the most significant predictor of Uber fares.
*   **Tree-based models** (Random Forest, XGBoost) consistently outperform Linear Regression, indicating non-linear relationships in the data.
*   Adding **geographical cluster features** significantly improved the performance of all models, especially the tree-based ones.
*   The **Random Forest model with cluster features** achieved the best performance (RMSE ~4.27, R² ~0.80), demonstrating the value of comprehensive feature engineering.
