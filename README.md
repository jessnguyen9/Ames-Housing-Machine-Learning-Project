# Ames Housing - A Price Prediction Model With Machine Learning

## Overview
Conducted a comprehensive analysis using the Ames Housing dataset to develop a machine learning model for predicting house prices. The project involved data acquisition, preparation, analysis, and visualization. Using SQL, Python, and machine learning techniques, we built predictive models and surfaced insights into housing trends and key price determinants. A dashboard in Tableau was also created to visualize trends related to housing attributes, neighborhoods, and pricing patterns.

This repo containst the following:
* data folder: contains the dataset used for modeling
  
* graphs and plots folder: contains screenshots of our graphs, which are also represented in part below
  
* notebooks:
  * K-Means clustering and plots.ipynb: Clustering analysis
    
  * Ames_Modeling.ipynb: Model development and evaluation (Google Colab)
    
  * best_model.pkl: Saved final model

## Data Acquisition
We used the public Ames Housing dataset. This dataset is available on Kaggle and other places around the web.
The data were cleaned and hosted on AWS S3 in a public bucket.

## Data Preparation
Cleaning: Removed missing values and incomplete records

Transformation: Converted categorical variables into numerical formats for modeling

Feature Engineering: Created new features based on existing data to improve model accuracy

Clustering: Applied K-Means clustering to segment housing data into meaningful groups

## Methods
* Jupyter Notebook: Used for Python-based modeling and clustering
* Tableau: Data visualization and analysis
* AWS: Hosting data for easy access
* Python + Flask: Web application development for serving predictions
* HTML, CSS, JS: Created an interactive webpage for user inputs

## Data Analysis
### Tableau Visualizations

https://public.tableau.com/app/profile/mary.hills/viz/AmesAmenitiesComparison/AmenitesComparison
https://public.tableau.com/app/profile/mary.hills/viz/AmesHousingData_16929331848960/NeighborhoodData

### Features: Correlations & Clustering
- Related Features: Sale price showed strong correlations with square footage, total rooms, and overall quality.

- Clustering: A K-Means elbow curve was used to determine the optimal number of clusters for segmentation.

- Visual Analysis: Charts comparing sale price with living area, year built, overall quality, and condition.
  
#### Related Features

We see that there are strong correlations between intuitively related features such as total rooms and square footage.
Our sale price target variable strongly correlates to features related to home size, as well as quality.

![correlation_heatmap_plot](https://github.com/gmitt98/Ames-Housing-ML/assets/11577627/74d74d67-7b1b-4757-a3b0-b0ae434b3cb1)

#### Clustering: Cluster Count

We created a k-means elbow curve to find the "right" number of clusters to use in our analysis

![image](https://github.com/gmitt98/Ames-Housing-ML/assets/11577627/d8fa5e81-b507-4c6e-97a2-45f3dfb8368a)

#### Features that matter

Next, we can see how well our clustering works by visually inspecting the groupings

![Sale Price vs LivingArea](https://github.com/gmitt98/Ames-Housing-ML/assets/11577627/fda879a4-f660-411f-b395-a40fb184e5f7)

![Sale Price vs Year Built](https://github.com/gmitt98/Ames-Housing-ML/assets/11577627/f14f47ae-bdbd-4b65-92b0-bb3f9c4db6c3)

![Sale Price vs Overall Quality](https://github.com/gmitt98/Ames-Housing-ML/assets/11577627/6e3050c3-fec4-4a43-b9f6-bde9301f6824)

![Sale Price vs Overall Condition](https://github.com/gmitt98/Ames-Housing-ML/assets/11577627/5a4ffc5c-20ad-4dc6-847c-211655788533)

#### Features that might not matter

![Mean Sale Price vs Neighborhood](https://github.com/gmitt98/Ames-Housing-ML/assets/11577627/9a58bb26-c49d-4527-88b0-9960e7596337)

![Mean Sale Price vs House Type](https://github.com/gmitt98/Ames-Housing-ML/assets/11577627/0a5d181d-f425-4760-a244-ef2ff6d209a6)

## Predictions: Machine Learning Model

### Models

Two models were run: a linear regression model to establish a basline, and a Gradient Boost model to try and improve on the baseline.
The linear regression model performed well, with the following results:

* Training R-Squared: 0.9401
* Testing R-Squared: 0.9397
* Mean Squared Error: 0.0695

Residuals on the model were thus:


![image](https://github.com/gmitt98/Ames-Housing-ML/assets/11577627/3d3991b7-f9a5-4ba6-a77d-6c0e096e8d79)


Gradient boosting slightly improved the results. We used cross-validation to determine the best hyperparamters.

* Training R-Squared: 0.9676501961674518
* Testing R-Squared: 0.9501703258657423
* Mean Squared Error: 0.05743137013095087

Residuals on the gradient boost model were:

![image](https://github.com/gmitt98/Ames-Housing-ML/assets/11577627/8a3f9b03-0f9f-44ee-a5a5-6c8cc441e5ac)

### Feature Importances

The model is dominated by features related to size and quality

|Feature Names|Feature Importances|
|---|---|
|Overall Quality|0\.39272483495402866|
|Total Square Footage|0\.3568763700410612|
|Year Built|0\.045396744133224295|
|Year Remodeled/Added|0\.022098346479286855|
|Ground Living Area|0\.0193034148646336|
|Kitchen Quality\_TA|0\.017791286766306636|
|Garage Cars|0\.016039563131456474|
|BsmtFin Square Footage 1|0\.012385942230966514|
|Garage Area|0\.011932867284658226|
|Lot Area|0\.011541017959086157|
|Fireplaces|0\.009578511945560421|
|Overall Condition|0\.007511069312884516|

Additionally, we re-ran the model with CatBoost to use with model inferencing, and CatBoost gives us the following importances:

![image](https://github.com/gmitt98/Ames-Housing-ML/assets/11577627/87e83e14-2b1e-46a8-ba84-cb8b3da45fd2)

## Challenges
- Handling missing data without introducing bias.

- Selecting the best features while avoiding multicollinearity.

- Optimizing model performance without overfitting.

## Future Improvements
- Implementing more advanced models like XGBoost for better accuracy.

- Improving feature engineering through interaction terms.

- Enhancing the web application with real-time predictions.

## Conclusion
This project demonstrates end-to-end data processing, from acquisition to predictive modeling. By integrating SQL, Python, Flask, and machine learning, we created an interactive tool that helps analyze housing trends and predict property prices accurately.
