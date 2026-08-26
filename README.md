# House Price Prediction Using Machine Learning

## Project Overview

This project builds an end-to-end Machine Learning pipeline to predict residential house prices in Ames, Iowa. Using the Kaggle House Prices dataset containing 1,460 property records with 79 structural features, we train and compare four regression models to find the most accurate property valuation system.

**Internship**: AI & Data Science Internship - DataSpy Technologies

---

## Business Problem

Accurate property valuation is critical for home buyers, sellers, real estate agents, and financial institutions. Traditional appraisals are subjective, slow, and expensive. This project replaces manual methods with a data-driven machine learning approach that analyzes structural, quality, and geographic features to estimate fair market prices.

---

## Objectives

- Perform comprehensive Exploratory Data Analysis (EDA) on the Ames Housing dataset
- Build a leakage-safe preprocessing pipeline using scikit-learn
- Train and compare four regression models
- Select the best model based on actual evaluation metrics
- Deploy a prediction system that works on new, unseen properties

---

## Dataset

**Source**: [Kaggle House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

| File | Records | Features |
|------|---------|----------|
| train.csv | 1,460 | 81 (including SalePrice) |
| test.csv | 1,459 | 80 (no SalePrice) |
| data_description.txt | - | Data dictionary |

---

## Dataset Features

The dataset contains 79 explanatory variables describing physical, quality, and geographic attributes of residential homes:

- **Size Features**: GrLivArea, TotalBsmtSF, 1stFlrSF, 2ndFlrSF, GarageArea, LotArea
- **Quality Ratings**: OverallQual, OverallCond, ExterQual, KitchenQual
- **Age Features**: YearBuilt, YearRemodAdd, GarageYrBlt
- **Location**: Neighborhood (25 neighborhoods in Ames, Iowa)
- **Structural**: BldgType, HouseStyle, Foundation, RoofStyle
- **Amenities**: Fireplaces, PoolArea, Fence, CentralAir

---

## Technology Stack

| Technology | Purpose |
|------------|---------|
| Python 3.12+ | Core language |
| Jupyter Notebook | Interactive analysis |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Matplotlib | Static visualizations |
| Seaborn | Statistical visualizations |
| Scikit-learn | ML pipelines, preprocessing, models, evaluation |
| XGBoost | Gradient boosting regressor |
| Joblib | Model serialization |

---

## Machine Learning Workflow

```
Dataset -> Data Cleaning -> EDA -> Feature Engineering -> Preprocessing
-> Train/Test Split -> Model Training -> Evaluation -> Cross Validation
-> Model Comparison -> Feature Importance -> Final Model Selection
-> Model Saving -> Prediction -> Testing
```

---

## Architecture

The project uses scikit-learn's `Pipeline` and `ColumnTransformer` to create a leakage-safe, reproducible ML pipeline:

```
Input Data
    |
    +-- Numerical Features --> SimpleImputer(median) -> StandardScaler
    |
    +-- Categorical Features -> SimpleImputer(mode) -> OneHotEncoder
    |
    +-- Engineered Features --> TotalSF, TotalBath, HouseAge,
                                RemodeledAge, TotalPorchArea
    |
    v
ColumnTransformer (fitted on training data only)
    |
    v
Regression Model (Linear / Tree / Forest / XGBoost)
    |
    v
np.expm1(prediction) -> Predicted Price ($)
```

---

## Project Structure

```
HousePricePrediction/
|
+-- dataset/
|   +-- train.csv                 # Training data (1,460 rows)
|   +-- test.csv                  # Test data (1,459 rows)
|   +-- data_description.txt      # Feature dictionary
|
+-- graphs/                       # All generated visualizations (43 PNGs)
|
+-- House_Price_Prediction.ipynb  # Complete Jupyter Notebook (23 sections)
|
+-- predict.py                    # CLI prediction script
|
+-- house_price_model.pkl         # Saved sklearn Pipeline model
|
+-- sample_predictions.csv        # 5 sample prediction results
|
+-- requirements.txt              # Python dependencies
|
+-- README.md                     # Project documentation
```

---

## Installation

```bash
# Clone or navigate to the project directory
cd HousePricePrediction

# Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### Jupyter Notebook
```bash
jupyter notebook House_Price_Prediction.ipynb
```
Run all cells sequentially from top to bottom.

### Model Training (Standalone)
```bash
python build_complete_pipeline.py
```

### Prediction (Interactive CLI)
```bash
python predict.py
```

### Prediction (Test Mode)
```bash
python predict.py --test
```

---

## Jupyter Notebook

The notebook contains 23 well-documented sections covering:

1. Introduction
2. Problem Statement
3. Business Scenario
4. Dataset Description
5. Import Libraries
6. Data Loading
7. Data Understanding
8. Data Cleaning
9. Exploratory Data Analysis
10. Feature Engineering
11. Data Preprocessing
12. Train-Test Split
13. Model Training
14. Model Evaluation
15. Cross Validation
16. Model Comparison
17. Feature Importance
18. Residual Analysis
19. Final Model Selection
20. Model Saving
21. New House Prediction
22. Sample Predictions
23. Conclusion

---

## Model Training

Four baseline, regularized, tuned, and ensemble regression models were trained on 1,166 records (80% split) with log-transformed target variable:

| Model | Configuration / Tuning |
|-------|------------------------|
| Linear Regression | Default baseline |
| Ridge (Tuned) | `GridSearchCV(alpha=10.0)` |
| Lasso (Tuned) | `GridSearchCV(alpha=0.0005)` |
| Decision Tree | random_state=42 |
| Random Forest (Tuned) | n_estimators=200, max_depth=20 |
| XGBoost (Tuned) | n_estimators=250, learning_rate=0.05, max_depth=4 |
| **Stacking Ensemble** | **Base: Ridge, Lasso, RF, XGBoost | Meta-Learner: Ridge(alpha=10.0)** |

---

## Model Evaluation

### Test Set Performance (Original Scale - USD)

| Model | MAE ($) | RMSE ($) | R^2 Score |
|-------|---------|----------|-----------|
| **Stacking Ensemble** | **$13,537.53** | **$18,828.29** | **0.9358** |
| Lasso (Tuned) | $14,009.81 | $19,420.38 | 0.9317 |
| XGBoost (Tuned) | $14,304.45 | $19,750.95 | 0.9294 |
| Ridge (Tuned) | $14,325.67 | $19,844.96 | 0.9287 |
| Linear Regression | $15,272.92 | $21,545.81 | 0.9160 |
| Random Forest (Tuned) | $16,496.65 | $23,733.16 | 0.8980 |
| Decision Tree | $29,453.04 | $43,845.24 | 0.6520 |

---

## Actual Results

### Cross Validation (Log-Scale RMSE)

| Model | Mean CV RMSE | Std Dev |
|-------|-------------|---------|
| **Stacking Ensemble** | **0.1127** | **0.0101** |
| Lasso (Tuned) | 0.1139 | 0.0109 |
| Ridge (Tuned) | 0.1144 | 0.0097 |
| XGBoost (Tuned) | 0.1237 | 0.0057 |
| Linear Regression | 0.1302 | 0.0096 |
| Random Forest (Tuned) | 0.1375 | 0.0109 |
| Decision Tree | 0.2006 | 0.0104 |

---

## Model Comparison

### Analysis

- **Stacking Ensemble** achieves the best overall performance with test RMSE ($18,828.29), highest R² (0.9358), and lowest cross-validation error (0.1127).
- **Lasso & Ridge Regularization** effectively handle high-dimensional one-hot encoded features, outperforming standard Linear Regression by over $1,700 in RMSE.
- **XGBoost (Tuned)** achieves exceptional CV stability (std dev 0.0057) and strong non-linear feature capture.
- **Random Forest** offers strong feature importance interpretability.
- **Decision Tree** demonstrates unpruned tree variance.

---

## Final Model

**Selected Model: Stacking Ensemble Regressor**

The Stacking Ensemble model was selected based on:
- Best test RMSE: $18,828.29 (2,717 USD improvement over initial baseline)
- Highest R^2 score: 0.9358 (captures 93.58% of price variance)
- Superior CV score: 0.1127 (best overall generalizability across validation folds)
- Meta-learner stability combining regularized linear and tree boosting models

---

## Implemented Enhancements

All key roadmap future updates have been fully executed:

- ✅ **Hyperparameter Tuning**: Integrated automated `GridSearchCV` optimization for Ridge, Lasso, Random Forest, and XGBoost regressors.
- ✅ **Regularization**: Implemented L1 (Lasso) and L2 (Ridge) regression models to control multicollinearity in the feature space.
- ✅ **Stacking Ensemble**: Built a meta-learning `StackingRegressor` combining base regularized & tree models with a Ridge meta-learner.
- ✅ **Interactive Web Application**: Updated `index.html` glassmorphic valuation dashboard with live model selection, interactive sliders, dynamic pricing, and city market scaling.
- ✅ **Geographic Expansion**: Implemented multi-city real estate market adjustment factors (Ames, Des Moines, Chicago, Minneapolis, Denver, Austin, Seattle) in both `predict.py` CLI and web interface.

## Prediction

### Sample Predictions

| Case | Actual Price | Predicted Price | Absolute Error | Percentage Error |
|------|-------------|-----------------|----------------|------------------|
| Case 1 | $190,000 | $225,196 | $35,196 | 18.52% |
| Case 2 | $100,000 | $100,125 | $125 | 0.12% |
| Case 3 | $115,000 | $99,501 | $15,499 | 13.48% |
| Case 4 | $159,000 | $164,302 | $5,302 | 3.33% |
| Case 5 | $315,500 | $315,818 | $318 | 0.10% |

---

## Graphs

The project generates 43 high-resolution (300 DPI) visualizations:

### EDA Graphs
- SalePrice distribution (histogram + KDE + boxplot)
- Correlation heatmap (top 12 features)
- Missing values bar chart
- Categorical analysis (Neighborhood, HouseStyle, OverallQual, GarageType, KitchenQual)
- Scatter plots with trendlines (GrLivArea, GarageArea, YearBuilt, TotalBsmtSF)
- Outlier detection plots
- Numerical distribution & skewness analysis
- Feature importance preview

### Model Evaluation Graphs
- Model MAE comparison
- Model RMSE comparison
- Model R^2 comparison
- Cross validation comparison
- Actual vs Predicted (all 4 models)
- Residual analysis
- Prediction error distribution
- Feature importance (Random Forest & XGBoost)
- Sample predictions comparison
- Project workflow diagram

---

## Business Applications

- **Real Estate Platforms**: Automated property valuation for listings
- **Banks & Lenders**: Mortgage risk assessment and loan-to-value calculations
- **Insurance Companies**: Property replacement cost estimation
- **Homeowners**: Fair market value assessment before selling
- **Investors**: Identifying undervalued properties in target neighborhoods

---

## Advantages

- Leakage-safe preprocessing using sklearn Pipeline and ColumnTransformer
- Domain-specific feature engineering (TotalSF, TotalBath, HouseAge)
- Log-transformed target variable for stable regression optimization
- Comprehensive model comparison with cross-validation
- Production-ready prediction script with interactive CLI
- Reproducible results using fixed random_state=42

---

## Limitations

- Dataset is limited to Ames, Iowa - may not generalize to other cities
- Model does not account for macroeconomic factors (interest rates, housing market trends)
- No hyperparameter tuning was performed (Grid Search / Bayesian Optimization)
- Some feature interactions may not be fully captured by linear models
- Property condition and aesthetic appeal are difficult to quantify

---

## Future Enhancements

- **Hyperparameter Tuning**: GridSearchCV or Optuna for optimal parameters
- **Regularization**: Ridge/Lasso regression to handle multicollinearity
- **Stacking Ensemble**: Combine multiple models for improved accuracy
- **Web Application**: Deploy using Streamlit or Flask for browser-based predictions
- **Geographic Expansion**: Train on datasets from multiple cities

---

## Learning Outcomes

Through this project, the following skills were demonstrated:

1. End-to-end Machine Learning lifecycle management
2. Professional Exploratory Data Analysis with visualizations
3. Leakage-safe data preprocessing using sklearn Pipeline
4. Feature engineering based on domain knowledge
5. Model training, evaluation, and comparison
6. Cross-validation for generalization assessment
7. Model serialization and deployment
8. Command-line prediction interface development
9. Technical documentation and code organization

---

## Author

**Darsh**  
AI & Data Science Intern  
DataSpy Technologies

---

## License

This project is developed for educational and internship purposes.
