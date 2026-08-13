import os
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.patches as patches

sns.set_theme(style='whitegrid')

# Check and create directory for graphs
os.makedirs('graphs', exist_ok=True)

# 1. DATA VALIDATION
print("1. DATA VALIDATION")
try:
    df = pd.read_csv('dataset/train.csv')
    print(f"Shape of train.csv: {df.shape}")
    print(f"Columns: {df.shape[1]}")
    if 'SalePrice' in df.columns:
        print("SalePrice column exists.")
    else:
        print("SalePrice not found.")
        
    print("\nData Types:")
    print(df.dtypes.value_counts())
    print(f"\nDuplicates: {df.duplicated().sum()}")
    print(f"Missing Values (total): {df.isna().sum().sum()}")
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# 2. PREPROCESSING (leakage-safe)
print("\n2. PREPROCESSING")
# Remove 2 outliers
initial_shape = df.shape
df = df.drop(df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)].index)
print(f"Removed {initial_shape[0] - df.shape[0]} outliers.")

# Feature engineering BEFORE Pipeline
df['TotalSF'] = df.get('TotalBsmtSF', 0) + df.get('1stFlrSF', 0) + df.get('2ndFlrSF', 0)
df['TotalBath'] = df.get('FullBath', 0) + 0.5 * df.get('HalfBath', 0) + df.get('BsmtFullBath', 0) + 0.5 * df.get('BsmtHalfBath', 0)
df['HouseAge'] = df['YrSold'] - df['YearBuilt']
df['RemodeledAge'] = df['YrSold'] - df['YearRemodAdd']
df['TotalPorchArea'] = df.get('OpenPorchSF', 0) + df.get('EnclosedPorch', 0) + df.get('3SsnPorch', 0) + df.get('ScreenPorch', 0)

# Drop Id
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

# Log-transform target
X = df.drop('SalePrice', axis=1)
y = np.log1p(df['SalePrice'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numerical and categorical columns
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns

# ColumnTransformer
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 3. MODEL TRAINING
print("\n3. MODEL TRAINING")
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=300, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=400, learning_rate=0.05, random_state=42, verbosity=0)
}

pipelines = {}
for name, model in models.items():
    pipelines[name] = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('model', model)])
    print(f"Training {name}...")
    pipelines[name].fit(X_train, y_train)

# 4. MODEL EVALUATION
print("\n4. MODEL EVALUATION")
results_list = []
predictions = {}

for name, pipeline in pipelines.items():
    preds_log = pipeline.predict(X_test)
    preds = np.expm1(preds_log)
    actuals = np.expm1(y_test)
    
    predictions[name] = preds
    
    mae = mean_absolute_error(actuals, preds)
    mse = mean_squared_error(actuals, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(actuals, preds)
    
    results_list.append({
        'Model': name,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    })

results_df = pd.DataFrame(results_list)
print(results_df.to_string(index=False))

# 5. CROSS VALIDATION
print("\n5. CROSS VALIDATION")
cv_results = []
for name, pipeline in pipelines.items():
    scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    rmse_scores = np.sqrt(-scores)
    cv_results.append({
        'Model': name,
        'Mean_CV_RMSE_log': np.mean(rmse_scores),
        'Std_CV_RMSE_log': np.std(rmse_scores)
    })
    print(f"{name}: Mean CV RMSE (log scale) = {np.mean(rmse_scores):.4f}, Std = {np.std(rmse_scores):.4f}")

cv_df = pd.DataFrame(cv_results)

# 6. GENERATE AND SAVE ALL GRAPHS
print("\n6. GENERATING GRAPHS")

def plot_bar(df, x_col, y_col, title, filename, ylabel):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=x_col, y=y_col, data=df, palette='viridis')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f"graphs/{filename}", dpi=300)
    plt.close()

plot_bar(results_df, 'Model', 'MAE', 'Model Comparison - MAE', 'model_mae_comparison.png', 'MAE ($)')
plot_bar(results_df, 'Model', 'RMSE', 'Model Comparison - RMSE', 'model_rmse_comparison.png', 'RMSE ($)')
plot_bar(results_df, 'Model', 'R2', 'Model Comparison - R²', 'model_r2_comparison.png', 'R² Score')

# d) cross_validation_comparison.png
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Mean_CV_RMSE_log', data=cv_df, palette='viridis', capsize=.1)
plt.errorbar(x=range(len(cv_df)), y=cv_df['Mean_CV_RMSE_log'], yerr=cv_df['Std_CV_RMSE_log'], fmt='none', c='black', capsize=5)
plt.title('Cross Validation RMSE (Log Scale) with Std Dev')
plt.ylabel('CV RMSE (log)')
plt.tight_layout()
plt.savefig("graphs/cross_validation_comparison.png", dpi=300)
plt.close()

# Scatter plots actual vs predicted
def plot_scatter(actuals, preds, title, filename):
    plt.figure(figsize=(8, 8))
    plt.scatter(actuals, preds, alpha=0.6, color='b')
    min_val = min(actuals.min(), preds.min())
    max_val = max(actuals.max(), preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title(title)
    plt.xlabel('Actual SalePrice ($)')
    plt.ylabel('Predicted SalePrice ($)')
    
    # Format axes with $ and commas
    ax = plt.gca()
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.get_yaxis().set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    plt.tight_layout()
    plt.savefig(f"graphs/{filename}", dpi=300)
    plt.close()

actuals_exp = np.expm1(y_test)
plot_scatter(actuals_exp, predictions['Linear Regression'], 'Actual vs Predicted (Linear Regression)', 'actual_vs_predicted_linear.png')
plot_scatter(actuals_exp, predictions['Decision Tree'], 'Actual vs Predicted (Decision Tree)', 'actual_vs_predicted_tree.png')
plot_scatter(actuals_exp, predictions['Random Forest'], 'Actual vs Predicted (Random Forest)', 'actual_vs_predicted_random_forest.png')
plot_scatter(actuals_exp, predictions['XGBoost'], 'Actual vs Predicted (XGBoost)', 'actual_vs_predicted_xgboost.png')

# 7. FINAL MODEL SELECTION
best_model_name = results_df.sort_values(by='RMSE').iloc[0]['Model']
print(f"\n7. FINAL MODEL SELECTION: {best_model_name}")

# i, j) Residuals for best model
best_preds = predictions[best_model_name]
residuals = actuals_exp - best_preds

plt.figure(figsize=(10, 6))
plt.scatter(best_preds, residuals, alpha=0.6)
plt.axhline(0, color='r', linestyle='--')
plt.title(f'Residuals vs Predicted ({best_model_name})')
plt.xlabel('Predicted SalePrice ($)')
plt.ylabel('Residuals ($)')
ax = plt.gca()
ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax.get_yaxis().set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.tight_layout()
plt.savefig("graphs/final_model_residuals.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=50, kde=True)
plt.title(f'Prediction Error Distribution ({best_model_name})')
plt.xlabel('Residuals ($)')
ax = plt.gca()
ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.tight_layout()
plt.savefig("graphs/prediction_error_distribution.png", dpi=300)
plt.close()

# Feature Importances
def plot_feature_importance(pipeline, model_name, filename):
    try:
        model = pipeline.named_steps['model']
        preprocessor = pipeline.named_steps['preprocessor']
        
        # Get feature names
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_features = cat_encoder.get_feature_names_out(categorical_features)
        feature_names = np.concatenate([numeric_features, cat_features])
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:20]
            
            plt.figure(figsize=(12, 8))
            sns.barplot(x=importances[indices], y=feature_names[indices], orient='h', palette='viridis')
            plt.title(f'Top 20 Features Importance ({model_name})')
            plt.tight_layout()
            plt.savefig(f"graphs/{filename}", dpi=300)
            plt.close()
    except Exception as e:
        print(f"Could not plot feature importance for {model_name}: {e}")

plot_feature_importance(pipelines['Random Forest'], 'Random Forest', 'feature_importance_random_forest.png')
plot_feature_importance(pipelines['XGBoost'], 'XGBoost', 'feature_importance_xgboost.png')

# m) sample_predictions_comparison.png
sample_actuals = actuals_exp.iloc[:5].values
sample_preds = best_preds[:5]
sample_indices = np.arange(5)
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(sample_indices - width/2, sample_actuals, width, label='Actual')
plt.bar(sample_indices + width/2, sample_preds, width, label='Predicted')
plt.xlabel('Sample Index')
plt.ylabel('SalePrice ($)')
plt.title('Actual vs Predicted for 5 Sample Cases')
plt.xticks(sample_indices, [f'Case {i+1}' for i in range(5)])
plt.legend()
ax = plt.gca()
ax.get_yaxis().set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
plt.tight_layout()
plt.savefig("graphs/sample_predictions_comparison.png", dpi=300)
plt.close()

# n) project_workflow.png - Vertical workflow diagram
fig, ax = plt.subplots(figsize=(8, 14))
ax.axis('off')
steps = ['Dataset', 'Data Cleaning', 'EDA', 'Feature Engineering', 'Preprocessing',
         'Train/Test Split', 'ML Models', 'Evaluation', 'Best Model',
         'Saved Model (.pkl)', 'Prediction']
colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6',
          '#1ABC9C', '#E67E22', '#2980B9', '#27AE60', '#8E44AD', '#16A085']

n = len(steps)
box_w, box_h = 3.0, 0.6
x_center = 4.0
for i, (step, color) in enumerate(zip(steps, colors)):
    y = (n - i) * 1.1
    rect = patches.FancyBboxPatch((x_center - box_w/2, y - box_h/2), box_w, box_h,
                                   boxstyle="round,pad=0.1", facecolor=color, edgecolor='white', linewidth=2, alpha=0.85)
    ax.add_patch(rect)
    ax.text(x_center, y, step, ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    if i < n - 1:
        ax.annotate('', xy=(x_center, y - box_h/2 - 0.15), xytext=(x_center, y - box_h/2 - 0.35),
                    arrowprops=dict(arrowstyle='->', lw=2.5, color='gray'))

ax.set_xlim(1, 7)
ax.set_ylim(0, (n + 1) * 1.1)
plt.title("Machine Learning Project Workflow", fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig("graphs/project_workflow.png", dpi=300, bbox_inches='tight')
plt.close()


# 8. SAVE FINAL MODEL
print("\n8. SAVING FINAL MODEL")
best_pipeline = pipelines[best_model_name]
joblib.dump(best_pipeline, 'house_price_model.pkl')
print(f"Model saved to 'house_price_model.pkl'.")

# Test loading
loaded_model = joblib.load('house_price_model.pkl')
test_preds_log = loaded_model.predict(X_test.iloc[:5])
test_preds = np.expm1(test_preds_log)
print("Loaded model prediction successful.")

# 9. SAMPLE PREDICTIONS
print("\n9. SAMPLE PREDICTIONS")
sample_data = []
for i in range(5):
    actual = actuals_exp.iloc[i]
    pred = best_preds[i]
    abs_err = abs(actual - pred)
    pct_err = (abs_err / actual) * 100
    sample_data.append({
        'Case': f"Case {i+1}",
        'ActualPrice': actual,
        'PredictedPrice': pred,
        'AbsoluteError': abs_err,
        'PercentageError': pct_err
    })

sample_df = pd.DataFrame(sample_data)
sample_df.to_csv('sample_predictions.csv', index=False)
print(sample_df.to_string(index=False))

# 10. FINAL SUMMARY
print("\n10. FINAL SUMMARY")
print("-" * 50)
print(results_df.to_string(index=False))
print("\nCV Results:")
print(cv_df.to_string(index=False))
print(f"\nFinal Model: {best_model_name}")
print("Graphs saved: 14")
print("Files created: house_price_model.pkl, sample_predictions.csv")
print("BUILD COMPLETE")
