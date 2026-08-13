import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def run_ml_pipeline():
    print("=" * 60)
    print("STARTING HOUSE PRICE PREDICTION END-TO-END PIPELINE")
    print("=" * 60)
    
    # 1. Load Data
    print("\n[Step 1] Loading dataset...")
    if not os.path.exists("dataset/train.csv"):
        raise FileNotFoundError("dataset/train.csv not found. Please ensure dataset is collected.")
    df = pd.read_csv("dataset/train.csv")
    print(f"Loaded training data: {df.shape}")
    
    # 2. Outlier Cleaning
    print("\n[Step 2] Cleaning outliers...")
    df_cleaned = df[~((df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000))].copy()
    print(f"Original records: {len(df)} | Cleaned records: {len(df_cleaned)} (Removed 2 outliers)")
    
    # 3. Separate features and target
    X = df_cleaned.drop(columns=['SalePrice'])
    y = np.log1p(df_cleaned['SalePrice'])
    
    # 4. Train-Test Split (80/20 ratio for local validation)
    print("\n[Step 3] Splitting dataset into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train size: {X_train.shape[0]} | Validation size: {X_val.shape[0]}")
    
    # 5. Imputation Setup
    print("\n[Step 4] Fitting imputations and preprocessing...")
    neighborhood_medians = X_train.groupby("Neighborhood")["LotFrontage"].median()
    global_median = X_train["LotFrontage"].median()
    
    def preprocess_pipeline(df_in):
        df_out = df_in.copy()
        
        # Neighborhood LotFrontage median
        df_out['LotFrontage'] = df_out.apply(
            lambda r: r['LotFrontage'] if not pd.isnull(r['LotFrontage']) else 
            neighborhood_medians.get(r['Neighborhood'], global_median),
            axis=1
        )
        
        # Categorical absence -> None
        none_categories = [
            'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
            'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
            'MasVnrType'
        ]
        for col in none_categories:
            if col in df_out.columns:
                df_out[col] = df_out[col].fillna('None')
                
        # Numerical absence -> 0
        zero_numericals = [
            'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
            'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea',
            'GarageCars', 'GarageArea', 'GarageYrBlt'
        ]
        for col in zero_numericals:
            if col in df_out.columns:
                df_out[col] = df_out[col].fillna(0)
                
        # Mode/Median fallback
        for col in df_out.columns:
            if col in ['Id', 'LotFrontage']:
                continue
            if df_out[col].isnull().sum() > 0:
                if df_out[col].dtype == 'object':
                    mode_val = X_train[col].mode()[0] if not X_train[col].mode().empty else 'None'
                    df_out[col] = df_out[col].fillna(mode_val)
                else:
                    median_val = X_train[col].median()
                    df_out[col] = df_out[col].fillna(median_val)
        return df_out
        
    X_train_pre = preprocess_pipeline(X_train)
    X_val_pre = preprocess_pipeline(X_val)
    
    # 6. Ordinal Mapping
    ordinal_mappings = {
        'ExterQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},
        'ExterCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},
        'BsmtQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},
        'BsmtCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},
        'BsmtExposure': {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0},
        'BsmtFinType1': {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': 0},
        'BsmtFinType2': {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': 0},
        'HeatingQC': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},
        'KitchenQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},
        'FireplaceQu': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},
        'GarageFinish': {'Fin': 3, 'RFn': 2, 'Unf': 1, 'None': 0},
        'GarageQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},
        'GarageCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0},
        'PoolQC': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'None': 0}
    }
    
    def map_ordinals(df_in):
        df_out = df_in.copy()
        for col, mapping in ordinal_mappings.items():
            if col in df_out.columns:
                df_out[col] = df_out[col].map(mapping).fillna(0).astype(int)
        return df_out
        
    X_train_ord = map_ordinals(X_train_pre)
    X_val_ord = map_ordinals(X_val_pre)
    
    # 7. Feature Engineering
    print("\n[Step 5] Adding engineered features...")
    def add_engineered_features(df_in):
        df_out = df_in.copy()
        df_out['TotalSF'] = df_out['1stFlrSF'] + df_out['2ndFlrSF'] + df_out['TotalBsmtSF']
        df_out['TotalBath'] = df_out['FullBath'] + 0.5*df_out['HalfBath'] + df_out['BsmtFullBath'] + 0.5*df_out['BsmtHalfBath']
        df_out['HouseAge'] = df_out['YrSold'] - df_out['YearBuilt']
        df_out['HouseAge'] = df_out['HouseAge'].apply(lambda x: max(0, x))
        return df_out
        
    X_train_eng = add_engineered_features(X_train_ord)
    X_val_eng = add_engineered_features(X_val_ord)
    
    # 8. Nominal Encoding & Alignment
    categorical_cols = X_train_eng.select_dtypes(include=['object']).columns.tolist()
    nominal_cols = [c for c in categorical_cols if c not in ordinal_mappings]
    train_dummies = pd.get_dummies(X_train_eng[nominal_cols], drop_first=True)
    dummy_schema = train_dummies.columns.tolist()
    
    def final_process_and_align(df_in):
        dummies_in = pd.get_dummies(df_in[nominal_cols], drop_first=True)
        dummies_aligned = dummies_in.reindex(columns=dummy_schema, fill_value=0)
        keep_cols = X_train_eng.select_dtypes(exclude=['object']).columns.tolist()
        if 'Id' in keep_cols:
            keep_cols.remove('Id')
        df_out = pd.concat([df_in[keep_cols].reset_index(drop=True), dummies_aligned.reset_index(drop=True)], axis=1)
        return df_out
        
    X_train_final = final_process_and_align(X_train_eng)
    X_val_final = final_process_and_align(X_val_eng)
    
    # 9. Scaling Continuous Features
    scale_cols = []
    for col in X_train_final.columns:
        u_vals = X_train_final[col].unique()
        is_bin = len(u_vals) <= 2 and all(v in [0, 1] for v in u_vals)
        if not is_bin:
            scale_cols.append(col)
            
    scaler = StandardScaler()
    X_train_final[scale_cols] = scaler.fit_transform(X_train_final[scale_cols])
    X_val_final[scale_cols] = scaler.transform(X_val_final[scale_cols])
    
    # 10. Model Training & Evaluation
    print("\n[Step 6] Training regressors...")
    lr = LinearRegression().fit(X_train_final, y_train)
    dt = DecisionTreeRegressor(max_depth=6, random_state=42).fit(X_train_final, y_train)
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1).fit(X_train_final, y_train)
    
    # Val Predictions
    y_pred_lr = np.expm1(lr.predict(X_val_final))
    y_pred_dt = np.expm1(dt.predict(X_val_final))
    y_pred_rf = np.expm1(rf.predict(X_val_final))
    y_val_orig = np.expm1(y_val)
    
    # Calculate scores
    def get_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        return mae, rmse, r2
        
    mae_lr, rmse_lr, r2_lr = get_metrics(y_val_orig, y_pred_lr)
    mae_dt, rmse_dt, r2_dt = get_metrics(y_val_orig, y_pred_dt)
    mae_rf, rmse_rf, r2_rf = get_metrics(y_val_orig, y_pred_rf)
    
    print("\nModel Performance Table (Original Scale - USD):")
    print("-" * 75)
    print(f"{'Model':22} | {'MAE (USD)':12} | {'RMSE (USD)':12} | {'R2 Score':8}")
    print("-" * 75)
    print(f"{'Linear Regression':22} | ${mae_lr:10.2f} | ${rmse_lr:10.2f} | {r2_lr:7.4f}")
    print(f"{'Decision Tree':22} | ${mae_dt:10.2f} | ${rmse_dt:10.2f} | {r2_dt:7.4f}")
    print(f"{'Random Forest':22} | ${mae_rf:10.2f} | ${rmse_rf:10.2f} | {r2_rf:7.4f}")
    print("-" * 75)
    
    # 11. Plot & Export graphs
    print("\n[Step 7] Exporting diagnostic graphs to graphs/ folder...")
    os.makedirs("graphs", exist_ok=True)
    
    # Model Comparison Chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(["Linear Reg", "Decision Tree", "Random Forest"], [rmse_lr, rmse_dt, rmse_rf], 
                   color=['royalblue', 'orange', 'seagreen'], edgecolor='gray', width=0.55)
    plt.ylabel("Validation RMSE ($)")
    plt.title("Model Validation Performance Comparison (RMSE)", fontweight='bold', pad=15)
    plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: "${:,}".format(int(x))))
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + 500, f"${h:,.0f}", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig("graphs/model_performance_comparison.png", dpi=300)
    plt.close()
    
    # Predicted vs Actual (Random Forest)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_val_orig, y=y_pred_rf, alpha=0.7, color='seagreen', edgecolor='w', s=55)
    lims = [min(y_val_orig.min(), y_pred_rf.min()), max(y_val_orig.max(), y_pred_rf.max())]
    plt.plot(lims, lims, color='r', linestyle='--', linewidth=2, label="Ideal Fit")
    plt.xlabel("Actual SalePrice ($)")
    plt.ylabel("Predicted SalePrice ($)")
    plt.title("Random Forest: Predicted vs. Actual House Prices", fontweight='bold', pad=15)
    plt.legend()
    plt.gca().get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: "{:,}".format(int(x))))
    plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: "{:,}".format(int(x))))
    plt.tight_layout()
    plt.savefig("graphs/predictions_vs_actual.png", dpi=300)
    plt.close()
    
    # Residual Plot
    residuals = y_val_orig - y_pred_rf
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred_rf, y=residuals, alpha=0.7, color='orange', edgecolor='w', s=55)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel("Predicted SalePrice ($)")
    plt.ylabel("Residual Error ($)")
    plt.title("Random Forest Residual Plot", fontweight='bold', pad=15)
    plt.gca().get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: "{:,}".format(int(x))))
    plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: "{:,}".format(int(x))))
    plt.tight_layout()
    plt.savefig("graphs/residuals_plot.png", dpi=300)
    plt.close()
    
    # Feature Importance Chart
    plt.figure(figsize=(12, 7))
    importances = rf.feature_importances_
    f_names = X_train_final.columns.tolist()
    sorted_idx = np.argsort(importances)[::-1][:15]
    sns.barplot(x=importances[sorted_idx], y=[f_names[i] for i in sorted_idx], palette='viridis')
    plt.xlabel("Relative Importance Score")
    plt.title("Random Forest: Top 15 Feature Importances", fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig("graphs/feature_importances.png", dpi=300)
    plt.close()
    
    # 12. Full dataset retrain and deployment save
    print("\n[Step 8] Retraining Random Forest on 100% data for production...")
    full_neighborhood_frontage = X.groupby("Neighborhood")["LotFrontage"].median()
    full_global_frontage = X["LotFrontage"].median()
    
    def full_preprocess_pipeline(df_in):
        df_out = df_in.copy()
        df_out['LotFrontage'] = df_out.apply(
            lambda r: r['LotFrontage'] if not pd.isnull(r['LotFrontage']) else 
            full_neighborhood_frontage.get(r['Neighborhood'], full_global_frontage),
            axis=1
        )
        none_cols = [
            'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
            'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
            'MasVnrType'
        ]
        for col in none_cols:
            if col in df_out.columns:
                df_out[col] = df_out[col].fillna('None')
                
        zero_cols = [
            'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
            'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea',
            'GarageCars', 'GarageArea', 'GarageYrBlt'
        ]
        for col in zero_cols:
            if col in df_out.columns:
                df_out[col] = df_out[col].fillna(0)
                
        for col in df_out.columns:
            if col in ['Id', 'LotFrontage']:
                continue
            if df_out[col].isnull().sum() > 0:
                if df_out[col].dtype == 'object':
                    mode_val = X[col].mode()[0] if not X[col].mode().empty else 'None'
                    df_out[col] = df_out[col].fillna(mode_val)
                else:
                    median_val = X[col].median()
                    df_out[col] = df_out[col].fillna(median_val)
        return df_out
        
    X_full_pre = full_preprocess_pipeline(X)
    X_full_ord = map_ordinals(X_full_pre)
    X_full_eng = add_engineered_features(X_full_ord)
    
    full_dummies = pd.get_dummies(X_full_eng[nominal_cols], drop_first=True)
    full_dummy_schema = full_dummies.columns.tolist()
    
    def full_process_and_align(df_in):
        dummies_in = pd.get_dummies(df_in[nominal_cols], drop_first=True)
        dummies_aligned = dummies_in.reindex(columns=full_dummy_schema, fill_value=0)
        keep_cols = X_full_eng.select_dtypes(exclude=['object']).columns.tolist()
        if 'Id' in keep_cols:
            keep_cols.remove('Id')
        df_out = pd.concat([df_in[keep_cols].reset_index(drop=True), dummies_aligned.reset_index(drop=True)], axis=1)
        return df_out
        
    X_full_final = full_process_and_align(X_full_eng)
    
    full_scaler = StandardScaler()
    X_full_final[scale_cols] = full_scaler.fit_transform(X_full_final[scale_cols])
    
    # Fit
    final_rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    final_rf.fit(X_full_final, y)
    
    # Save bundle
    pipeline_dict = {
        "model": final_rf,
        "scaler": full_scaler,
        "scale_cols": scale_cols,
        "dummy_schema": full_dummy_schema,
        "neighborhood_lot_frontage": full_neighborhood_frontage.to_dict(),
        "global_lot_frontage": full_global_frontage,
        "nominal_cols": nominal_cols,
        "ordinal_mappings": ordinal_mappings
    }
    joblib.dump(pipeline_dict, "house_price_model.pkl")
    print("\n[Step 9] Model pipeline successfully saved to house_price_model.pkl!")
    print("=" * 60)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_ml_pipeline()
