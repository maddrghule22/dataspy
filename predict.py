"""
House Price Prediction - Command Line Interface
================================================
Loads the trained sklearn Pipeline model from house_price_model.pkl
and predicts house prices based on user input.

Usage:
    python predict.py              # Interactive mode (prompts for input)
    python predict.py --test       # Run with sample test data
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib


def load_model(model_path="house_price_model.pkl"):
    """Loads the saved sklearn Pipeline from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. "
            "Please run the training pipeline first."
        )
    return joblib.load(model_path)


def create_input_dataframe(user_inputs):
    """
    Creates a single-row DataFrame from a dictionary of user inputs.
    Missing columns are filled with reasonable defaults so the
    sklearn Pipeline can process the record.
    """
    # Default values for all 80 features (excluding Id and SalePrice)
    defaults = {
        "MSSubClass": 60, "MSZoning": "RL", "LotFrontage": 70.0,
        "LotArea": 9000, "Street": "Pave", "Alley": np.nan,
        "LotShape": "Reg", "LandContour": "Lvl", "Utilities": "AllPub",
        "LotConfig": "Inside", "LandSlope": "Gtl", "Neighborhood": "NAmes",
        "Condition1": "Norm", "Condition2": "Norm", "BldgType": "1Fam",
        "HouseStyle": "1Story", "OverallQual": 5, "OverallCond": 5,
        "YearBuilt": 2000, "YearRemodAdd": 2000, "RoofStyle": "Gable",
        "RoofMatl": "CompShg", "Exterior1st": "VinylSd",
        "Exterior2nd": "VinylSd", "MasVnrType": np.nan, "MasVnrArea": 0.0,
        "ExterQual": "TA", "ExterCond": "TA", "Foundation": "PConc",
        "BsmtQual": "TA", "BsmtCond": "TA", "BsmtExposure": "No",
        "BsmtFinType1": "Unf", "BsmtFinSF1": 0.0, "BsmtFinType2": "Unf",
        "BsmtFinSF2": 0.0, "BsmtUnfSF": 0.0, "TotalBsmtSF": 0.0,
        "Heating": "GasA", "HeatingQC": "TA", "CentralAir": "Y",
        "Electrical": "SBrkr", "1stFlrSF": 1000, "2ndFlrSF": 0,
        "LowQualFinSF": 0, "GrLivArea": 1200, "BsmtFullBath": 0.0,
        "BsmtHalfBath": 0.0, "FullBath": 1, "HalfBath": 0,
        "BedroomAbvGr": 3, "KitchenAbvGr": 1, "KitchenQual": "TA",
        "TotRmsAbvGrd": 6, "Functional": "Typ", "Fireplaces": 0,
        "FireplaceQu": np.nan, "GarageType": "Attchd",
        "GarageYrBlt": 2000.0, "GarageFinish": "Unf", "GarageCars": 1.0,
        "GarageArea": 400.0, "GarageQual": "TA", "GarageCond": "TA",
        "PavedDrive": "Y", "WoodDeckSF": 0, "OpenPorchSF": 0,
        "EnclosedPorch": 0, "3SsnPorch": 0, "ScreenPorch": 0,
        "PoolArea": 0, "PoolQC": np.nan, "Fence": np.nan,
        "MiscFeature": np.nan, "MiscVal": 0, "MoSold": 6,
        "YrSold": 2008, "SaleType": "WD", "SaleCondition": "Normal"
    }

    # Override defaults with user-supplied values
    for key, value in user_inputs.items():
        defaults[key] = value

    df = pd.DataFrame([defaults])

    # Feature Engineering (must match training pipeline)
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBath'] = (df['FullBath'] + 0.5 * df['HalfBath'] +
                       df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath'])
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodeledAge'] = df['YrSold'] - df['YearRemodAdd']
    df['TotalPorchArea'] = (df['OpenPorchSF'] + df['EnclosedPorch'] +
                            df['3SsnPorch'] + df['ScreenPorch'])

    return df


def predict_price(model, input_df):
    """Predicts the house price using the loaded Pipeline model."""
    log_prediction = model.predict(input_df)[0]
    return np.expm1(log_prediction)


def interactive_mode(model):
    """Prompts the user for key property details and predicts price."""
    print("=" * 55)
    print("   HOUSE PRICE PREDICTION SYSTEM")
    print("   Powered by Machine Learning")
    print("=" * 55)
    print("\nEnter property details below.")
    print("Press Enter to use default values.\n")

    def get_input(prompt, default, dtype=float):
        val = input(f"  {prompt} [{default}]: ").strip()
        if val == "":
            return default
        try:
            return dtype(val)
        except ValueError:
            print(f"  Invalid input, using default: {default}")
            return default

    def get_str_input(prompt, default):
        val = input(f"  {prompt} [{default}]: ").strip()
        return val if val else default

    overall_qual = get_input("Overall Quality (1-10)", 7, int)
    gr_liv_area = get_input("Living Area sq ft", 1500, int)
    year_built = get_input("Year Built", 2005, int)
    garage_cars = get_input("Garage Capacity (cars)", 2, int)
    total_bsmt_sf = get_input("Total Basement Area sq ft", 800, float)
    full_bath = get_input("Full Bathrooms", 2, int)
    lot_area = get_input("Lot Area sq ft", 9000, int)
    neighborhood = get_str_input("Neighborhood", "CollgCr")

    user_inputs = {
        "OverallQual": overall_qual,
        "GrLivArea": gr_liv_area,
        "YearBuilt": year_built,
        "GarageCars": garage_cars,
        "TotalBsmtSF": total_bsmt_sf,
        "FullBath": full_bath,
        "LotArea": lot_area,
        "Neighborhood": neighborhood,
        "1stFlrSF": gr_liv_area if gr_liv_area <= 1500 else int(gr_liv_area * 0.55),
        "2ndFlrSF": 0 if gr_liv_area <= 1500 else int(gr_liv_area * 0.45),
        "GarageArea": garage_cars * 275,
        "GarageYrBlt": float(year_built),
        "YearRemodAdd": year_built,
    }

    input_df = create_input_dataframe(user_inputs)
    price = predict_price(model, input_df)

    print("\n" + "=" * 55)
    print("   PREDICTION RESULT")
    print("=" * 55)
    print(f"\n   Neighborhood:       {neighborhood}")
    print(f"   Overall Quality:    {overall_qual}/10")
    print(f"   Living Area:        {gr_liv_area:,} sq ft")
    print(f"   Year Built:         {year_built}")
    print(f"   Garage:             {garage_cars} cars")
    print(f"   Basement:           {total_bsmt_sf:,.0f} sq ft")
    print(f"   Bathrooms:          {full_bath}")
    print(f"   Lot Area:           {lot_area:,} sq ft")
    print(f"\n   PREDICTED PRICE:    ${price:,.2f}")
    print("=" * 55)


def test_mode(model):
    """Runs prediction on a sample premium house for verification."""
    print("Running test prediction on a sample house...\n")

    test_house = {
        "MSSubClass": 60, "MSZoning": "RL", "LotFrontage": 80.0,
        "LotArea": 10000, "Neighborhood": "CollgCr", "BldgType": "1Fam",
        "HouseStyle": "2Story", "OverallQual": 8, "OverallCond": 5,
        "YearBuilt": 2007, "YearRemodAdd": 2008,
        "Exterior1st": "VinylSd", "Exterior2nd": "VinylSd",
        "MasVnrType": "BrkFace", "MasVnrArea": 160.0,
        "ExterQual": "Gd", "Foundation": "PConc",
        "BsmtQual": "Gd", "BsmtCond": "TA", "BsmtExposure": "Av",
        "BsmtFinType1": "GLQ", "BsmtFinSF1": 850.0,
        "TotalBsmtSF": 1000.0, "HeatingQC": "Ex", "CentralAir": "Y",
        "1stFlrSF": 1000, "2ndFlrSF": 900, "GrLivArea": 1900,
        "BsmtFullBath": 1.0, "FullBath": 2, "HalfBath": 1,
        "BedroomAbvGr": 3, "KitchenQual": "Gd", "TotRmsAbvGrd": 8,
        "Fireplaces": 1, "FireplaceQu": "Gd",
        "GarageType": "Attchd", "GarageYrBlt": 2007.0,
        "GarageFinish": "RFn", "GarageCars": 2.0, "GarageArea": 550.0,
        "GarageQual": "TA", "GarageCond": "TA", "PavedDrive": "Y",
        "WoodDeckSF": 200, "OpenPorchSF": 60,
        "MoSold": 5, "YrSold": 2009, "SaleCondition": "Normal"
    }

    input_df = create_input_dataframe(test_house)
    price = predict_price(model, input_df)

    print("=" * 55)
    print("   TEST PREDICTION RESULT")
    print("=" * 55)
    print(f"\n   Neighborhood:       CollgCr")
    print(f"   Overall Quality:    8/10")
    print(f"   Living Area:        1,900 sq ft")
    print(f"   Year Built:         2007")
    print(f"   Garage:             2 cars")
    print(f"\n   PREDICTED PRICE:    ${price:,.2f}")
    print("=" * 55)


if __name__ == "__main__":
    print("Loading model pipeline...")
    model = load_model()
    print("Model loaded successfully.\n")

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_mode(model)
    else:
        interactive_mode(model)
