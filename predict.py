"""
House Price Prediction - Command Line Interface (Enhanced)
================================================================
Loads the trained ML Stacking / Regularized Pipeline model from house_price_model.pkl
and predicts house prices based on user input, featuring multi-city geographic scaling.

Usage:
    py predict.py              # Interactive mode (prompts for input & location)
    py predict.py --test       # Run with sample test data across cities
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

# Geographic Expansion City Market Multipliers (Base = Ames, IA)
CITY_MARKET_INDICES = {
    "Ames, IA (Base Market)": 1.00,
    "Des Moines, IA": 1.08,
    "Omaha, NE": 1.15,
    "Minneapolis, MN": 1.35,
    "Chicago, IL": 1.45,
    "Denver, CO": 1.65,
    "Austin, TX": 1.55,
    "Seattle, WA": 1.95
}


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


def predict_price(model, input_df, city_multiplier=1.00):
    """Predicts the house price using the loaded Pipeline model and applies city market scaling."""
    log_prediction = model.predict(input_df)[0]
    base_price = np.expm1(log_prediction)
    return base_price * city_multiplier


def interactive_mode(model):
    """Prompts the user for key property details and location market to predict price."""
    print("=" * 60)
    print("   AI PROPERTY VALUATION SYSTEM (ENHANCED)")
    print("   Powered by Stacking Ensemble & Hyperparameter Tuning")
    print("=" * 60)
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

    print("\nSelect Target City / Location Market:")
    cities = list(CITY_MARKET_INDICES.keys())
    for idx, city in enumerate(cities, 1):
        print(f"  [{idx}] {city} (Index: {CITY_MARKET_INDICES[city]}x)")
    
    city_choice = get_input("City Choice (1-8)", 1, int)
    selected_city = cities[city_choice - 1] if 1 <= city_choice <= len(cities) else cities[0]
    city_multiplier = CITY_MARKET_INDICES[selected_city]

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
    price = predict_price(model, input_df, city_multiplier)

    print("\n" + "=" * 60)
    print("   PREDICTION RESULT")
    print("=" * 60)
    print(f"\n   Target Market:      {selected_city}")
    print(f"   Market Factor:      {city_multiplier:.2f}x")
    print(f"   Neighborhood:       {neighborhood}")
    print(f"   Overall Quality:    {overall_qual}/10")
    print(f"   Living Area:        {gr_liv_area:,} sq ft")
    print(f"   Year Built:         {year_built}")
    print(f"   Garage:             {garage_cars} cars")
    print(f"   Basement:           {total_bsmt_sf:,.0f} sq ft")
    print(f"   Bathrooms:          {full_bath}")
    print(f"   Lot Area:           {lot_area:,} sq ft")
    print(f"\n   ESTIMATED VALUATION: ${price:,.2f}")
    print("=" * 60)


def test_mode(model):
    """Runs prediction on sample houses across multiple city markets for verification."""
    print("Running test prediction across multiple location markets...\n")

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
    base_price = predict_price(model, input_df, 1.00)

    print("=" * 60)
    print("   TEST PREDICTION RESULTS - GEOGRAPHIC EXPANSION")
    print("=" * 60)
    print(f"\n   Property Spec:      1,900 sq ft | Quality 8/10 | Built 2007 | 2 Cars")
    print(f"   Base Price (Ames):  ${base_price:,.2f}\n")
    print("   City Market Scaling Estimates:")
    for city, mult in CITY_MARKET_INDICES.items():
        scaled_price = base_price * mult
        print(f"   - {city:<25} ({mult:.2f}x): ${scaled_price:,.2f}")
    print("=" * 60)


if __name__ == "__main__":
    print("Loading model pipeline...")
    model = load_model()
    print("Model loaded successfully.\n")

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_mode(model)
    else:
        interactive_mode(model)
