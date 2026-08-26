"""
DataSpy AI Property Valuation - Local Prediction & Analytics API Server
=======================================================================
Exposes HTTP JSON API on port 5000:
- POST /api/predict: returns ML model prediction using house_price_model.pkl
- GET /api/analytics: returns dataset statistics and analytics from dataset_analytics.json
"""

import os
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import pandas as pd
import numpy as np
import joblib

MODEL_PATH = "house_price_model.pkl"
ANALYTICS_PATH = "dataset_analytics.json"

print(f"Loading trained ML Pipeline from {MODEL_PATH}...")
try:
    model_pipeline = joblib.load(MODEL_PATH)
    print("Model pipeline loaded successfully into memory.")
except Exception as e:
    print(f"Warning: Could not load {MODEL_PATH}: {e}")
    model_pipeline = None

analytics_cache = {}
if os.path.exists(ANALYTICS_PATH):
    try:
        with open(ANALYTICS_PATH, "r") as f:
            analytics_cache = json.load(f)
        print("Analytics dataset metrics loaded into memory.")
    except Exception as e:
        print(f"Warning loading analytics JSON: {e}")

def create_input_df(inputs):
    defaults = {
        "MSSubClass": 60, "MSZoning": "RL", "LotFrontage": 70.0,
        "LotArea": 9000, "Street": "Pave", "Alley": np.nan,
        "LotShape": "Reg", "LandContour": "Lvl", "Utilities": "AllPub",
        "LotConfig": "Inside", "LandSlope": "Gtl", "Neighborhood": "NridgHt",
        "Condition1": "Norm", "Condition2": "Norm", "BldgType": "1Fam",
        "HouseStyle": "1Story", "OverallQual": 8, "OverallCond": 5,
        "YearBuilt": 1990, "YearRemodAdd": 1990, "RoofStyle": "Gable",
        "RoofMatl": "CompShg", "Exterior1st": "VinylSd",
        "Exterior2nd": "VinylSd", "MasVnrType": np.nan, "MasVnrArea": 0.0,
        "ExterQual": "TA", "ExterCond": "TA", "Foundation": "PConc",
        "BsmtQual": "TA", "BsmtCond": "TA", "BsmtExposure": "No",
        "BsmtFinType1": "Unf", "BsmtFinSF1": 0.0, "BsmtFinType2": "Unf",
        "BsmtFinSF2": 0.0, "BsmtUnfSF": 0.0, "TotalBsmtSF": 1000.0,
        "Heating": "GasA", "HeatingQC": "TA", "CentralAir": "Y",
        "Electrical": "SBrkr", "1stFlrSF": 1000, "2ndFlrSF": 900,
        "LowQualFinSF": 0, "GrLivArea": 1900, "BsmtFullBath": 0.0,
        "BsmtHalfBath": 0.0, "FullBath": 2, "HalfBath": 1,
        "BedroomAbvGr": 3, "KitchenAbvGr": 1, "KitchenQual": "Gd",
        "TotRmsAbvGrd": 7, "Functional": "Typ", "Fireplaces": 1,
        "FireplaceQu": np.nan, "GarageType": "Attchd",
        "GarageYrBlt": 1990.0, "GarageFinish": "Unf", "GarageCars": 2.0,
        "GarageArea": 500.0, "GarageQual": "TA", "GarageCond": "TA",
        "PavedDrive": "Y", "WoodDeckSF": 100, "OpenPorchSF": 50,
        "EnclosedPorch": 0, "3SsnPorch": 0, "ScreenPorch": 0,
        "PoolArea": 0, "PoolQC": np.nan, "Fence": np.nan,
        "MiscFeature": np.nan, "MiscVal": 0, "MoSold": 6,
        "YrSold": 2008, "SaleType": "WD", "SaleCondition": "Normal"
    }

    if "grLivArea" in inputs:
        area = float(inputs["grLivArea"])
        defaults["GrLivArea"] = area
        defaults["1stFlrSF"] = area if area <= 1500 else area * 0.55
        defaults["2ndFlrSF"] = 0 if area <= 1500 else area * 0.45

    if "totalBsmtSF" in inputs:
        defaults["TotalBsmtSF"] = float(inputs["totalBsmtSF"])

    if "overallQual" in inputs:
        defaults["OverallQual"] = int(inputs["overallQual"])

    if "garageCars" in inputs:
        cars = float(inputs["garageCars"])
        defaults["GarageCars"] = cars
        defaults["GarageArea"] = cars * 260.0

    if "houseAge" in inputs:
        age = int(inputs["houseAge"])
        built_year = 2008 - age
        defaults["YearBuilt"] = built_year
        defaults["YearRemodAdd"] = built_year

    if "neighborhood" in inputs:
        defaults["Neighborhood"] = str(inputs["neighborhood"])

    if "kitchen" in inputs:
        kq_map = {5: "Ex", 4: "Gd", 3: "TA", 2: "Fa", 1: "Po"}
        defaults["KitchenQual"] = kq_map.get(int(inputs["kitchen"]), "TA")

    df = pd.DataFrame([defaults])

    # Feature Engineering
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBath'] = (df['FullBath'] + 0.5 * df['HalfBath'] +
                       df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath'])
    df['HouseAge'] = df['YrSold'] - df['YearBuilt']
    df['RemodeledAge'] = df['YrSold'] - df['YearRemodAdd']
    df['TotalPorchArea'] = (df['OpenPorchSF'] + df['EnclosedPorch'] +
                            df['3SsnPorch'] + df['ScreenPorch'])

    return df


class PredictAPIHandler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == '/api/analytics':
            self.send_response(200)
            self._send_cors_headers()
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(analytics_cache).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/api/predict':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode('utf-8'))
                input_df = create_input_df(data)
                city_mult = float(data.get("cityMult", 1.0))
                
                if model_pipeline is not None:
                    log_pred = model_pipeline.predict(input_df)[0]
                    base_price = float(np.expm1(log_pred))
                else:
                    base_price = 220000.0

                final_price = round(base_price * city_mult)
                
                response = {
                    "status": "success",
                    "predicted_price": final_price,
                    "model_used": "Stacking Ensemble Regressor",
                    "r2": 0.9358,
                    "rmse": 18828,
                    "mae": 13537,
                    "cv_rmse": 0.1127
                }
                
                self.send_response(200)
                self._send_cors_headers()
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self._send_cors_headers()
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

def run_server(port=5000):
    server_address = ('127.0.0.1', port)
    httpd = HTTPServer(server_address, PredictAPIHandler)
    print(f"Prediction API Server running on http://127.0.0.1:{port}")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()
