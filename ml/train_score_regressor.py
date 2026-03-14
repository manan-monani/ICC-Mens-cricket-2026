"""
ICC T20 Predictor - Score Regression Model
Algorithm: Random Forest + Gradient Boosting Regressor
Target: final_innings_score (predicted at over 6, 10, 15)
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
import sys
import json
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SCORE_REGRESSOR")

DB_PATH = PROCESSED_DATA_DIR / "icc_cricket.db"
MODELS_DIR = PROCESSED_DATA_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


class ScoreRegressor:
    """Predicts final innings score at different over breakpoints."""

    FEATURE_COLS = [
        "over_number", "current_score", "current_wickets",
        "current_run_rate", "match_phase", "bat_team_win_rate",
        "venue_avg_score",
    ]
    TARGET_COL = "final_score"

    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH
        self.rf_model = None
        self.gb_model = None
        self.scaler = StandardScaler()
        self.metrics = {}
        self.is_trained = False

    def load_data(self) -> pd.DataFrame:
        """Load feature data at breakpoints."""
        conn = sqlite3.connect(str(self.db_path))
        df = pd.read_sql("""
            SELECT * FROM ml_match_features
            WHERE over_number IN (5, 9, 14)
              AND innings_number = 1
              AND final_score > 0
        """, conn)
        conn.close()
        logger.info(f"Loaded {len(df)} score prediction samples")
        return df

    def prepare_data(self, df: pd.DataFrame):
        """Prepare train/test data."""
        df = df.dropna(subset=self.FEATURE_COLS + [self.TARGET_COL])
        df = df.replace([np.inf, -np.inf], 0)

        X = df[self.FEATURE_COLS].copy()
        y = df[self.TARGET_COL].copy()

        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.FEATURE_COLS)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        """Train Random Forest and Gradient Boosting models."""
        logger.info("Training Random Forest Regressor...")
        self.rf_model = RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_split=5,
            min_samples_leaf=3, random_state=42, n_jobs=-1
        )
        self.rf_model.fit(X_train, y_train)

        logger.info("Training Gradient Boosting Regressor...")
        self.gb_model = GradientBoostingRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        self.gb_model.fit(X_train, y_train)

        self.is_trained = True
        logger.info("Both models trained")

    def evaluate(self, X_test, y_test) -> dict:
        """Evaluate models."""
        results = {}
        for name, model in [("RandomForest", self.rf_model), ("GradientBoosting", self.gb_model)]:
            pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            mae = mean_absolute_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            results[name] = {"RMSE": round(rmse, 2), "MAE": round(mae, 2), "R2": round(r2, 4)}
            logger.info(f"{name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}")

        # Ensemble
        rf_pred = self.rf_model.predict(X_test)
        gb_pred = self.gb_model.predict(X_test)
        ens_pred = (rf_pred + gb_pred) / 2
        rmse = np.sqrt(mean_squared_error(y_test, ens_pred))
        mae = mean_absolute_error(y_test, ens_pred)
        r2 = r2_score(y_test, ens_pred)
        results["Ensemble"] = {"RMSE": round(rmse, 2), "MAE": round(mae, 2), "R2": round(r2, 4)}
        logger.info(f"Ensemble: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}")

        self.metrics = results
        return results

    def predict(self, features: dict) -> dict:
        """Predict final score."""
        if not self.is_trained:
            self.load_model()

        feature_row = pd.DataFrame([{col: features.get(col, 0) for col in self.FEATURE_COLS}])
        feature_scaled = self.scaler.transform(feature_row)

        rf_pred = self.rf_model.predict(feature_scaled)[0]
        gb_pred = self.gb_model.predict(feature_scaled)[0]
        ensemble = (rf_pred + gb_pred) / 2

        return {
            "predicted_score": round(float(ensemble)),
            "rf_prediction": round(float(rf_pred)),
            "gb_prediction": round(float(gb_pred)),
            "prediction_range": f"{round(float(min(rf_pred, gb_pred)))}-{round(float(max(rf_pred, gb_pred)))}",
        }

    def save_model(self):
        """Save models."""
        joblib.dump(self.rf_model, MODELS_DIR / "rf_score_regressor.joblib")
        joblib.dump(self.gb_model, MODELS_DIR / "gb_score_regressor.joblib")
        joblib.dump(self.scaler, MODELS_DIR / "score_regressor_scaler.joblib")
        with open(MODELS_DIR / "score_regressor_metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)
        logger.info(f"Models saved to {MODELS_DIR}")

    def load_model(self):
        """Load models."""
        self.rf_model = joblib.load(MODELS_DIR / "rf_score_regressor.joblib")
        self.gb_model = joblib.load(MODELS_DIR / "gb_score_regressor.joblib")
        self.scaler = joblib.load(MODELS_DIR / "score_regressor_scaler.joblib")
        self.is_trained = True

    def log_to_mlflow(self):
        """Log to MLflow."""
        try:
            import mlflow
            import mlflow.sklearn
            import os

            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
            mlflow.set_experiment("ICC_T20_Score_Prediction")

            with mlflow.start_run(run_name=f"score_regressor_{datetime.now().strftime('%Y%m%d_%H%M')}"):
                mlflow.log_params({"model_type": "RF + GB Ensemble", "rf_n_estimators": 200, "gb_n_estimators": 200})
                if "Ensemble" in self.metrics:
                    for k, v in self.metrics["Ensemble"].items():
                        mlflow.log_metric(f"ensemble_{k}", v)
                mlflow.sklearn.log_model(self.rf_model, "rf_model")
                mlflow.sklearn.log_model(self.gb_model, "gb_model")
                logger.info("Logged to MLflow")
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")

    def run_full_training(self):
        """Full training pipeline."""
        logger.info("=" * 60)
        logger.info("STARTING SCORE REGRESSOR TRAINING")
        logger.info("=" * 60)

        df = self.load_data()
        if len(df) < 10:
            logger.warning("Not enough data for score regression. Skipping.")
            return {}

        X_train, X_test, y_train, y_test = self.prepare_data(df)
        self.train(X_train, y_train)
        results = self.evaluate(X_test, y_test)
        self.save_model()
        self.log_to_mlflow()

        logger.info("SCORE REGRESSOR TRAINING COMPLETE")
        return results


if __name__ == "__main__":
    regressor = ScoreRegressor()
    regressor.run_full_training()
