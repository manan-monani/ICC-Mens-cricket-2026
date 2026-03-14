"""
ICC T20 Predictor - Win Probability Prediction Model
Algorithm: XGBoost + LightGBM Ensemble (Classification)
Target: match_winner (1 = batting team wins, 0 = bowling team wins)

Integrates with MLflow for experiment tracking and model registry.
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
import sys
import os
import json
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
    confusion_matrix, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DATA_DIR, PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("WIN_PREDICTOR")

DB_PATH = PROCESSED_DATA_DIR / "icc_cricket.db"
MODELS_DIR = PROCESSED_DATA_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


class WinPredictor:
    """XGBoost + LightGBM ensemble for T20 match win prediction."""

    FEATURE_COLS = [
        "innings_number_enc", "over_number", "match_phase",
        "current_score", "current_wickets", "current_run_rate",
        "required_run_rate", "balls_remaining", "bat_team_win_rate",
        "venue_avg_score", "toss_winner_batting", "target",
        "pressure_index", "momentum_score",
    ]
    TARGET_COL = "match_winner"

    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH
        self.xgb_model = None
        self.lgb_model = None
        self.scaler = StandardScaler()
        self.metrics = {}
        self.is_trained = False

    def load_data(self) -> pd.DataFrame:
        """Load ML features from the database."""
        logger.info("Loading feature data...")
        conn = sqlite3.connect(str(self.db_path))
        df = pd.read_sql("SELECT * FROM ml_match_features", conn)
        conn.close()
        logger.info(f"Loaded {len(df)} feature rows")
        return df

    def prepare_data(self, df: pd.DataFrame):
        """Prepare train/test split."""
        logger.info("Preparing train/test data...")

        # Remove rows with NaN in features or target
        df = df.dropna(subset=self.FEATURE_COLS + [self.TARGET_COL])

        # Replace infinite values
        df = df.replace([np.inf, -np.inf], 0)

        X = df[self.FEATURE_COLS].copy()
        y = df[self.TARGET_COL].copy()

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.FEATURE_COLS)

        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
        logger.info(f"Target distribution - Train: {y_train.value_counts().to_dict()}")
        logger.info(f"Target distribution - Test: {y_test.value_counts().to_dict()}")

        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        """Train XGBoost and LightGBM models."""
        logger.info("Training XGBoost model...")

        # XGBoost
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        self.xgb_model.fit(X_train, y_train)
        logger.info("XGBoost training complete")

        # LightGBM
        logger.info("Training LightGBM model...")
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
        )
        self.lgb_model.fit(X_train, y_train)
        logger.info("LightGBM training complete")

        self.is_trained = True

    def evaluate(self, X_test, y_test) -> dict:
        """Evaluate both models and the ensemble."""
        logger.info("Evaluating models...")

        results = {}

        # XGBoost predictions
        xgb_pred = self.xgb_model.predict(X_test)
        xgb_prob = self.xgb_model.predict_proba(X_test)[:, 1]

        # LightGBM predictions
        lgb_pred = self.lgb_model.predict(X_test)
        lgb_prob = self.lgb_model.predict_proba(X_test)[:, 1]

        # Ensemble (average probabilities)
        ensemble_prob = (xgb_prob + lgb_prob) / 2
        ensemble_pred = (ensemble_prob >= 0.5).astype(int)

        for name, pred, prob in [
            ("XGBoost", xgb_pred, xgb_prob),
            ("LightGBM", lgb_pred, lgb_prob),
            ("Ensemble", ensemble_pred, ensemble_prob),
        ]:
            acc = accuracy_score(y_test, pred)
            f1 = f1_score(y_test, pred, zero_division=0)
            try:
                auc = roc_auc_score(y_test, prob)
            except ValueError:
                auc = 0.5

            results[name] = {
                "accuracy": round(acc, 4),
                "f1_score": round(f1, 4),
                "auc_roc": round(auc, 4),
            }

            logger.info(f"\n{name} Results:")
            logger.info(f"  Accuracy: {acc:.4f}")
            logger.info(f"  F1 Score: {f1:.4f}")
            logger.info(f"  AUC-ROC:  {auc:.4f}")

        # Feature importance (XGBoost)
        importance = dict(zip(self.FEATURE_COLS, self.xgb_model.feature_importances_))
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        logger.info("\nFeature Importance (XGBoost):")
        for feat, imp in sorted_importance.items():
            logger.info(f"  {feat}: {imp:.4f}")

        results["feature_importance"] = {k: round(float(v), 4) for k, v in sorted_importance.items()}

        # Classification report for ensemble
        logger.info(f"\nEnsemble Classification Report:")
        logger.info(classification_report(y_test, ensemble_pred, zero_division=0))

        self.metrics = results
        return results

    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation."""
        logger.info(f"Running {cv}-fold cross-validation...")
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.xgb_model, X, y, cv=skf, scoring="roc_auc")
        logger.info(f"CV AUC-ROC: {scores.mean():.4f} (+/- {scores.std():.4f})")
        return {"cv_mean_auc": round(scores.mean(), 4), "cv_std_auc": round(scores.std(), 4)}

    def predict(self, features: dict) -> dict:
        """Make a prediction for a single match state."""
        if not self.is_trained:
            self.load_model()

        # Prepare features
        feature_row = pd.DataFrame([{col: features.get(col, 0) for col in self.FEATURE_COLS}])
        feature_scaled = self.scaler.transform(feature_row)

        # Ensemble prediction
        xgb_prob = self.xgb_model.predict_proba(feature_scaled)[0]
        lgb_prob = self.lgb_model.predict_proba(feature_scaled)[0]
        ensemble_prob = (xgb_prob + lgb_prob) / 2

        return {
            "batting_team_win_probability": round(float(ensemble_prob[1]), 4),
            "bowling_team_win_probability": round(float(ensemble_prob[0]), 4),
            "xgb_probability": round(float(xgb_prob[1]), 4),
            "lgb_probability": round(float(lgb_prob[1]), 4),
        }

    def save_model(self):
        """Save trained models and scaler."""
        logger.info("Saving models...")
        joblib.dump(self.xgb_model, MODELS_DIR / "xgb_win_predictor.joblib")
        joblib.dump(self.lgb_model, MODELS_DIR / "lgb_win_predictor.joblib")
        joblib.dump(self.scaler, MODELS_DIR / "win_predictor_scaler.joblib")

        # Save metrics
        with open(MODELS_DIR / "win_predictor_metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)

        logger.info(f"Models saved to {MODELS_DIR}")

    def load_model(self):
        """Load trained models."""
        logger.info("Loading models...")
        self.xgb_model = joblib.load(MODELS_DIR / "xgb_win_predictor.joblib")
        self.lgb_model = joblib.load(MODELS_DIR / "lgb_win_predictor.joblib")
        self.scaler = joblib.load(MODELS_DIR / "win_predictor_scaler.joblib")
        self.is_trained = True
        logger.info("Models loaded successfully")

    def log_to_mlflow(self):
        """Log models and metrics to MLflow."""
        try:
            import mlflow
            import mlflow.sklearn

            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
            mlflow.set_experiment("ICC_T20_Win_Prediction")

            with mlflow.start_run(run_name=f"win_predictor_{datetime.now().strftime('%Y%m%d_%H%M')}"):
                # Log parameters
                mlflow.log_params({
                    "model_type": "XGBoost + LightGBM Ensemble",
                    "xgb_n_estimators": 200,
                    "xgb_max_depth": 6,
                    "xgb_learning_rate": 0.1,
                    "lgb_n_estimators": 200,
                    "lgb_max_depth": 6,
                    "n_features": len(self.FEATURE_COLS),
                })

                # Log metrics
                if "Ensemble" in self.metrics:
                    for k, v in self.metrics["Ensemble"].items():
                        mlflow.log_metric(f"ensemble_{k}", v)
                if "XGBoost" in self.metrics:
                    for k, v in self.metrics["XGBoost"].items():
                        mlflow.log_metric(f"xgb_{k}", v)
                if "LightGBM" in self.metrics:
                    for k, v in self.metrics["LightGBM"].items():
                        mlflow.log_metric(f"lgb_{k}", v)

                # Log models
                mlflow.sklearn.log_model(self.xgb_model, "xgb_model")
                mlflow.sklearn.log_model(self.lgb_model, "lgb_model")

                # Log feature importance
                if "feature_importance" in self.metrics:
                    mlflow.log_dict(self.metrics["feature_importance"], "feature_importance.json")

                logger.info("Models and metrics logged to MLflow")

        except ImportError:
            logger.warning("MLflow not available, skipping model logging")
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")

    def run_full_training(self):
        """Execute the full training pipeline."""
        logger.info("=" * 60)
        logger.info("STARTING WIN PREDICTOR TRAINING")
        logger.info("=" * 60)

        # Load and prepare data
        df = self.load_data()
        X_train, X_test, y_train, y_test = self.prepare_data(df)

        # Train
        self.train(X_train, y_train)

        # Evaluate
        results = self.evaluate(X_test, y_test)

        # Cross-validate
        X_all = pd.DataFrame(
            self.scaler.transform(df[self.FEATURE_COLS].fillna(0).replace([np.inf, -np.inf], 0)),
            columns=self.FEATURE_COLS,
        )
        y_all = df[self.TARGET_COL].dropna()
        if len(X_all) == len(y_all):
            cv_results = self.cross_validate(X_all, y_all)
            self.metrics["cross_validation"] = cv_results

        # Save
        self.save_model()

        # Log to MLflow
        self.log_to_mlflow()

        logger.info("=" * 60)
        logger.info("WIN PREDICTOR TRAINING COMPLETE")
        logger.info("=" * 60)

        return results


if __name__ == "__main__":
    predictor = WinPredictor()
    predictor.run_full_training()
