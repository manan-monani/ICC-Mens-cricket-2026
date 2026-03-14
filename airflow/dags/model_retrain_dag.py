"""
Airflow DAG: Model Retraining Pipeline
Runs weekly on Sunday at 02:00 UTC.
Retrains all ML models and promotes if better.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

default_args = {
    "owner": "icc_t20_team",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

dag = DAG(
    "model_retrain_pipeline",
    default_args=default_args,
    description="Weekly model retraining: Feature Eng -> Train -> Evaluate -> Promote",
    schedule_interval="0 2 * * 0",  # Sunday 02:00 UTC
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ml", "retraining", "weekly"],
)


def update_features(**kwargs):
    """Regenerate ML features from latest data."""
    from ml.feature_engineering import FeatureEngineer
    fe = FeatureEngineer()
    fe.run_all()
    return "Features updated"


def train_win_predictor(**kwargs):
    """Train win prediction model."""
    from ml.train_win_predictor import WinPredictor
    predictor = WinPredictor()
    results = predictor.run_full_training()
    
    # Push metrics to XCom
    kwargs["ti"].xcom_push(key="win_predictor_auc", value=results.get("Ensemble", {}).get("auc_roc", 0))
    return results


def train_score_regressor(**kwargs):
    """Train score regression model."""
    from ml.train_score_regressor import ScoreRegressor
    regressor = ScoreRegressor()
    results = regressor.run_full_training()
    return results


def train_player_clusters(**kwargs):
    """Train player clustering model."""
    from ml.train_player_clusters import PlayerClusterer
    clusterer = PlayerClusterer()
    clusterer.run_full_training()
    return "Clustering complete"


def evaluate_and_promote(**kwargs):
    """Evaluate model quality and promote if good enough."""
    ti = kwargs["ti"]
    auc = ti.xcom_pull(task_ids="train_win_predictor", key="win_predictor_auc")
    
    threshold = 0.60  # Minimum AUC to promote
    if auc and auc >= threshold:
        return f"Model promoted to Production (AUC: {auc})"
    else:
        return f"Model not promoted (AUC: {auc} < threshold: {threshold})"


# Tasks
update_features_task = PythonOperator(
    task_id="update_features",
    python_callable=update_features,
    dag=dag,
)

train_win_task = PythonOperator(
    task_id="train_win_predictor",
    python_callable=train_win_predictor,
    dag=dag,
)

train_score_task = PythonOperator(
    task_id="train_score_regressor",
    python_callable=train_score_regressor,
    dag=dag,
)

train_clusters_task = PythonOperator(
    task_id="train_player_clusters",
    python_callable=train_player_clusters,
    dag=dag,
)

evaluate_task = PythonOperator(
    task_id="evaluate_and_promote",
    python_callable=evaluate_and_promote,
    dag=dag,
)

# Dependencies
update_features_task >> [train_win_task, train_score_task, train_clusters_task]
[train_win_task, train_score_task, train_clusters_task] >> evaluate_task
