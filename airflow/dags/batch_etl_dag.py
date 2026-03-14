"""
Airflow DAG: Batch ETL Pipeline
Runs daily at midnight - processes new data, transforms, and validates.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import sys
import os

# Add project to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

default_args = {
    "owner": "icc_t20_team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "batch_etl_pipeline",
    default_args=default_args,
    description="Daily batch ETL: Load CSV -> Bronze -> Silver/Gold -> Quality Checks",
    schedule_interval="0 0 * * *",  # Daily at midnight
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["etl", "batch", "daily"],
)


def run_batch_etl(**kwargs):
    """Execute the full batch ETL pipeline."""
    from etl.batch_etl import CricketETL
    etl = CricketETL()
    result = etl.run_full_pipeline()
    return "ETL completed successfully" if result else "ETL failed"


def run_feature_engineering(**kwargs):
    """Execute feature engineering after ETL."""
    from ml.feature_engineering import FeatureEngineer
    fe = FeatureEngineer()
    fe.run_all()
    return "Feature engineering completed"


def run_quality_validation(**kwargs):
    """Run data quality checks."""
    from etl.batch_etl import CricketETL
    etl = CricketETL()
    etl.connect()
    report = etl.run_quality_checks()
    etl.close()
    
    failed = report.get("failed", 0)
    if failed > 0:
        raise ValueError(f"Data quality checks failed: {failed} failures")
    return f"Quality checks passed: {report.get('passed', 0)} passed"


# Task 1: Run batch ETL
batch_etl_task = PythonOperator(
    task_id="run_batch_etl",
    python_callable=run_batch_etl,
    dag=dag,
)

# Task 2: Run feature engineering
feature_eng_task = PythonOperator(
    task_id="run_feature_engineering",
    python_callable=run_feature_engineering,
    dag=dag,
)

# Task 3: Data quality validation
dq_validation_task = PythonOperator(
    task_id="run_quality_validation",
    python_callable=run_quality_validation,
    dag=dag,
)

# Task dependencies: ETL -> Features -> DQ Checks
batch_etl_task >> feature_eng_task >> dq_validation_task
