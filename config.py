"""
ICC T20 Predictor - Shared Configuration Module
Centralizes all configuration, database connections, and path management.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ===================== PATHS =====================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
DATASET_DIR = PROJECT_ROOT / "Dataset"

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, EMBEDDINGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ===================== DATABASE =====================
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "icc_cricket")
POSTGRES_USER = os.getenv("POSTGRES_USER", "admin")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "secret")

DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# ===================== KAFKA =====================
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPICS = {
    "ball_events": "ball_events",
    "match_state": "match_state",
    "player_updates": "player_updates",
    "venue_conditions": "venue_conditions",
}

# ===================== MLFLOW =====================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# ===================== GENAI =====================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# ===================== APP =====================
APP_ENV = os.getenv("APP_ENV", "development")
APP_DEBUG = os.getenv("APP_DEBUG", "true").lower() == "true"
APP_PORT = int(os.getenv("APP_PORT", "8000"))
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))

# ===================== DATA FILES =====================
DATASET_FILES = {
    "matches": DATASET_DIR / "matches.csv",
    "deliveries": DATASET_DIR / "deliveries.csv",
    "innings": DATASET_DIR / "innings.csv",
    "match_teams": DATASET_DIR / "match_teams.csv",
    "officials": DATASET_DIR / "officials.csv",
    "player_of_match": DATASET_DIR / "player_of_match.csv",
    "powerplays": DATASET_DIR / "powerplays.csv",
    "wickets": DATASET_DIR / "wickets.csv",
}

# ===================== SIMULATION =====================
SIMULATION_INTERVAL_SECONDS = 5  # One ball every 5 seconds
MATCH_OVERS = 20
BALLS_PER_OVER = 6

# ===================== ML MODELS =====================
MODEL_NAMES = {
    "win_predictor": "WinPredictor",
    "score_regressor": "ScoreRegressor",
    "player_clusters": "PlayerClusters",
}

# Ball outcome probabilities for simulation
BALL_OUTCOMES = {
    0: 0.40,    # Dot ball
    1: 0.25,    # Single
    2: 0.10,    # Double
    3: 0.05,    # Triple
    4: 0.10,    # Four
    6: 0.05,    # Six
    "wicket": 0.05  # Wicket
}
