"""
ICC T20 Predictor - Player Clustering
Algorithm: K-Means + PCA for visualization
Groups players into playing styles/archetypes.
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
import sys
import json
import joblib
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PLAYER_CLUSTERS")

DB_PATH = PROCESSED_DATA_DIR / "icc_cricket.db"
MODELS_DIR = PROCESSED_DATA_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Cluster labels
CLUSTER_LABELS = {
    0: "Anchor (Consistent Accumulator)",
    1: "Power Hitter (Aggressive Boundary Scorer)",
    2: "All-Rounder (Dual Threat)",
    3: "Specialist Bowler (Economy Expert)",
    4: "Impact Player (Match Winner)",
}


class PlayerClusterer:
    """K-Means clustering of T20 players by playing style."""

    CLUSTER_FEATURES = [
        "batting_average", "strike_rate", "boundary_pct",
        "consistency_score", "economy_rate", "wickets_taken",
    ]

    def __init__(self, n_clusters=5, db_path=None):
        self.n_clusters = n_clusters
        self.db_path = db_path or DB_PATH
        self.kmeans = None
        self.pca = None
        self.scaler = StandardScaler()
        self.cluster_centers = None
        self.player_clusters = None
        self.is_trained = False

    def load_data(self) -> pd.DataFrame:
        """Load player features."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            df = pd.read_sql("SELECT * FROM ml_player_features", conn)
        except Exception:
            # Fallback: merge batting and bowling
            batting = pd.read_sql("""
                SELECT player_name, team_name, matches_played,
                       batting_average, strike_rate, boundary_pct, consistency_score
                FROM agg_player_batting WHERE matches_played >= 5
            """, conn)
            bowling = pd.read_sql("""
                SELECT player_name, economy_rate, wickets_taken
                FROM agg_player_bowling WHERE matches_played >= 5
            """, conn)
            df = batting.merge(bowling, on="player_name", how="outer")
            df = df.fillna(0)
        conn.close()
        logger.info(f"Loaded {len(df)} players")
        return df

    def train(self, df: pd.DataFrame):
        """Train K-Means clustering."""
        logger.info(f"Training K-Means with {self.n_clusters} clusters...")

        # Prepare features
        available_features = [f for f in self.CLUSTER_FEATURES if f in df.columns]
        if len(available_features) < 3:
            logger.error(f"Not enough features available: {available_features}")
            return

        X = df[available_features].fillna(0).replace([np.inf, -np.inf], 0)

        # Scale
        X_scaled = self.scaler.fit_transform(X)

        # K-Means
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        clusters = self.kmeans.fit_predict(X_scaled)

        # PCA for visualization
        self.pca = PCA(n_components=2)
        pca_coords = self.pca.fit_transform(X_scaled)

        # Add results to DataFrame
        df = df.copy()
        df["cluster_id"] = clusters
        df["cluster_label"] = df["cluster_id"].map(CLUSTER_LABELS)
        df["pca_x"] = pca_coords[:, 0]
        df["pca_y"] = pca_coords[:, 1]

        self.player_clusters = df
        self.is_trained = True

        # Log cluster distribution
        for cid in range(self.n_clusters):
            count = (clusters == cid).sum()
            label = CLUSTER_LABELS.get(cid, f"Cluster {cid}")
            logger.info(f"  {label}: {count} players")

        # Save results
        self._save_results(df)

        logger.info("Clustering complete")
        return df

    def _save_results(self, df: pd.DataFrame):
        """Save clustering results."""
        conn = sqlite3.connect(str(self.db_path))
        cols_to_save = [
            "player_name", "team_name", "cluster_id", "cluster_label",
            "pca_x", "pca_y", "batting_average", "strike_rate",
            "boundary_pct", "consistency_score",
        ]
        save_cols = [c for c in cols_to_save if c in df.columns]
        df[save_cols].to_sql("ml_player_clusters", conn, if_exists="replace", index=False)
        conn.close()

        # Save models
        joblib.dump(self.kmeans, MODELS_DIR / "player_kmeans.joblib")
        joblib.dump(self.pca, MODELS_DIR / "player_pca.joblib")
        joblib.dump(self.scaler, MODELS_DIR / "player_cluster_scaler.joblib")

        # Save cluster summary
        summary = {}
        for cid in range(self.n_clusters):
            cluster_players = df[df["cluster_id"] == cid]
            summary[CLUSTER_LABELS.get(cid, f"Cluster {cid}")] = {
                "count": len(cluster_players),
                "avg_batting_avg": round(cluster_players["batting_average"].mean(), 2) if "batting_average" in cluster_players else 0,
                "avg_strike_rate": round(cluster_players["strike_rate"].mean(), 2) if "strike_rate" in cluster_players else 0,
                "sample_players": cluster_players["player_name"].head(5).tolist(),
            }
        with open(MODELS_DIR / "player_cluster_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Results saved to database and {MODELS_DIR}")

    def log_to_mlflow(self):
        """Log to MLflow."""
        try:
            import mlflow
            import os

            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "mlruns"))
            mlflow.set_experiment("ICC_T20_Player_Clustering")

            with mlflow.start_run(run_name="player_clusters"):
                mlflow.log_params({"n_clusters": self.n_clusters, "algorithm": "KMeans"})
                mlflow.log_metric("inertia", float(self.kmeans.inertia_))
                mlflow.sklearn.log_model(self.kmeans, "kmeans_model")
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")

    def run_full_training(self):
        """Full clustering pipeline."""
        logger.info("=" * 60)
        logger.info("STARTING PLAYER CLUSTERING")
        logger.info("=" * 60)

        df = self.load_data()
        if len(df) < self.n_clusters:
            logger.warning("Not enough players for clustering")
            return None

        result = self.train(df)
        self.log_to_mlflow()

        logger.info("PLAYER CLUSTERING COMPLETE")
        return result


if __name__ == "__main__":
    clusterer = PlayerClusterer()
    clusterer.run_full_training()
