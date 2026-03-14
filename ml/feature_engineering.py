"""
ICC T20 Predictor - Feature Engineering
Generates ML-ready features from the warehouse data for:
  1. Win Prediction (Classification)
  2. Score Prediction (Regression)
  3. Player Clustering (Unsupervised)
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FEATURE_ENG")

DB_PATH = PROCESSED_DATA_DIR / "icc_cricket.db"


class FeatureEngineer:
    """Generates ML features from warehouse data."""

    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH

    def _get_conn(self):
        return sqlite3.connect(str(self.db_path))

    def build_match_features(self) -> pd.DataFrame:
        """
        Build per-over features for win prediction.
        Each row = snapshot of match state at end of an over.
        Target: did the batting team win the match?
        """
        logger.info("Building match features for win prediction...")
        conn = self._get_conn()

        # Load deliveries
        deliveries = pd.read_sql("""
            SELECT match_id, innings_number, over_number, ball_in_over,
                   batting_team, runs_batter, runs_extras, runs_total,
                   is_wicket
            FROM raw_deliveries
        """, conn)

        # Load matches
        matches = pd.read_sql("""
            SELECT match_id, venue, city, toss_winner, toss_decision,
                   winner, result_type, result_margin, season, match_date
            FROM raw_matches
        """, conn)

        # Load team performance
        team_perf = pd.read_sql("SELECT * FROM agg_team_performance", conn)
        team_win_rates = dict(zip(team_perf["team_name"], team_perf["win_pct"]))

        # Load venue stats
        venue_stats = pd.read_sql("SELECT * FROM dim_venues", conn)
        venue_avg_scores = dict(zip(venue_stats["venue_name"], venue_stats["avg_first_innings_score"]))

        conn.close()

        # Filter: only matches with a valid winner
        valid_matches = matches[
            (matches["winner"].notna()) & 
            (matches["winner"] != "No Result") &
            (matches["winner"] != "")
        ]
        valid_match_ids = set(valid_matches["match_id"].unique())
        deliveries = deliveries[deliveries["match_id"].isin(valid_match_ids)]

        logger.info(f"Processing {len(valid_match_ids)} matches with valid results...")

        # Build per-over snapshots
        features_list = []

        for match_id, match_dels in deliveries.groupby("match_id"):
            match_info = matches[matches["match_id"] == match_id].iloc[0]
            winner = match_info["winner"]
            venue = match_info["venue"]
            toss_winner = match_info["toss_winner"]
            toss_decision = match_info["toss_decision"]

            for innings_num in [1, 2]:
                innings_data = match_dels[match_dels["innings_number"] == innings_num]
                if len(innings_data) == 0:
                    continue

                batting_team = innings_data["batting_team"].iloc[0]

                # Get first innings score (for 2nd innings RRR calc)
                first_innings_score = None
                if innings_num == 2:
                    inn1 = match_dels[match_dels["innings_number"] == 1]
                    first_innings_score = inn1["runs_total"].sum() if len(inn1) > 0 else 0

                # Cumulative over-by-over
                cumulative_runs = 0
                cumulative_wickets = 0
                over_runs = []
                last_3_over_runs = []

                for over_num in sorted(innings_data["over_number"].unique()):
                    over_data = innings_data[innings_data["over_number"] == over_num]
                    over_runs_val = over_data["runs_total"].sum()
                    over_wickets = over_data["is_wicket"].sum()

                    cumulative_runs += over_runs_val
                    cumulative_wickets += over_wickets
                    over_runs.append(over_runs_val)

                    balls_bowled = (over_num + 1) * 6
                    balls_remaining = 120 - balls_bowled
                    current_rr = (cumulative_runs / balls_bowled) * 6 if balls_bowled > 0 else 0

                    # Required run rate (2nd innings)
                    target = (first_innings_score + 1) if first_innings_score is not None else None
                    rrr = 0
                    if target and balls_remaining > 0:
                        remaining_runs = target - cumulative_runs
                        rrr = (remaining_runs / balls_remaining) * 6 if balls_remaining > 0 else 999

                    # Match phase
                    if over_num < 6:
                        phase = 0  # powerplay
                    elif over_num >= 16:
                        phase = 2  # death
                    else:
                        phase = 1  # middle

                    # Momentum (runs last 3 overs / wickets lost last 3 overs + 1)
                    last_3 = over_runs[-3:] if len(over_runs) >= 3 else over_runs
                    momentum = sum(last_3) / max(1, 1)

                    # Pressure index
                    pressure = 0
                    if innings_num == 2 and target and balls_remaining > 0:
                        pressure = (rrr - current_rr) / max(balls_remaining / 6, 1)

                    # Team features
                    bat_win_rate = team_win_rates.get(batting_team, 50)
                    venue_avg = venue_avg_scores.get(venue, 150)

                    # Toss feature
                    toss_winner_batting = 1 if toss_winner == batting_team else 0

                    # Target: did batting team win?
                    match_winner = 1 if winner == batting_team else 0

                    # Final score of this innings
                    final_score_innings = innings_data["runs_total"].sum()

                    features_list.append({
                        "match_id": match_id,
                        "innings_number": innings_num,
                        "over_number": over_num,
                        "match_phase": phase,
                        "current_score": cumulative_runs,
                        "current_wickets": cumulative_wickets,
                        "current_run_rate": round(current_rr, 2),
                        "required_run_rate": round(rrr, 2),
                        "balls_remaining": balls_remaining,
                        "bat_team_win_rate": bat_win_rate,
                        "venue_avg_score": venue_avg if venue_avg else 150,
                        "toss_winner_batting": toss_winner_batting,
                        "innings_number_enc": innings_num - 1,
                        "target": target if target else 0,
                        "pressure_index": round(pressure, 4),
                        "momentum_score": round(momentum, 2),
                        "powerplay_score": cumulative_runs if over_num == 5 else 0,
                        "match_winner": match_winner,
                        "final_score": final_score_innings,
                    })

        features_df = pd.DataFrame(features_list)
        logger.info(f"Generated {len(features_df)} feature rows for win prediction")

        # Save to DB
        conn = self._get_conn()
        features_df.to_sql("ml_match_features", conn, if_exists="replace", index=False)
        conn.close()
        logger.info("Features saved to ml_match_features table")

        return features_df

    def build_player_features(self) -> pd.DataFrame:
        """
        Build player-level features for clustering.
        Combines batting and bowling stats.
        """
        logger.info("Building player features for clustering...")
        conn = self._get_conn()

        batting = pd.read_sql("""
            SELECT player_name, team_name, matches_played,
                   total_runs, balls_faced, batting_average, strike_rate,
                   fours, sixes, dot_ball_pct, boundary_pct,
                   powerplay_sr, death_sr, consistency_score, form_index
            FROM agg_player_batting
            WHERE matches_played >= 5
        """, conn)

        bowling = pd.read_sql("""
            SELECT player_name, team_name as bowl_team, matches_played as bowl_matches,
                   balls_bowled, runs_conceded, wickets_taken, economy_rate,
                   bowling_average, bowling_strike_rate, dot_ball_pct as bowl_dot_pct,
                   powerplay_economy, death_economy
            FROM agg_player_bowling
            WHERE matches_played >= 5
        """, conn)

        conn.close()

        # Merge batting and bowling
        players = batting.merge(bowling, on="player_name", how="outer")

        # Fill NaN for players who only bat or bowl
        for col in players.columns:
            if players[col].dtype in [np.float64, np.int64]:
                players[col] = players[col].fillna(0)
            else:
                players[col] = players[col].fillna("")

        # Set team
        players["team"] = players["team_name"].where(players["team_name"] != "", players["bowl_team"])

        # Determine role
        def classify_role(row):
            has_batting = row["total_runs"] > 50
            has_bowling = row["wickets_taken"] > 5
            if has_batting and has_bowling:
                return "allrounder"
            elif has_bowling:
                return "bowler"
            else:
                return "batter"

        players["role"] = players.apply(classify_role, axis=1)

        # Select features for clustering
        cluster_features = [
            "batting_average", "strike_rate", "boundary_pct",
            "consistency_score", "form_index",
            "economy_rate", "wickets_taken", "bowling_strike_rate",
        ]

        logger.info(f"Player features for {len(players)} players")

        # Save
        conn = self._get_conn()
        players.to_sql("ml_player_features", conn, if_exists="replace", index=False)
        conn.close()

        return players

    def build_score_prediction_features(self) -> pd.DataFrame:
        """
        Build features at over 6, 10, 15 breakpoints for score regression.
        Target: final innings score
        """
        logger.info("Building score prediction features...")
        conn = self._get_conn()

        features = pd.read_sql("""
            SELECT match_id, innings_number, over_number,
                   current_score, current_wickets, current_run_rate,
                   match_phase, bat_team_win_rate, venue_avg_score,
                   final_score
            FROM ml_match_features
            WHERE over_number IN (5, 9, 14)
              AND innings_number = 1
        """, conn)

        conn.close()

        logger.info(f"Score prediction features: {len(features)} rows")
        return features

    def run_all(self):
        """Run all feature engineering pipelines."""
        logger.info("=" * 60)
        logger.info("STARTING FEATURE ENGINEERING")
        logger.info("=" * 60)

        match_features = self.build_match_features()
        player_features = self.build_player_features()
        score_features = self.build_score_prediction_features()

        logger.info("=" * 60)
        logger.info("FEATURE ENGINEERING COMPLETE")
        logger.info(f"  Match features: {len(match_features)} rows")
        logger.info(f"  Player features: {len(player_features)} rows")
        logger.info(f"  Score features: {len(score_features)} rows")
        logger.info("=" * 60)


if __name__ == "__main__":
    fe = FeatureEngineer()
    fe.run_all()
