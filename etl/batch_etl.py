"""
ICC T20 Predictor - ETL Batch Pipeline
Handles the complete ETL process:
  1. Extract: Read raw CSVs from Dataset/
  2. Transform: Clean, filter (men's only), normalize, quality checks
  3. Load: Store into SQLite (local) or PostgreSQL warehouse

This is the main batch ETL that processes historical data.
Uses SQLite by default for local development (no PostgreSQL needed).
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATASET_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT

# ===================== LOGGING =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "logs" / "etl_batch.log", mode="a"),
    ],
)
logger = logging.getLogger("ETL_BATCH")

# Ensure logs dir
(PROJECT_ROOT / "logs").mkdir(exist_ok=True)

# SQLite for local dev (no PostgreSQL required)
DB_PATH = PROCESSED_DATA_DIR / "icc_cricket.db"


class CricketETL:
    """Complete ETL pipeline for ICC T20 Cricket data."""

    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH
        self.conn = None
        self.stats = {
            "total_matches_raw": 0,
            "male_matches": 0,
            "total_deliveries_raw": 0,
            "male_deliveries": 0,
            "players_extracted": 0,
            "venues_extracted": 0,
            "teams_extracted": 0,
        }

    def connect(self):
        """Connect to SQLite database."""
        self.conn = sqlite3.connect(str(self.db_path))
        logger.info(f"Connected to database: {self.db_path}")

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    # ===================== SCHEMA CREATION =====================

    def create_schema(self):
        """Create all warehouse tables in SQLite."""
        logger.info("Creating database schema...")
        cursor = self.conn.cursor()

        # --- Bronze Layer: Raw Tables ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS raw_matches (
                match_id INTEGER PRIMARY KEY,
                data_version TEXT,
                created TEXT,
                revision INTEGER,
                match_date TEXT,
                season TEXT,
                event_name TEXT,
                event_match_number INTEGER,
                match_type TEXT,
                match_type_number INTEGER,
                gender TEXT,
                team_type TEXT,
                venue TEXT,
                city TEXT,
                overs INTEGER,
                balls_per_over INTEGER,
                toss_winner TEXT,
                toss_decision TEXT,
                winner TEXT,
                result_type TEXT,
                result_margin REAL,
                result_text TEXT,
                method TEXT,
                ingestion_timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS raw_deliveries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER,
                innings_number INTEGER,
                over_number INTEGER,
                ball_in_over INTEGER,
                batting_team TEXT,
                batter TEXT,
                bowler TEXT,
                non_striker TEXT,
                runs_batter INTEGER DEFAULT 0,
                runs_extras INTEGER DEFAULT 0,
                runs_total INTEGER DEFAULT 0,
                extras_byes INTEGER DEFAULT 0,
                extras_legbyes INTEGER DEFAULT 0,
                extras_noballs INTEGER DEFAULT 0,
                extras_wides INTEGER DEFAULT 0,
                extras_penalty INTEGER DEFAULT 0,
                is_wicket INTEGER DEFAULT 0,
                wicket_kind TEXT,
                wicket_player_out TEXT,
                wicket_fielders TEXT,
                ingestion_timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                source TEXT DEFAULT 'batch'
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS raw_innings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER,
                innings_number INTEGER,
                team TEXT,
                total_runs INTEGER,
                total_wickets INTEGER,
                total_balls INTEGER,
                extras_byes INTEGER DEFAULT 0,
                extras_legbyes INTEGER DEFAULT 0,
                extras_noballs INTEGER DEFAULT 0,
                extras_wides INTEGER DEFAULT 0,
                extras_penalty INTEGER DEFAULT 0,
                ingestion_timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # --- Dimension Tables (Gold Layer) ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dim_teams (
                team_id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_name TEXT NOT NULL UNIQUE,
                matches_played INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dim_players (
                player_id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_name TEXT NOT NULL,
                primary_team TEXT,
                role TEXT,
                matches_played INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(player_name, primary_team)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dim_venues (
                venue_id INTEGER PRIMARY KEY AUTOINCREMENT,
                venue_name TEXT NOT NULL,
                city TEXT,
                matches_played INTEGER DEFAULT 0,
                avg_first_innings_score REAL,
                avg_second_innings_score REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(venue_name, city)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dim_dates (
                date_id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_date TEXT NOT NULL UNIQUE,
                year INTEGER,
                month INTEGER,
                day INTEGER,
                day_of_week TEXT,
                season TEXT
            )
        """)

        # --- Fact Tables (Gold Layer) ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fact_match_results (
                match_id INTEGER PRIMARY KEY,
                team1_id INTEGER,
                team2_id INTEGER,
                winner_id INTEGER,
                venue_id INTEGER,
                date_id INTEGER,
                season TEXT,
                event_name TEXT,
                toss_winner_id INTEGER,
                toss_decision TEXT,
                result_type TEXT,
                result_margin REAL,
                team1_score INTEGER,
                team1_wickets INTEGER,
                team1_overs REAL,
                team2_score INTEGER,
                team2_wickets INTEGER,
                team2_overs REAL,
                first_bat_team_id INTEGER,
                first_bat_won INTEGER,
                method TEXT,
                FOREIGN KEY (team1_id) REFERENCES dim_teams(team_id),
                FOREIGN KEY (team2_id) REFERENCES dim_teams(team_id),
                FOREIGN KEY (winner_id) REFERENCES dim_teams(team_id),
                FOREIGN KEY (venue_id) REFERENCES dim_venues(venue_id),
                FOREIGN KEY (date_id) REFERENCES dim_dates(date_id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fact_ball_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER,
                innings_number INTEGER,
                over_number INTEGER,
                ball_in_over INTEGER,
                batting_team_id INTEGER,
                batter_id INTEGER,
                bowler_id INTEGER,
                venue_id INTEGER,
                date_id INTEGER,
                runs_batter INTEGER DEFAULT 0,
                runs_extras INTEGER DEFAULT 0,
                runs_total INTEGER DEFAULT 0,
                is_boundary_four INTEGER DEFAULT 0,
                is_boundary_six INTEGER DEFAULT 0,
                is_dot_ball INTEGER DEFAULT 0,
                is_wicket INTEGER DEFAULT 0,
                wicket_kind TEXT,
                match_phase TEXT,
                cumulative_score INTEGER,
                cumulative_wickets INTEGER,
                current_run_rate REAL,
                UNIQUE(match_id, innings_number, over_number, ball_in_over)
            )
        """)

        # --- Aggregation Tables ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agg_player_batting (
                player_id INTEGER PRIMARY KEY,
                player_name TEXT,
                team_name TEXT,
                matches_played INTEGER DEFAULT 0,
                innings_batted INTEGER DEFAULT 0,
                total_runs INTEGER DEFAULT 0,
                balls_faced INTEGER DEFAULT 0,
                batting_average REAL,
                strike_rate REAL,
                highest_score INTEGER DEFAULT 0,
                fours INTEGER DEFAULT 0,
                sixes INTEGER DEFAULT 0,
                dot_ball_pct REAL,
                boundary_pct REAL,
                powerplay_sr REAL,
                death_sr REAL,
                consistency_score REAL,
                form_index REAL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agg_player_bowling (
                player_id INTEGER PRIMARY KEY,
                player_name TEXT,
                team_name TEXT,
                matches_played INTEGER DEFAULT 0,
                innings_bowled INTEGER DEFAULT 0,
                balls_bowled INTEGER DEFAULT 0,
                runs_conceded INTEGER DEFAULT 0,
                wickets_taken INTEGER DEFAULT 0,
                economy_rate REAL,
                bowling_average REAL,
                bowling_strike_rate REAL,
                dot_ball_pct REAL,
                powerplay_economy REAL,
                death_economy REAL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agg_team_performance (
                team_id INTEGER PRIMARY KEY,
                team_name TEXT,
                matches_played INTEGER DEFAULT 0,
                matches_won INTEGER DEFAULT 0,
                matches_lost INTEGER DEFAULT 0,
                win_pct REAL,
                avg_score_batting REAL,
                avg_run_rate REAL,
                toss_win_pct REAL,
                bat_first_win_pct REAL,
                chase_win_pct REAL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # --- ML Feature Store ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ml_match_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER,
                innings_number INTEGER,
                over_number INTEGER,
                venue_id INTEGER,
                team_batting_id INTEGER,
                team_bowling_id INTEGER,
                toss_winner_batting INTEGER,
                current_score INTEGER,
                current_wickets INTEGER,
                current_run_rate REAL,
                required_run_rate REAL,
                balls_remaining INTEGER,
                team_bat_win_rate REAL,
                team_bowl_win_rate REAL,
                head_to_head_wins INTEGER,
                team_avg_score_venue REAL,
                powerplay_score INTEGER,
                powerplay_wickets INTEGER,
                pressure_index REAL,
                momentum_score REAL,
                match_winner INTEGER,
                final_score INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        self.conn.commit()
        logger.info("Schema created successfully")

    # ===================== EXTRACT =====================

    def extract_matches(self) -> pd.DataFrame:
        """Extract and filter matches - MEN'S T20 ONLY."""
        logger.info("Extracting matches data...")
        df = pd.read_csv(DATASET_DIR / "matches.csv")
        self.stats["total_matches_raw"] = len(df)
        logger.info(f"  Raw matches: {len(df)}")

        # Filter: Men's data only
        df = df[df["gender"] == "male"].copy()
        self.stats["male_matches"] = len(df)
        logger.info(f"  Male T20 matches: {len(df)}")

        return df

    def extract_deliveries(self, male_match_ids: set) -> pd.DataFrame:
        """Extract deliveries for male matches only."""
        logger.info("Extracting deliveries data (this may take a moment)...")
        df = pd.read_csv(DATASET_DIR / "deliveries.csv")
        self.stats["total_deliveries_raw"] = len(df)
        logger.info(f"  Raw deliveries: {len(df)}")

        # Filter to male matches only
        df = df[df["match_id"].isin(male_match_ids)].copy()
        self.stats["male_deliveries"] = len(df)
        logger.info(f"  Male T20 deliveries: {len(df)}")

        return df

    def extract_innings(self, male_match_ids: set) -> pd.DataFrame:
        """Extract innings for male matches only."""
        logger.info("Extracting innings data...")
        df = pd.read_csv(DATASET_DIR / "innings.csv")
        df = df[df["match_id"].isin(male_match_ids)].copy()
        logger.info(f"  Male T20 innings: {len(df)}")
        return df

    def extract_wickets(self, male_match_ids: set) -> pd.DataFrame:
        """Extract wickets for male matches only."""
        logger.info("Extracting wickets data...")
        df = pd.read_csv(DATASET_DIR / "wickets.csv")
        df = df[df["match_id"].isin(male_match_ids)].copy()
        logger.info(f"  Male T20 wickets: {len(df)}")
        return df

    def extract_player_of_match(self, male_match_ids: set) -> pd.DataFrame:
        """Extract player of match for male matches."""
        logger.info("Extracting player of match data...")
        df = pd.read_csv(DATASET_DIR / "player_of_match.csv")
        df = df[df["match_id"].isin(male_match_ids)].copy()
        logger.info(f"  Male T20 player of match: {len(df)}")
        return df

    # ===================== TRANSFORM =====================

    def transform_matches(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and transform matches data."""
        logger.info("Transforming matches data...")

        # Handle missing values
        df["city"] = df["city"].fillna("Unknown")
        df["winner"] = df["winner"].fillna("No Result")
        df["result_type"] = df["result_type"].fillna("no result")
        df["result_margin"] = df["result_margin"].fillna(0)
        df["method"] = df["method"].fillna("")
        df["event_name"] = df["event_name"].fillna("Unknown Event")
        df["event_match_number"] = df["event_match_number"].fillna(0).astype(int)
        df["match_type_number"] = df["match_type_number"].fillna(0).astype(int)

        # Date parsing
        df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")

        # Normalize team names (trim whitespace)
        for col in ["toss_winner", "winner"]:
            df[col] = df[col].str.strip()

        # Remove duplicates
        df = df.drop_duplicates(subset=["match_id"])

        # Data validation
        invalid_overs = df[df["overs"].isna() | (df["overs"] <= 0)]
        if len(invalid_overs) > 0:
            logger.warning(f"  Found {len(invalid_overs)} matches with invalid overs, setting to 20")
            df.loc[df["overs"].isna() | (df["overs"] <= 0), "overs"] = 20

        logger.info(f"  Transformed matches: {len(df)}")
        return df

    def transform_deliveries(self, df: pd.DataFrame, wickets_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and transform deliveries with wicket info merged."""
        logger.info("Transforming deliveries data...")

        # Handle missing values
        numeric_cols = [
            "runs_batter", "runs_extras", "runs_total",
            "extras_byes", "extras_legbyes", "extras_noballs",
            "extras_wides", "extras_penalty",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)

        # Rename 'over' to 'over_number' if needed
        if "over" in df.columns and "over_number" not in df.columns:
            df = df.rename(columns={"over": "over_number"})

        # Merge wicket information
        if wickets_df is not None and len(wickets_df) > 0:
            wickets_df = wickets_df.rename(columns={"over": "over_number"})
            wickets_merge = wickets_df[
                ["match_id", "innings_number", "over_number", "ball_in_over", "player_out", "kind", "fielders"]
            ].copy()
            wickets_merge["is_wicket"] = 1

            df = df.merge(
                wickets_merge,
                on=["match_id", "innings_number", "over_number", "ball_in_over"],
                how="left",
                suffixes=("", "_wicket"),
            )
            df["is_wicket"] = df["is_wicket"].fillna(0).astype(int)
            df["wicket_kind"] = df.get("kind", pd.Series(dtype=str))
            df["wicket_player_out"] = df.get("player_out", pd.Series(dtype=str))
            df["wicket_fielders"] = df.get("fielders", pd.Series(dtype=str))
        else:
            df["is_wicket"] = 0
            df["wicket_kind"] = None
            df["wicket_player_out"] = None
            df["wicket_fielders"] = None

        # Remove review/replacement columns (not needed for analysis)
        drop_cols = [
            "review_by", "review_batter", "review_decision", "review_type",
            "replacement_role", "replacement_team", "replacement_in", "replacement_out",
            "kind", "player_out", "fielders",
        ]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        # Validate: runs_batter should be 0-6
        invalid_runs = df[df["runs_batter"] > 6]
        if len(invalid_runs) > 0:
            logger.warning(f"  Found {len(invalid_runs)} deliveries with runs > 6, capping at 6")
            df.loc[df["runs_batter"] > 6, "runs_batter"] = 6

        # Validate: over should be 0-19
        invalid_overs = df[(df["over_number"] < 0) | (df["over_number"] > 19)]
        if len(invalid_overs) > 0:
            logger.warning(f"  Found {len(invalid_overs)} deliveries with invalid over numbers")

        # Remove complete duplicates
        df = df.drop_duplicates(
            subset=["match_id", "innings_number", "over_number", "ball_in_over", "batter", "bowler"]
        )

        logger.info(f"  Transformed deliveries: {len(df)}")
        return df

    def transform_innings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean innings data."""
        logger.info("Transforming innings data...")
        numeric_cols = [
            "total_runs", "total_wickets", "total_balls",
            "extras_byes", "extras_legbyes", "extras_noballs",
            "extras_wides", "extras_penalty",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)

        df = df.drop_duplicates(subset=["match_id", "innings_number"])
        logger.info(f"  Transformed innings: {len(df)}")
        return df

    # ===================== DIMENSION BUILDERS =====================

    def build_dim_teams(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Extract unique teams and their match counts."""
        logger.info("Building dim_teams...")
        match_teams = pd.read_csv(DATASET_DIR / "match_teams.csv")
        male_match_ids = set(matches_df["match_id"].unique())
        match_teams = match_teams[match_teams["match_id"].isin(male_match_ids)]

        teams = match_teams.groupby("team").size().reset_index(name="matches_played")
        teams = teams.rename(columns={"team": "team_name"})
        self.stats["teams_extracted"] = len(teams)
        logger.info(f"  Teams extracted: {len(teams)}")
        return teams

    def build_dim_players(self, deliveries_df: pd.DataFrame) -> pd.DataFrame:
        """Extract unique players from deliveries data."""
        logger.info("Building dim_players...")

        # Get all batters
        batters = deliveries_df[["batter", "batting_team"]].drop_duplicates()
        batters = batters.rename(columns={"batter": "player_name", "batting_team": "primary_team"})

        # Get all bowlers
        bowlers = deliveries_df[["bowler", "batting_team"]].drop_duplicates()
        # Bowler's team is NOT the batting team - we'll handle this later
        bowlers = bowlers.rename(columns={"bowler": "player_name", "batting_team": "opponent_team"})
        bowlers = bowlers.drop(columns=["opponent_team"])

        # Combine
        all_players = pd.concat([batters[["player_name", "primary_team"]], bowlers[["player_name"]]])
        all_players = all_players.drop_duplicates(subset=["player_name"])

        # Determine player role based on data
        batter_names = set(batters["player_name"].unique())
        bowler_names = set(bowlers["player_name"].unique())
        allrounders = batter_names & bowler_names

        def get_role(name):
            if name in allrounders:
                return "allrounder"
            elif name in bowler_names and name not in batter_names:
                return "bowler"
            else:
                return "batter"

        all_players["role"] = all_players["player_name"].apply(get_role)
        all_players["primary_team"] = all_players["primary_team"].fillna("Unknown")

        self.stats["players_extracted"] = len(all_players)
        logger.info(f"  Players extracted: {len(all_players)}")
        return all_players

    def build_dim_venues(self, matches_df: pd.DataFrame, innings_df: pd.DataFrame) -> pd.DataFrame:
        """Extract unique venues with aggregated stats."""
        logger.info("Building dim_venues...")

        venues = matches_df[["venue", "city"]].drop_duplicates()
        venues = venues.rename(columns={"venue": "venue_name"})

        # Count matches per venue
        venue_counts = matches_df.groupby(["venue", "city"]).size().reset_index(name="matches_played")
        venue_counts = venue_counts.rename(columns={"venue": "venue_name"})
        venues = venues.merge(venue_counts, on=["venue_name", "city"], how="left")

        # Calculate avg scores per venue
        match_venues = matches_df[["match_id", "venue", "city"]].rename(columns={"venue": "venue_name"})
        innings_with_venue = innings_df.merge(
            matches_df[["match_id", "venue", "city"]].rename(columns={"venue": "venue_name"}),
            on="match_id",
            how="left"
        )

        first_innings = innings_with_venue[innings_with_venue["innings_number"] == 1]
        second_innings = innings_with_venue[innings_with_venue["innings_number"] == 2]

        avg_1st = first_innings.groupby(["venue_name", "city"])["total_runs"].mean().reset_index()
        avg_1st.columns = ["venue_name", "city", "avg_first_innings_score"]

        avg_2nd = second_innings.groupby(["venue_name", "city"])["total_runs"].mean().reset_index()
        avg_2nd.columns = ["venue_name", "city", "avg_second_innings_score"]

        venues = venues.merge(avg_1st, on=["venue_name", "city"], how="left")
        venues = venues.merge(avg_2nd, on=["venue_name", "city"], how="left")

        self.stats["venues_extracted"] = len(venues)
        logger.info(f"  Venues extracted: {len(venues)}")
        return venues

    def build_dim_dates(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Extract unique dates."""
        logger.info("Building dim_dates...")
        dates = matches_df[["match_date", "season"]].drop_duplicates(subset=["match_date"])
        dates = dates.dropna(subset=["match_date"])
        dates["year"] = dates["match_date"].dt.year
        dates["month"] = dates["match_date"].dt.month
        dates["day"] = dates["match_date"].dt.day
        dates["day_of_week"] = dates["match_date"].dt.day_name()
        dates = dates.rename(columns={"match_date": "full_date"})
        dates["full_date"] = dates["full_date"].astype(str)
        logger.info(f"  Dates extracted: {len(dates)}")
        return dates

    # ===================== AGGREGATION BUILDERS =====================

    def build_agg_player_batting(self, deliveries_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Build aggregated batting statistics for all players."""
        logger.info("Building agg_player_batting...")

        # Add match phase based on over number
        deliveries_df = deliveries_df.copy()
        deliveries_df["match_phase"] = deliveries_df["over_number"].apply(
            lambda x: "powerplay" if x < 6 else ("death" if x >= 16 else "middle")
        )

        # Group by batter
        batter_stats = deliveries_df.groupby("batter").agg(
            total_runs=("runs_batter", "sum"),
            balls_faced=("runs_batter", "count"),
            fours=("runs_batter", lambda x: (x == 4).sum()),
            sixes=("runs_batter", lambda x: (x == 6).sum()),
            dot_balls=("runs_batter", lambda x: (x == 0).sum()),
        ).reset_index()

        batter_stats = batter_stats.rename(columns={"batter": "player_name"})

        # Calculate derived metrics
        batter_stats["strike_rate"] = np.where(
            batter_stats["balls_faced"] > 0,
            (batter_stats["total_runs"] / batter_stats["balls_faced"]) * 100,
            0,
        )
        batter_stats["dot_ball_pct"] = np.where(
            batter_stats["balls_faced"] > 0,
            (batter_stats["dot_balls"] / batter_stats["balls_faced"]) * 100,
            0,
        )
        batter_stats["boundary_pct"] = np.where(
            batter_stats["balls_faced"] > 0,
            ((batter_stats["fours"] + batter_stats["sixes"]) / batter_stats["balls_faced"]) * 100,
            0,
        )

        # Innings count & matches
        innings = deliveries_df.groupby("batter")["match_id"].nunique().reset_index()
        innings.columns = ["player_name", "matches_played"]
        batter_stats = batter_stats.merge(innings, on="player_name", how="left")

        # Batting average (runs / dismissals)
        if "is_wicket" in deliveries_df.columns and "wicket_player_out" in deliveries_df.columns:
            dismissals = deliveries_df[deliveries_df["is_wicket"] == 1].groupby("wicket_player_out").size().reset_index(name="dismissals")
            dismissals = dismissals.rename(columns={"wicket_player_out": "player_name"})
            batter_stats = batter_stats.merge(dismissals, on="player_name", how="left")
            batter_stats["dismissals"] = batter_stats["dismissals"].fillna(0)
        else:
            batter_stats["dismissals"] = 0

        batter_stats["batting_average"] = np.where(
            batter_stats["dismissals"] > 0,
            batter_stats["total_runs"] / batter_stats["dismissals"],
            batter_stats["total_runs"],
        )

        # Highest score per match
        match_scores = deliveries_df.groupby(["batter", "match_id"])["runs_batter"].sum().reset_index()
        highest = match_scores.groupby("batter")["runs_batter"].max().reset_index()
        highest.columns = ["player_name", "highest_score"]
        batter_stats = batter_stats.merge(highest, on="player_name", how="left")

        # Powerplay & Death over strike rates
        pp_data = deliveries_df[deliveries_df["match_phase"] == "powerplay"]
        pp_sr = pp_data.groupby("batter").agg(
            pp_runs=("runs_batter", "sum"),
            pp_balls=("runs_batter", "count"),
        ).reset_index()
        pp_sr["powerplay_sr"] = np.where(pp_sr["pp_balls"] > 0, (pp_sr["pp_runs"] / pp_sr["pp_balls"]) * 100, 0)
        pp_sr = pp_sr.rename(columns={"batter": "player_name"})[["player_name", "powerplay_sr"]]

        death_data = deliveries_df[deliveries_df["match_phase"] == "death"]
        death_sr = death_data.groupby("batter").agg(
            d_runs=("runs_batter", "sum"),
            d_balls=("runs_batter", "count"),
        ).reset_index()
        death_sr["death_sr"] = np.where(death_sr["d_balls"] > 0, (death_sr["d_runs"] / death_sr["d_balls"]) * 100, 0)
        death_sr = death_sr.rename(columns={"batter": "player_name"})[["player_name", "death_sr"]]

        batter_stats = batter_stats.merge(pp_sr, on="player_name", how="left")
        batter_stats = batter_stats.merge(death_sr, on="player_name", how="left")

        # Consistency score (lower std dev = more consistent)
        consistency = match_scores.groupby("batter")["runs_batter"].std().reset_index()
        consistency.columns = ["player_name", "score_std"]
        consistency["consistency_score"] = np.where(
            consistency["score_std"] > 0,
            1 / (1 + consistency["score_std"]),
            1,
        )
        batter_stats = batter_stats.merge(
            consistency[["player_name", "consistency_score"]], on="player_name", how="left"
        )

        # Form index (weighted avg of last 5 matches)
        match_dates = matches_df[["match_id", "match_date"]].copy()
        match_scores_dated = match_scores.merge(match_dates, on="match_id", how="left")
        match_scores_dated = match_scores_dated.sort_values("match_date", ascending=False)

        form_data = []
        for player, group in match_scores_dated.groupby("batter"):
            last5 = group.head(5)["runs_batter"].values
            weights = np.array([5, 4, 3, 2, 1][:len(last5)])
            form_idx = np.average(last5, weights=weights) if len(last5) > 0 else 0
            form_data.append({"player_name": player, "form_index": form_idx})

        form_df = pd.DataFrame(form_data)
        batter_stats = batter_stats.merge(form_df, on="player_name", how="left")

        # Get primary team
        team_map = deliveries_df.groupby("batter")["batting_team"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown").reset_index()
        team_map.columns = ["player_name", "team_name"]
        batter_stats = batter_stats.merge(team_map, on="player_name", how="left")

        # Select final columns
        batter_stats["innings_batted"] = batter_stats["matches_played"]
        cols = [
            "player_name", "team_name", "matches_played", "innings_batted",
            "total_runs", "balls_faced", "batting_average", "strike_rate",
            "highest_score", "fours", "sixes", "dot_ball_pct", "boundary_pct",
            "powerplay_sr", "death_sr", "consistency_score", "form_index",
        ]
        batter_stats = batter_stats[[c for c in cols if c in batter_stats.columns]]
        batter_stats = batter_stats.fillna(0)

        logger.info(f"  Batting stats for {len(batter_stats)} players")
        return batter_stats

    def build_agg_player_bowling(self, deliveries_df: pd.DataFrame) -> pd.DataFrame:
        """Build aggregated bowling statistics for all players."""
        logger.info("Building agg_player_bowling...")

        deliveries_df = deliveries_df.copy()
        deliveries_df["match_phase"] = deliveries_df["over_number"].apply(
            lambda x: "powerplay" if x < 6 else ("death" if x >= 16 else "middle")
        )

        bowler_stats = deliveries_df.groupby("bowler").agg(
            balls_bowled=("runs_total", "count"),
            runs_conceded=("runs_total", "sum"),
            dot_balls=("runs_total", lambda x: (x == 0).sum()),
        ).reset_index()
        bowler_stats = bowler_stats.rename(columns={"bowler": "player_name"})

        # Wickets
        if "is_wicket" in deliveries_df.columns:
            wickets = deliveries_df[deliveries_df["is_wicket"] == 1].groupby("bowler").size().reset_index(name="wickets_taken")
            wickets = wickets.rename(columns={"bowler": "player_name"})
            bowler_stats = bowler_stats.merge(wickets, on="player_name", how="left")
            bowler_stats["wickets_taken"] = bowler_stats["wickets_taken"].fillna(0).astype(int)
        else:
            bowler_stats["wickets_taken"] = 0

        # Derived metrics
        bowler_stats["economy_rate"] = np.where(
            bowler_stats["balls_bowled"] > 0,
            (bowler_stats["runs_conceded"] / bowler_stats["balls_bowled"]) * 6,
            0,
        )
        bowler_stats["bowling_average"] = np.where(
            bowler_stats["wickets_taken"] > 0,
            bowler_stats["runs_conceded"] / bowler_stats["wickets_taken"],
            999,
        )
        bowler_stats["bowling_strike_rate"] = np.where(
            bowler_stats["wickets_taken"] > 0,
            bowler_stats["balls_bowled"] / bowler_stats["wickets_taken"],
            999,
        )
        bowler_stats["dot_ball_pct"] = np.where(
            bowler_stats["balls_bowled"] > 0,
            (bowler_stats["dot_balls"] / bowler_stats["balls_bowled"]) * 100,
            0,
        )

        # Matches & innings
        innings = deliveries_df.groupby("bowler")["match_id"].nunique().reset_index()
        innings.columns = ["player_name", "matches_played"]
        bowler_stats = bowler_stats.merge(innings, on="player_name", how="left")
        bowler_stats["innings_bowled"] = bowler_stats["matches_played"]

        # Powerplay & Death economy
        pp_data = deliveries_df[deliveries_df["match_phase"] == "powerplay"]
        pp_eco = pp_data.groupby("bowler").agg(
            pp_runs=("runs_total", "sum"),
            pp_balls=("runs_total", "count"),
        ).reset_index()
        pp_eco["powerplay_economy"] = np.where(pp_eco["pp_balls"] > 0, (pp_eco["pp_runs"] / pp_eco["pp_balls"]) * 6, 0)
        pp_eco = pp_eco.rename(columns={"bowler": "player_name"})[["player_name", "powerplay_economy"]]

        death_data = deliveries_df[deliveries_df["match_phase"] == "death"]
        death_eco = death_data.groupby("bowler").agg(
            d_runs=("runs_total", "sum"),
            d_balls=("runs_total", "count"),
        ).reset_index()
        death_eco["death_economy"] = np.where(death_eco["d_balls"] > 0, (death_eco["d_runs"] / death_eco["d_balls"]) * 6, 0)
        death_eco = death_eco.rename(columns={"bowler": "player_name"})[["player_name", "death_economy"]]

        bowler_stats = bowler_stats.merge(pp_eco, on="player_name", how="left")
        bowler_stats = bowler_stats.merge(death_eco, on="player_name", how="left")

        # Get team
        team_map = deliveries_df.groupby("bowler")["batting_team"].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown"
        ).reset_index()
        team_map.columns = ["player_name", "team_name"]
        # Note: bowler's team is the OPPOSITE of batting_team - we'll fix in dim
        bowler_stats = bowler_stats.merge(team_map, on="player_name", how="left")

        cols = [
            "player_name", "team_name", "matches_played", "innings_bowled",
            "balls_bowled", "runs_conceded", "wickets_taken", "economy_rate",
            "bowling_average", "bowling_strike_rate", "dot_ball_pct",
            "powerplay_economy", "death_economy",
        ]
        bowler_stats = bowler_stats[[c for c in cols if c in bowler_stats.columns]]
        bowler_stats = bowler_stats.fillna(0)

        logger.info(f"  Bowling stats for {len(bowler_stats)} players")
        return bowler_stats

    def build_agg_team_performance(self, matches_df: pd.DataFrame, innings_df: pd.DataFrame) -> pd.DataFrame:
        """Build aggregated team performance stats."""
        logger.info("Building agg_team_performance...")

        match_teams = pd.read_csv(DATASET_DIR / "match_teams.csv")
        male_match_ids = set(matches_df["match_id"].unique())
        match_teams = match_teams[match_teams["match_id"].isin(male_match_ids)]

        team_stats = []
        for team_name, group in match_teams.groupby("team"):
            team_matches = matches_df[matches_df["match_id"].isin(group["match_id"])]
            total = len(team_matches)
            wins = len(team_matches[team_matches["winner"] == team_name])
            losses = total - wins
            win_pct = (wins / total * 100) if total > 0 else 0

            # Toss stats
            toss_wins = len(team_matches[team_matches["toss_winner"] == team_name])
            toss_win_pct = (toss_wins / total * 100) if total > 0 else 0

            # Batting first stats
            team_innings = innings_df[
                (innings_df["match_id"].isin(group["match_id"])) &
                (innings_df["team"] == team_name) &
                (innings_df["innings_number"] == 1)
            ]
            bat_first_matches = team_innings["match_id"].unique()
            bat_first_wins = len(team_matches[
                (team_matches["match_id"].isin(bat_first_matches)) &
                (team_matches["winner"] == team_name)
            ])
            bat_first_win_pct = (bat_first_wins / len(bat_first_matches) * 100) if len(bat_first_matches) > 0 else 0

            # Chase stats
            team_innings_chase = innings_df[
                (innings_df["match_id"].isin(group["match_id"])) &
                (innings_df["team"] == team_name) &
                (innings_df["innings_number"] == 2)
            ]
            chase_matches = team_innings_chase["match_id"].unique()
            chase_wins = len(team_matches[
                (team_matches["match_id"].isin(chase_matches)) &
                (team_matches["winner"] == team_name)
            ])
            chase_win_pct = (chase_wins / len(chase_matches) * 100) if len(chase_matches) > 0 else 0

            # Avg score
            all_team_innings = innings_df[
                (innings_df["match_id"].isin(group["match_id"])) &
                (innings_df["team"] == team_name)
            ]
            avg_score = all_team_innings["total_runs"].mean() if len(all_team_innings) > 0 else 0
            avg_rr = 0
            if len(all_team_innings) > 0:
                valid = all_team_innings[all_team_innings["total_balls"] > 0]
                if len(valid) > 0:
                    avg_rr = (valid["total_runs"] / valid["total_balls"] * 6).mean()

            team_stats.append({
                "team_name": team_name,
                "matches_played": total,
                "matches_won": wins,
                "matches_lost": losses,
                "win_pct": round(win_pct, 2),
                "avg_score_batting": round(avg_score, 2),
                "avg_run_rate": round(avg_rr, 2),
                "toss_win_pct": round(toss_win_pct, 2),
                "bat_first_win_pct": round(bat_first_win_pct, 2),
                "chase_win_pct": round(chase_win_pct, 2),
            })

        team_df = pd.DataFrame(team_stats)
        logger.info(f"  Team performance for {len(team_df)} teams")
        return team_df

    # ===================== LOAD =====================

    def load_to_db(self, df: pd.DataFrame, table_name: str, if_exists: str = "replace"):
        """Load a DataFrame into the database."""
        logger.info(f"Loading {len(df)} rows into {table_name}...")
        df.to_sql(table_name, self.conn, if_exists=if_exists, index=False)
        logger.info(f"  Loaded {table_name} successfully")

    # ===================== DATA QUALITY CHECKS =====================

    def run_quality_checks(self) -> dict:
        """Run data quality validations and return report."""
        logger.info("Running data quality checks...")
        report = {"checks": [], "passed": 0, "failed": 0, "warnings": 0}

        cursor = self.conn.cursor()

        # Check 1: No null match_ids in deliveries
        cursor.execute("SELECT COUNT(*) FROM raw_deliveries WHERE match_id IS NULL")
        null_count = cursor.fetchone()[0]
        check = {
            "name": "No null match_ids in deliveries",
            "status": "PASS" if null_count == 0 else "FAIL",
            "detail": f"{null_count} null match_ids found",
        }
        report["checks"].append(check)
        if check["status"] == "PASS":
            report["passed"] += 1
        else:
            report["failed"] += 1

        # Check 2: runs_batter between 0 and 6
        cursor.execute("SELECT COUNT(*) FROM raw_deliveries WHERE runs_batter < 0 OR runs_batter > 6")
        invalid = cursor.fetchone()[0]
        check = {
            "name": "runs_batter between 0 and 6",
            "status": "PASS" if invalid == 0 else "WARN",
            "detail": f"{invalid} invalid runs found",
        }
        report["checks"].append(check)
        if check["status"] == "PASS":
            report["passed"] += 1
        else:
            report["warnings"] += 1

        # Check 3: over numbers valid (0-19)
        cursor.execute("SELECT COUNT(*) FROM raw_deliveries WHERE over_number < 0 OR over_number > 19")
        invalid = cursor.fetchone()[0]
        check = {
            "name": "over_number between 0 and 19",
            "status": "PASS" if invalid == 0 else "WARN",
            "detail": f"{invalid} invalid overs found",
        }
        report["checks"].append(check)
        if check["status"] == "PASS":
            report["passed"] += 1
        else:
            report["warnings"] += 1

        # Check 4: No duplicate ball events
        cursor.execute("""
            SELECT COUNT(*) FROM (
                SELECT match_id, innings_number, over_number, ball_in_over, COUNT(*) as cnt
                FROM raw_deliveries
                GROUP BY match_id, innings_number, over_number, ball_in_over
                HAVING cnt > 1
            )
        """)
        dupes = cursor.fetchone()[0]
        check = {
            "name": "No duplicate ball events",
            "status": "PASS" if dupes == 0 else "WARN",
            "detail": f"{dupes} duplicate combinations found",
        }
        report["checks"].append(check)
        if check["status"] == "PASS":
            report["passed"] += 1
        else:
            report["warnings"] += 1

        # Check 5: All match_ids in deliveries exist in matches
        cursor.execute("""
            SELECT COUNT(DISTINCT d.match_id) FROM raw_deliveries d
            LEFT JOIN raw_matches m ON d.match_id = m.match_id
            WHERE m.match_id IS NULL
        """)
        orphan = cursor.fetchone()[0]
        check = {
            "name": "All delivery match_ids exist in matches",
            "status": "PASS" if orphan == 0 else "FAIL",
            "detail": f"{orphan} orphan match_ids found",
        }
        report["checks"].append(check)
        if check["status"] == "PASS":
            report["passed"] += 1
        else:
            report["failed"] += 1

        # Check 6: Row counts
        tables = ["raw_matches", "raw_deliveries", "raw_innings", "dim_teams", "dim_players", "dim_venues"]
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                check = {
                    "name": f"{table} row count",
                    "status": "PASS" if count > 0 else "WARN",
                    "detail": f"{count} rows",
                }
                report["checks"].append(check)
                if check["status"] == "PASS":
                    report["passed"] += 1
                else:
                    report["warnings"] += 1
            except Exception as e:
                report["checks"].append({"name": f"{table} exists", "status": "FAIL", "detail": str(e)})
                report["failed"] += 1

        logger.info(f"Quality checks complete: {report['passed']} passed, {report['failed']} failed, {report['warnings']} warnings")
        return report

    # ===================== MAIN PIPELINE =====================

    def run_full_pipeline(self):
        """Execute the complete ETL pipeline."""
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("STARTING FULL ETL PIPELINE")
        logger.info("=" * 60)

        try:
            self.connect()
            self.create_schema()

            # ---- EXTRACT ----
            logger.info("\n--- EXTRACT PHASE ---")
            matches_df = self.extract_matches()
            male_match_ids = set(matches_df["match_id"].unique())

            deliveries_df = self.extract_deliveries(male_match_ids)
            innings_df = self.extract_innings(male_match_ids)
            wickets_df = self.extract_wickets(male_match_ids)
            pom_df = self.extract_player_of_match(male_match_ids)

            # ---- TRANSFORM ----
            logger.info("\n--- TRANSFORM PHASE ---")
            matches_df = self.transform_matches(matches_df)
            deliveries_df = self.transform_deliveries(deliveries_df, wickets_df)
            innings_df = self.transform_innings(innings_df)

            # ---- LOAD BRONZE ----
            logger.info("\n--- LOAD BRONZE LAYER ---")
            self.load_to_db(matches_df, "raw_matches")
            # For deliveries, rename 'over_number' back to match DB
            del_for_db = deliveries_df.copy()
            self.load_to_db(del_for_db, "raw_deliveries")
            self.load_to_db(innings_df, "raw_innings")

            # ---- BUILD DIMENSIONS (GOLD) ----
            logger.info("\n--- BUILD DIMENSION TABLES ---")
            teams_df = self.build_dim_teams(matches_df)
            self.load_to_db(teams_df, "dim_teams")

            players_df = self.build_dim_players(deliveries_df)
            self.load_to_db(players_df, "dim_players")

            venues_df = self.build_dim_venues(matches_df, innings_df)
            self.load_to_db(venues_df, "dim_venues")

            dates_df = self.build_dim_dates(matches_df)
            self.load_to_db(dates_df, "dim_dates")

            # ---- BUILD AGGREGATIONS (GOLD) ----
            logger.info("\n--- BUILD AGGREGATION TABLES ---")
            batting_agg = self.build_agg_player_batting(deliveries_df, matches_df)
            self.load_to_db(batting_agg, "agg_player_batting")

            bowling_agg = self.build_agg_player_bowling(deliveries_df)
            self.load_to_db(bowling_agg, "agg_player_bowling")

            team_perf = self.build_agg_team_performance(matches_df, innings_df)
            self.load_to_db(team_perf, "agg_team_performance")

            # ---- DATA QUALITY ----
            logger.info("\n--- DATA QUALITY CHECKS ---")
            dq_report = self.run_quality_checks()

            # Save DQ report
            import json
            dq_path = PROCESSED_DATA_DIR / "dq_report.json"
            with open(dq_path, "w") as f:
                json.dump(dq_report, f, indent=2)
            logger.info(f"DQ report saved to {dq_path}")

            # ---- SUMMARY ----
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info("\n" + "=" * 60)
            logger.info("ETL PIPELINE COMPLETE")
            logger.info("=" * 60)
            logger.info(f"Elapsed time: {elapsed:.1f} seconds")
            logger.info(f"Stats: {self.stats}")
            logger.info(f"Database: {self.db_path}")

            return True

        except Exception as e:
            logger.error(f"ETL Pipeline failed: {e}", exc_info=True)
            raise
        finally:
            self.close()


if __name__ == "__main__":
    etl = CricketETL()
    etl.run_full_pipeline()
