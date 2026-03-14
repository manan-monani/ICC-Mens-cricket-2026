"""
ICC T20 Predictor - FastAPI Backend
Serves all APIs for predictions, chatbot, and dashboard data.
"""

import os
import sys
import json
import sqlite3
import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DATA_DIR, APP_PORT, GOOGLE_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

DB_PATH = PROCESSED_DATA_DIR / "icc_cricket.db"

# ===================== FASTAPI APP =====================

app = FastAPI(
    title="ICC T20 Cricket Predictor API",
    description="Real-time predictions, statistics, and AI-powered cricket analytics",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================== MODELS (Request/Response) =====================

class PredictionRequest(BaseModel):
    innings_number: int = 1
    over_number: int = 10
    current_score: int = 80
    current_wickets: int = 2
    batting_team: str = "India"
    bowling_team: str = "Australia"
    venue: Optional[str] = None
    target: Optional[int] = None

class ChatRequest(BaseModel):
    query: str
    api_key: Optional[str] = None

class ChatResponse(BaseModel):
    query: str
    answer: str
    source: str


# ===================== HELPER FUNCTIONS =====================

def get_db():
    return sqlite3.connect(str(DB_PATH))


# ===================== LAZY-LOADED ML MODELS =====================

_win_predictor = None
_score_regressor = None


def get_win_predictor():
    global _win_predictor
    if _win_predictor is None:
        try:
            from ml.train_win_predictor import WinPredictor
            _win_predictor = WinPredictor()
            _win_predictor.load_model()
        except Exception as e:
            logger.warning(f"Win predictor not available: {e}")
    return _win_predictor


def get_score_regressor():
    global _score_regressor
    if _score_regressor is None:
        try:
            from ml.train_score_regressor import ScoreRegressor
            _score_regressor = ScoreRegressor()
            _score_regressor.load_model()
        except Exception as e:
            logger.warning(f"Score regressor not available: {e}")
    return _score_regressor


# ===================== ENDPOINTS =====================

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "db_exists": DB_PATH.exists(),
    }


@app.post("/predict/win_probability")
def predict_win_probability(request: PredictionRequest):
    """Predict win probability for current match state."""
    predictor = get_win_predictor()
    if predictor is None:
        raise HTTPException(status_code=503, detail="Win predictor model not loaded")

    # Get team win rates
    conn = get_db()
    team_stats = pd.read_sql("SELECT team_name, win_pct FROM agg_team_performance", conn)
    conn.close()

    bat_win_rate = team_stats[team_stats["team_name"] == request.batting_team]["win_pct"].values
    bat_win_rate = float(bat_win_rate[0]) if len(bat_win_rate) > 0 else 50.0

    balls_bowled = request.over_number * 6
    current_rr = (request.current_score / balls_bowled * 6) if balls_bowled > 0 else 0
    balls_remaining = 120 - balls_bowled

    rrr = 0
    if request.target and balls_remaining > 0:
        rrr = ((request.target - request.current_score) / balls_remaining) * 6

    pressure = (rrr - current_rr) / max(balls_remaining / 6, 1) if request.innings_number == 2 else 0

    features = {
        "innings_number_enc": request.innings_number - 1,
        "over_number": request.over_number,
        "match_phase": 0 if request.over_number < 6 else (2 if request.over_number >= 16 else 1),
        "current_score": request.current_score,
        "current_wickets": request.current_wickets,
        "current_run_rate": round(current_rr, 2),
        "required_run_rate": round(rrr, 2),
        "balls_remaining": balls_remaining,
        "bat_team_win_rate": bat_win_rate,
        "venue_avg_score": 155,
        "toss_winner_batting": 1,
        "target": request.target or 0,
        "pressure_index": round(pressure, 4),
        "momentum_score": 0,
    }

    prediction = predictor.predict(features)
    return {
        "batting_team": request.batting_team,
        "bowling_team": request.bowling_team,
        **prediction,
        "match_state": {
            "score": request.current_score,
            "wickets": request.current_wickets,
            "overs": request.over_number,
            "run_rate": round(current_rr, 2),
        },
    }


@app.post("/predict/final_score")
def predict_final_score(request: PredictionRequest):
    """Predict final innings score."""
    regressor = get_score_regressor()
    if regressor is None:
        raise HTTPException(status_code=503, detail="Score regressor model not loaded")

    balls_bowled = request.over_number * 6
    current_rr = (request.current_score / balls_bowled * 6) if balls_bowled > 0 else 0

    features = {
        "over_number": request.over_number,
        "current_score": request.current_score,
        "current_wickets": request.current_wickets,
        "current_run_rate": round(current_rr, 2),
        "match_phase": 0 if request.over_number < 6 else (2 if request.over_number >= 16 else 1),
        "bat_team_win_rate": 50,
        "venue_avg_score": 155,
    }

    prediction = regressor.predict(features)
    return {
        "batting_team": request.batting_team,
        **prediction,
    }


@app.get("/players/{player_name}/stats")
def get_player_stats(player_name: str):
    """Get comprehensive player statistics."""
    conn = get_db()
    batting = pd.read_sql(
        "SELECT * FROM agg_player_batting WHERE player_name LIKE ?",
        conn, params=[f"%{player_name}%"]
    )
    bowling = pd.read_sql(
        "SELECT * FROM agg_player_bowling WHERE player_name LIKE ?",
        conn, params=[f"%{player_name}%"]
    )
    conn.close()

    if len(batting) == 0 and len(bowling) == 0:
        raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found")

    result = {"player_name": player_name}
    if len(batting) > 0:
        result["batting"] = batting.iloc[0].to_dict()
    if len(bowling) > 0:
        result["bowling"] = bowling.iloc[0].to_dict()

    return result


@app.get("/teams/{team_name}/stats")
def get_team_stats(team_name: str):
    """Get team performance statistics."""
    conn = get_db()
    df = pd.read_sql(
        "SELECT * FROM agg_team_performance WHERE team_name LIKE ?",
        conn, params=[f"%{team_name}%"]
    )
    conn.close()

    if len(df) == 0:
        raise HTTPException(status_code=404, detail=f"Team '{team_name}' not found")

    return df.iloc[0].to_dict()


@app.get("/dashboard/eda/top_batters")
def get_top_batters(limit: int = Query(default=20, le=100)):
    """Get top batters for EDA dashboard."""
    conn = get_db()
    df = pd.read_sql(
        f"SELECT * FROM agg_player_batting ORDER BY total_runs DESC LIMIT {limit}",
        conn
    )
    conn.close()
    return df.to_dict("records")


@app.get("/dashboard/eda/top_bowlers")
def get_top_bowlers(limit: int = Query(default=20, le=100)):
    """Get top bowlers for EDA dashboard."""
    conn = get_db()
    df = pd.read_sql(
        f"SELECT * FROM agg_player_bowling ORDER BY wickets_taken DESC LIMIT {limit}",
        conn
    )
    conn.close()
    return df.to_dict("records")


@app.get("/dashboard/eda/team_performance")
def get_team_performance():
    """Get all team performance stats."""
    conn = get_db()
    df = pd.read_sql("SELECT * FROM agg_team_performance ORDER BY win_pct DESC", conn)
    conn.close()
    return df.to_dict("records")


@app.get("/dashboard/eda/venue_stats")
def get_venue_stats():
    """Get venue statistics."""
    conn = get_db()
    df = pd.read_sql("SELECT * FROM dim_venues ORDER BY matches_played DESC", conn)
    conn.close()
    return df.to_dict("records")


@app.get("/dashboard/dq/report")
def get_dq_report():
    """Get latest data quality report."""
    report_path = PROCESSED_DATA_DIR / "dq_report.json"
    if report_path.exists():
        with open(report_path) as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="DQ report not found. Run ETL pipeline first.")


@app.get("/dashboard/clusters")
def get_player_clusters():
    """Get player cluster assignments."""
    conn = get_db()
    try:
        df = pd.read_sql("SELECT * FROM ml_player_clusters", conn)
        conn.close()
        return df.to_dict("records")
    except Exception:
        conn.close()
        raise HTTPException(status_code=404, detail="Run player clustering first")


@app.post("/chat/query")
def chat_query(request: ChatRequest):
    """RAG chatbot query endpoint."""
    from genai.rag_pipeline import CricketChatbot

    api_key = request.api_key or GOOGLE_API_KEY
    chatbot = CricketChatbot(api_key=api_key)
    result = chatbot.query(request.query)

    return ChatResponse(
        query=result["query"],
        answer=result["answer"],
        source=result["source"],
    )


@app.get("/match/live")
def get_live_match_state():
    """Get current live/simulated match state."""
    events_file = PROCESSED_DATA_DIR / "simulation_events.jsonl"
    if not events_file.exists():
        return {"status": "no_live_match", "events": []}

    # Read last N events
    events = []
    with open(events_file) as f:
        lines = f.readlines()
        for line in lines[-20:]:  # Last 20 events
            try:
                events.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    if events:
        last = events[-1]
        return {
            "status": "live" if not last.get("match_complete") else "completed",
            "match_state": {
                "match_id": last.get("match_id"),
                "batting_team": last.get("batting_team"),
                "bowling_team": last.get("bowling_team"),
                "score": last.get("score"),
                "wickets": last.get("wickets"),
                "overs": f"{last.get('over_number', 0)}.{last.get('ball_in_over', 0)}",
                "run_rate": last.get("run_rate"),
                "target": last.get("target"),
            },
            "recent_events": events[-5:],
        }

    return {"status": "no_events", "events": []}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=APP_PORT)
