"""
Page 5: Live Match Predictor
Real-time predictions and simulation dashboard.
"""

import streamlit as st
import pandas as pd
import sqlite3
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

st.set_page_config(page_title="Live Predictor | ICC T20", page_icon="🔮", layout="wide")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.markdown("# 🔮 Live Match Predictor")
st.markdown("*Real-time win probability and score predictions*")

DB_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "icc_cricket.db"
MODELS_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "models"

if not DB_PATH.exists():
    st.error("Database not found. Run ETL pipeline first.")
    st.stop()

conn = sqlite3.connect(str(DB_PATH))

# ===================== MATCH INPUT =====================
st.markdown("## ⚙️ Match Configuration")

teams = pd.read_sql("SELECT team_name FROM agg_team_performance ORDER BY win_pct DESC", conn)
team_list = teams["team_name"].tolist()

col1, col2, col3 = st.columns(3)
with col1:
    batting_team = st.selectbox("Batting Team", team_list, index=0)
with col2:
    bowling_team = st.selectbox("Bowling Team", [t for t in team_list if t != batting_team], index=0)
with col3:
    innings = st.radio("Innings", [1, 2], horizontal=True)

st.markdown("---")

# ===================== MATCH STATE INPUT =====================
st.markdown("## 📊 Current Match State")

col1, col2, col3, col4 = st.columns(4)
with col1:
    current_score = st.number_input("Current Score", 0, 350, 80)
with col2:
    current_wickets = st.number_input("Wickets", 0, 10, 2)
with col3:
    current_overs = st.number_input("Overs Completed", 0, 20, 10)
with col4:
    target = None
    if innings == 2:
        target = st.number_input("Target", 1, 350, 180)

# Calculate derived metrics
balls_bowled = current_overs * 6
current_rr = (current_score / balls_bowled * 6) if balls_bowled > 0 else 0
balls_remaining = 120 - balls_bowled

rrr = 0
if target and balls_remaining > 0:
    rrr = ((target - current_score) / balls_remaining) * 6

# Display live metrics
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Score", f"{current_score}/{current_wickets}")
with col2:
    st.metric("Overs", f"{current_overs}")
with col3:
    st.metric("Run Rate", f"{current_rr:.2f}")
with col4:
    if innings == 2 and target:
        st.metric("Required RR", f"{rrr:.2f}")
    else:
        st.metric("Projected", f"{int(current_rr * 20) if current_rr > 0 else '—'}")
with col5:
    st.metric("Balls Remaining", balls_remaining)

st.markdown("---")

# ===================== PREDICTIONS =====================
st.markdown("## 🔮 Predictions")

# Try to load and use the ML model
prediction_made = False

if st.button("🎯 Get Predictions", type="primary"):
    try:
        from ml.train_win_predictor import WinPredictor
        
        predictor = WinPredictor()
        predictor.load_model()
        
        # Get team win rate
        team_stats = pd.read_sql(
            "SELECT win_pct FROM agg_team_performance WHERE team_name = ?",
            conn, params=[batting_team]
        )
        bat_win_rate = float(team_stats.iloc[0]["win_pct"]) if len(team_stats) > 0 else 50.0
        
        pressure = (rrr - current_rr) / max(balls_remaining / 6, 1) if innings == 2 else 0
        
        features = {
            "innings_number_enc": innings - 1,
            "over_number": current_overs,
            "match_phase": 0 if current_overs < 6 else (2 if current_overs >= 16 else 1),
            "current_score": current_score,
            "current_wickets": current_wickets,
            "current_run_rate": round(current_rr, 2),
            "required_run_rate": round(rrr, 2),
            "balls_remaining": balls_remaining,
            "bat_team_win_rate": bat_win_rate,
            "venue_avg_score": 155,
            "toss_winner_batting": 1,
            "target": target or 0,
            "pressure_index": round(pressure, 4),
            "momentum_score": 0,
        }
        
        result = predictor.predict(features)
        prediction_made = True
        
        # Display win probability gauge
        col1, col2 = st.columns(2)
        
        with col1:
            bat_prob = result["batting_team_win_probability"] * 100
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=bat_prob,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": f"{batting_team} Win Probability", "font": {"color": "white"}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#3b82f6"},
                    "steps": [
                        {"range": [0, 30], "color": "#991b1b"},
                        {"range": [30, 50], "color": "#92400e"},
                        {"range": [50, 70], "color": "#166534"},
                        {"range": [70, 100], "color": "#047857"},
                    ],
                },
                number={"suffix": "%", "font": {"color": "white"}},
            ))
            fig.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            bowl_prob = result["bowling_team_win_probability"] * 100
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=bowl_prob,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": f"{bowling_team} Win Probability", "font": {"color": "white"}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#ef4444"},
                    "steps": [
                        {"range": [0, 30], "color": "#991b1b"},
                        {"range": [30, 50], "color": "#92400e"},
                        {"range": [50, 70], "color": "#166534"},
                        {"range": [70, 100], "color": "#047857"},
                    ],
                },
                number={"suffix": "%", "font": {"color": "white"}},
            ))
            fig.update_layout(template="plotly_dark", height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Model details
        with st.expander("📊 Model Details"):
            st.json(result)
            st.markdown(f"**XGBoost Probability:** {result['xgb_probability']:.4f}")
            st.markdown(f"**LightGBM Probability:** {result['lgb_probability']:.4f}")
            st.markdown(f"**Ensemble (Average):** {result['batting_team_win_probability']:.4f}")
            
    except FileNotFoundError:
        st.warning("ML models not trained yet. Run: `python ml/train_win_predictor.py`")
    except Exception as e:
        st.error(f"Prediction error: {e}")

# Score Prediction
if prediction_made or True:
    st.markdown("---")
    st.markdown("## 📈 Score Projection")
    
    try:
        from ml.train_score_regressor import ScoreRegressor
        regressor = ScoreRegressor()
        regressor.load_model()
        
        score_features = {
            "over_number": current_overs,
            "current_score": current_score,
            "current_wickets": current_wickets,
            "current_run_rate": round(current_rr, 2),
            "match_phase": 0 if current_overs < 6 else (2 if current_overs >= 16 else 1),
            "bat_team_win_rate": 50,
            "venue_avg_score": 155,
        }
        
        score_result = regressor.predict(score_features)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Final Score", score_result["predicted_score"])
        with col2:
            st.metric("Score Range", score_result["prediction_range"])
        with col3:
            projected = int(current_rr * 20) if current_rr > 0 else 0
            st.metric("Run Rate Projection", projected)
            
    except Exception:
        # Fallback: simple projection
        projected = int(current_rr * 20) if current_rr > 0 else 0
        st.info(f"**Simple Run Rate Projection:** {projected} runs (based on current RR of {current_rr:.2f})")
        st.caption("Train score regressor for ML-based predictions: `python ml/train_score_regressor.py`")

# ===================== SIMULATION STATUS =====================
st.markdown("---")
st.markdown("## 🎮 Live Simulation")

events_file = Path(__file__).parent.parent.parent / "data" / "processed" / "simulation_events.jsonl"

if events_file.exists():
    events = []
    with open(events_file) as f:
        for line in f.readlines()[-20:]:
            try:
                events.append(json.loads(line.strip()))
            except:
                pass
    
    if events:
        last = events[-1]
        st.success(f"🏏 Live Match: {last.get('batting_team')} vs {last.get('bowling_team')}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Score", f"{last.get('score', 0)}/{last.get('wickets', 0)}")
        with col2:
            st.metric("Overs", f"{last.get('over_number', 0)}.{last.get('ball_in_over', 0)}")
        with col3:
            st.metric("Run Rate", last.get("run_rate", 0))
        with col4:
            if last.get("target"):
                st.metric("Target", last.get("target"))
        
        # Recent events
        st.markdown("### 📜 Recent Ball Events")
        events_df = pd.DataFrame(events[-10:])
        display_cols = [c for c in ["over_number", "ball_in_over", "batter", "bowler", "runs_batter", "is_wicket", "score", "wickets"] if c in events_df.columns]
        st.dataframe(events_df[display_cols], use_container_width=True)
    else:
        st.info("No simulation events yet.")
else:
    st.info("No live simulation running. Start one with: `python simulator/data_simulator.py --fast`")

conn.close()
