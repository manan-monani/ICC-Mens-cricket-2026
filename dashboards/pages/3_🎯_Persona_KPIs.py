"""
Page 3: Persona KPI Dashboard
Coach, Analyst, and Admin persona views with key KPIs.
"""

import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Persona KPIs | ICC T20", page_icon="🎯", layout="wide")

st.markdown("# 🎯 Persona KPI Dashboard")
st.markdown("*Role-specific analytics and key performance indicators*")

DB_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "icc_cricket.db"

if not DB_PATH.exists():
    st.error("Database not found. Run ETL pipeline first.")
    st.stop()

conn = sqlite3.connect(str(DB_PATH))

# Persona selector
persona = st.selectbox(
    "Select Persona", 
    ["🏏 Team Coach / Captain", "📊 Data Analyst / Scout", "🏛️ Tournament Administrator"],
    index=0
)

st.markdown("---")

# ===================== COACH DASHBOARD =====================
if persona == "🏏 Team Coach / Captain":
    st.markdown("## 🏏 Coach / Captain Dashboard")
    st.markdown("*Optimize playing XI, match strategy, and in-game decisions*")
    
    # Team selector
    teams = pd.read_sql("SELECT team_name FROM agg_team_performance ORDER BY win_pct DESC", conn)
    selected_team = st.selectbox("Select Your Team", teams["team_name"].tolist())
    
    # Team KPIs
    team_stats = pd.read_sql(
        "SELECT * FROM agg_team_performance WHERE team_name = ?", conn, params=[selected_team]
    )
    
    if len(team_stats) > 0:
        ts = team_stats.iloc[0]
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Win Rate", f"{ts['win_pct']:.1f}%")
        with col2:
            st.metric("Avg Score", f"{ts['avg_score_batting']:.0f}")
        with col3:
            st.metric("Avg Run Rate", f"{ts['avg_run_rate']:.2f}")
        with col4:
            st.metric("Bat First Win %", f"{ts['bat_first_win_pct']:.1f}%")
        with col5:
            st.metric("Chase Win %", f"{ts['chase_win_pct']:.1f}%")
    
    st.markdown("---")
    
    # Player Form Index
    st.markdown("### 📊 Player Form Index (Last 5 Matches)")
    team_batters = pd.read_sql("""
        SELECT player_name, team_name, form_index, batting_average, strike_rate, 
               consistency_score, total_runs, matches_played
        FROM agg_player_batting 
        WHERE team_name = ?
        ORDER BY form_index DESC
    """, conn, params=[selected_team])
    
    if len(team_batters) > 0:
        fig = px.bar(
            team_batters.head(11),
            x="player_name", y="form_index",
            title=f"{selected_team} - Player Form Index",
            color="form_index",
            color_continuous_scale="RdYlGn",
            template="plotly_dark",
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Bowling analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Death Overs Economy")
        team_bowlers = pd.read_sql("""
            SELECT player_name, death_economy, economy_rate, wickets_taken, matches_played
            FROM agg_player_bowling 
            WHERE team_name = ? AND death_economy > 0
            ORDER BY death_economy ASC
        """, conn, params=[selected_team])
        
        if len(team_bowlers) > 0:
            fig = px.bar(
                team_bowlers.head(8),
                x="player_name", y="death_economy",
                title="Death Overs Economy (Lower = Better)",
                color="death_economy",
                color_continuous_scale="RdYlGn_r",
                template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🏏 PowerPlay Boundary %")
        pp_data = team_batters[team_batters["matches_played"] >= 3].copy()
        if "boundary_pct" not in pp_data.columns:
            pp_data["boundary_pct"] = 0
        
        fig = px.bar(
            pp_data.head(8),
            x="player_name", y="strike_rate",
            title="Strike Rate (Higher = Better)",
            color="strike_rate",
            color_continuous_scale="YlOrRd",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Head-to-head preview
    st.markdown("### ⚔️ Head-to-Head Records")
    opponents = [t for t in teams["team_name"].tolist() if t != selected_team]
    opponent = st.selectbox("Select Opponent", opponents[:10])
    
    h2h_matches = pd.read_sql("""
        SELECT match_id, match_date, venue, winner, result_margin, result_type
        FROM raw_matches
        WHERE match_id IN (
            SELECT match_id FROM raw_matches WHERE winner = ? OR toss_winner = ?
        )
        AND (winner = ? OR winner = ?)
        ORDER BY match_date DESC
        LIMIT 10
    """, conn, params=[selected_team, selected_team, selected_team, opponent])
    
    if len(h2h_matches) > 0:
        team_wins = len(h2h_matches[h2h_matches["winner"] == selected_team])
        opponent_wins = len(h2h_matches[h2h_matches["winner"] == opponent])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{selected_team} Wins", team_wins)
        with col2:
            st.metric(f"{opponent} Wins", opponent_wins)
        with col3:
            st.metric("Total Matches", len(h2h_matches))
        
        st.dataframe(h2h_matches, use_container_width=True)


# ===================== ANALYST DASHBOARD =====================
elif persona == "📊 Data Analyst / Scout":
    st.markdown("## 📊 Data Analyst / Scout Dashboard")
    st.markdown("*Deep statistical analysis and talent identification*")
    
    # Player Clusters
    st.markdown("### 🔬 Player Clustering Analysis")
    
    try:
        clusters = pd.read_sql("SELECT * FROM ml_player_clusters", conn)
        
        if len(clusters) > 0 and "pca_x" in clusters.columns:
            fig = px.scatter(
                clusters, x="pca_x", y="pca_y",
                color="cluster_label",
                hover_name="player_name",
                title="Player Clusters (PCA Visualization)",
                template="plotly_dark",
                size_max=12,
            )
            fig.update_traces(marker=dict(size=8, opacity=0.7))
            st.plotly_chart(fig, use_container_width=True)
            
            # Cluster distribution
            cluster_dist = clusters["cluster_label"].value_counts().reset_index()
            cluster_dist.columns = ["Cluster", "Count"]
            fig = px.pie(
                cluster_dist, values="Count", names="Cluster",
                title="Player Distribution by Cluster",
                template="plotly_dark",
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("Run player clustering first: `python ml/train_player_clusters.py`")
    
    st.markdown("---")
    
    # Player Comparison Tool
    st.markdown("### 🔄 Player Comparison Tool")
    
    all_players = pd.read_sql("SELECT player_name FROM agg_player_batting ORDER BY total_runs DESC LIMIT 100", conn)
    
    col1, col2 = st.columns(2)
    with col1:
        player1 = st.selectbox("Player 1", all_players["player_name"].tolist(), index=0)
    with col2:
        player2 = st.selectbox("Player 2", all_players["player_name"].tolist(), index=min(1, len(all_players)-1))
    
    if player1 and player2:
        p1_bat = pd.read_sql("SELECT * FROM agg_player_batting WHERE player_name = ?", conn, params=[player1])
        p2_bat = pd.read_sql("SELECT * FROM agg_player_batting WHERE player_name = ?", conn, params=[player2])
        
        if len(p1_bat) > 0 and len(p2_bat) > 0:
            compare_metrics = ["batting_average", "strike_rate", "boundary_pct", "consistency_score", "form_index"]
            available = [m for m in compare_metrics if m in p1_bat.columns]
            
            if available:
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=[float(p1_bat.iloc[0].get(m, 0)) for m in available],
                    theta=available,
                    fill="toself",
                    name=player1,
                    line=dict(color="#3b82f6"),
                ))
                fig.add_trace(go.Scatterpolar(
                    r=[float(p2_bat.iloc[0].get(m, 0)) for m in available],
                    theta=available,
                    fill="toself",
                    name=player2,
                    line=dict(color="#ef4444"),
                ))
                fig.update_layout(
                    polar=dict(bgcolor="rgba(0,0,0,0)"),
                    template="plotly_dark",
                    title=f"{player1} vs {player2} - Radar Comparison",
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Consistency Analysis
    st.markdown("### 📈 Consistency Score Distribution")
    consistency_data = pd.read_sql("""
        SELECT player_name, team_name, consistency_score, batting_average, strike_rate, total_runs
        FROM agg_player_batting 
        WHERE matches_played >= 10
        ORDER BY consistency_score DESC
    """, conn)
    
    if len(consistency_data) > 0:
        fig = px.scatter(
            consistency_data.head(50),
            x="batting_average", y="consistency_score",
            size="total_runs", color="strike_rate",
            hover_name="player_name",
            title="Consistency vs Batting Average (Bubble = Total Runs)",
            color_continuous_scale="Turbo",
            template="plotly_dark",
            size_max=25,
        )
        st.plotly_chart(fig, use_container_width=True)


# ===================== ADMIN DASHBOARD =====================
elif persona == "🏛️ Tournament Administrator":
    st.markdown("## 🏛️ Tournament Administrator Dashboard")
    st.markdown("*Scheduling, venue allocation, and tournament insights*")
    
    # Venue statistics
    st.markdown("### 🏟️ Venue Analysis")
    venues = pd.read_sql("""
        SELECT venue_name, city, matches_played, 
               avg_first_innings_score, avg_second_innings_score
        FROM dim_venues 
        WHERE matches_played >= 5
        ORDER BY matches_played DESC
    """, conn)
    
    if len(venues) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                venues.head(15),
                x="venue_name", y="matches_played",
                title="Matches Played by Venue",
                color="matches_played",
                color_continuous_scale="Blues",
                template="plotly_dark",
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                venues,
                x="avg_first_innings_score", y="avg_second_innings_score",
                size="matches_played",
                hover_name="venue_name",
                title="Avg Scores: 1st vs 2nd Innings",
                template="plotly_dark",
                color_continuous_scale="Plasma",
                color="matches_played",
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Toss advantage by venue
    st.markdown("### 🪙 Toss Impact Analysis")
    
    matches = pd.read_sql("SELECT * FROM raw_matches", conn)
    
    toss_impact = matches.groupby("toss_decision").agg(
        total=("match_id", "count"),
    ).reset_index()
    
    # Win after winning toss
    matches["toss_won_and_match_won"] = matches["toss_winner"] == matches["winner"]
    toss_win_rate = matches.groupby("toss_decision")["toss_won_and_match_won"].mean().reset_index()
    toss_win_rate.columns = ["Toss Decision", "Win Rate After Toss Win"]
    toss_win_rate["Win Rate After Toss Win"] = (toss_win_rate["Win Rate After Toss Win"] * 100).round(2)
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            toss_win_rate, x="Toss Decision", y="Win Rate After Toss Win",
            title="Win Rate After Winning Toss",
            color="Win Rate After Toss Win",
            color_continuous_scale="RdYlGn",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Match outcome predictability
        result_margins = matches.dropna(subset=["result_margin"])
        fig = px.histogram(
            result_margins, x="result_margin",
            color="result_type",
            title="Result Margin Distribution",
            template="plotly_dark",
            barmode="overlay",
            opacity=0.7,
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Matches per event
    st.markdown("### 📅 Tournament Statistics")
    events = matches["event_name"].value_counts().reset_index()
    events.columns = ["Event", "Matches"]
    
    fig = px.treemap(
        events.head(20),
        path=["Event"], values="Matches",
        title="Matches by Tournament",
        template="plotly_dark",
        color="Matches",
        color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig, use_container_width=True)

conn.close()
