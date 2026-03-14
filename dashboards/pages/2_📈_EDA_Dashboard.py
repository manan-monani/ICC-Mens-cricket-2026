"""
Page 2: Exploratory Data Analysis Dashboard
Deep statistical analysis with interactive charts.
"""

import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path

st.set_page_config(page_title="EDA | ICC T20", page_icon="📈", layout="wide")

st.markdown("# 📈 Exploratory Data Analysis")
st.markdown("*Deep dive into T20 cricket statistics and trends*")

DB_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "icc_cricket.db"

if not DB_PATH.exists():
    st.error("Database not found. Run ETL pipeline first.")
    st.stop()

conn = sqlite3.connect(str(DB_PATH))

# ===================== SIDEBAR FILTERS =====================
st.sidebar.markdown("### 🎛️ Filters")

teams_df = pd.read_sql("SELECT DISTINCT team_name FROM agg_team_performance ORDER BY team_name", conn)
all_teams = teams_df["team_name"].tolist()
selected_teams = st.sidebar.multiselect("Filter Teams", all_teams, default=all_teams[:10])

# ===================== TEAM PERFORMANCE OVERVIEW =====================
st.markdown("## 🏆 Team Performance Overview")

team_perf = pd.read_sql("SELECT * FROM agg_team_performance ORDER BY win_pct DESC", conn)
if selected_teams:
    team_perf_filtered = team_perf[team_perf["team_name"].isin(selected_teams)]
else:
    team_perf_filtered = team_perf.head(15)

col1, col2 = st.columns(2)

with col1:
    fig = px.bar(
        team_perf_filtered.head(15),
        x="team_name", y="win_pct",
        title="Win Percentage by Team",
        color="win_pct",
        color_continuous_scale="Viridis",
        template="plotly_dark",
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.scatter(
        team_perf_filtered,
        x="bat_first_win_pct", y="chase_win_pct",
        size="matches_played", color="win_pct",
        hover_name="team_name",
        title="Bat First vs Chase Win %",
        color_continuous_scale="RdYlGn",
        template="plotly_dark",
        size_max=40,
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(dash="dash", color="gray"))
    st.plotly_chart(fig, use_container_width=True)

# Team comparison
col1, col2 = st.columns(2)

with col1:
    fig = px.bar(
        team_perf_filtered.head(10),
        x="team_name", y=["avg_score_batting", "avg_run_rate"],
        title="Avg Score & Run Rate by Team",
        barmode="group",
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Toss advantage
    fig = px.bar(
        team_perf_filtered.head(10),
        x="team_name", y="toss_win_pct",
        title="Toss Win Percentage",
        color="toss_win_pct",
        color_continuous_scale="Blues",
        template="plotly_dark",
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ===================== TOP PERFORMERS =====================
st.markdown("## 🌟 Top Performers")

tab1, tab2 = st.tabs(["🏏 Top Batters", "🎯 Top Bowlers"])

with tab1:
    batting = pd.read_sql(
        "SELECT * FROM agg_player_batting ORDER BY total_runs DESC LIMIT 30", conn
    )
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            batting.head(15),
            x="player_name", y="total_runs",
            title="Top 15 Run Scorers",
            color="strike_rate",
            color_continuous_scale="Plasma",
            template="plotly_dark",
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            batting[batting["balls_faced"] > 100],
            x="batting_average", y="strike_rate",
            size="total_runs", color="boundary_pct",
            hover_name="player_name",
            title="Batting Average vs Strike Rate",
            color_continuous_scale="Turbo",
            template="plotly_dark",
            size_max=30,
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Phase analysis
    col1, col2 = st.columns(2)
    with col1:
        pp_data = batting[batting["powerplay_sr"] > 0].head(15)
        fig = px.bar(
            pp_data, x="player_name", y="powerplay_sr",
            title="PowerPlay Strike Rate (Top 15)",
            color="powerplay_sr",
            color_continuous_scale="YlOrRd",
            template="plotly_dark",
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        death_data = batting[batting["death_sr"] > 0].nlargest(15, "death_sr")
        fig = px.bar(
            death_data, x="player_name", y="death_sr",
            title="Death Overs Strike Rate (Top 15)",
            color="death_sr",
            color_continuous_scale="Reds",
            template="plotly_dark",
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(batting, use_container_width=True)

with tab2:
    bowling = pd.read_sql(
        "SELECT * FROM agg_player_bowling ORDER BY wickets_taken DESC LIMIT 30", conn
    )
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            bowling.head(15),
            x="player_name", y="wickets_taken",
            title="Top 15 Wicket Takers",
            color="economy_rate",
            color_continuous_scale="RdYlGn_r",
            template="plotly_dark",
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            bowling[bowling["balls_bowled"] > 100],
            x="economy_rate", y="bowling_strike_rate",
            size="wickets_taken", color="dot_ball_pct",
            hover_name="player_name",
            title="Economy vs Strike Rate",
            color_continuous_scale="Viridis",
            template="plotly_dark",
            size_max=30,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(bowling, use_container_width=True)

st.markdown("---")

# ===================== VENUE ANALYSIS =====================
st.markdown("## 🏟️ Venue Analysis")

venues = pd.read_sql("SELECT * FROM dim_venues WHERE matches_played > 5 ORDER BY matches_played DESC", conn)

col1, col2 = st.columns(2)

with col1:
    fig = px.bar(
        venues.head(15),
        x="venue_name", y=["avg_first_innings_score", "avg_second_innings_score"],
        title="Avg Scores by Venue (1st vs 2nd Innings)",
        barmode="group",
        template="plotly_dark",
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.scatter(
        venues[venues["avg_first_innings_score"].notna()],
        x="avg_first_innings_score", y="avg_second_innings_score",
        size="matches_played",
        hover_name="venue_name",
        title="1st vs 2nd Innings Score by Venue",
        color="matches_played",
        color_continuous_scale="Plasma",
        template="plotly_dark",
        size_max=30,
    )
    fig.add_shape(type="line", x0=100, y0=100, x1=200, y1=200, line=dict(dash="dash", color="gray"))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ===================== MATCH RESULTS ANALYSIS =====================
st.markdown("## 📊 Match Results Analysis")

matches = pd.read_sql("SELECT * FROM raw_matches", conn)

col1, col2 = st.columns(2)

with col1:
    result_dist = matches["result_type"].value_counts().reset_index()
    result_dist.columns = ["Result Type", "Count"]
    fig = px.pie(
        result_dist, values="Count", names="Result Type",
        title="Result Type Distribution",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Win margin distribution
    wins_by_runs = matches[matches["result_type"] == "runs"]["result_margin"]
    wins_by_wickets = matches[matches["result_type"] == "wickets"]["result_margin"]
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Win by Runs (margin)", "Win by Wickets (margin)"])
    fig.add_trace(go.Histogram(x=wins_by_runs, nbinsx=30, marker_color="#3b82f6", name="Runs"), row=1, col=1)
    fig.add_trace(go.Histogram(x=wins_by_wickets, nbinsx=10, marker_color="#10b981", name="Wickets"), row=1, col=2)
    fig.update_layout(template="plotly_dark", title="Win Margin Distribution", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# Matches per season
matches["match_date_parsed"] = pd.to_datetime(matches["match_date"], errors="coerce")
matches["year"] = matches["match_date_parsed"].dt.year
year_counts = matches.groupby("year").size().reset_index(name="matches")

fig = px.area(
    year_counts, x="year", y="matches",
    title="Matches Per Year",
    template="plotly_dark",
    color_discrete_sequence=["#8b5cf6"],
)
fig.update_traces(fill="tonexty")
st.plotly_chart(fig, use_container_width=True)

# Toss decision analysis
toss_analysis = matches.groupby("toss_decision").agg(
    total=("match_id", "count"),
    toss_winner_won=("winner", lambda x: (x == matches.loc[x.index, "toss_winner"]).sum())
).reset_index()
toss_analysis["toss_winner_win_pct"] = (toss_analysis["toss_winner_won"] / toss_analysis["total"] * 100).round(2)

fig = px.bar(
    toss_analysis, x="toss_decision", y=["total", "toss_winner_won"],
    title="Toss Decision Analysis",
    barmode="group",
    template="plotly_dark",
)
st.plotly_chart(fig, use_container_width=True)

conn.close()
