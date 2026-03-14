"""
Page 1: Data Quality & Profiling Dashboard
Developer-focused view of data pipeline health.
"""

import streamlit as st
import pandas as pd
import sqlite3
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Data Quality | ICC T20", page_icon="📊", layout="wide")

st.markdown("# 📊 Data Quality & Profiling Dashboard")
st.markdown("*Developer view: Monitor data pipeline health, quality checks, and table profiling*")

DB_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "icc_cricket.db"
DQ_REPORT_PATH = Path(__file__).parent.parent.parent / "data" / "processed" / "dq_report.json"

if not DB_PATH.exists():
    st.error("Database not found. Run `python etl/batch_etl.py` first.")
    st.stop()

conn = sqlite3.connect(str(DB_PATH))

# ===================== DATA QUALITY CHECKS =====================
st.markdown("## ✅ Quality Check Results")

if DQ_REPORT_PATH.exists():
    with open(DQ_REPORT_PATH) as f:
        dq_report = json.load(f)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Passed ✅", dq_report.get("passed", 0))
    with col2:
        st.metric("Failed ❌", dq_report.get("failed", 0))
    with col3:
        st.metric("Warnings ⚠️", dq_report.get("warnings", 0))

    # Check details
    checks_df = pd.DataFrame(dq_report.get("checks", []))
    if len(checks_df) > 0:
        def color_status(val):
            if val == "PASS":
                return "background-color: #166534; color: white"
            elif val == "FAIL":
                return "background-color: #991b1b; color: white"
            else:
                return "background-color: #92400e; color: white"

        st.dataframe(
            checks_df.style.applymap(color_status, subset=["status"]),
            use_container_width=True,
        )
else:
    st.info("DQ report not generated yet. Run the ETL pipeline.")

st.markdown("---")

# ===================== TABLE PROFILING =====================
st.markdown("## 📋 Table Profiling")

tables = ["raw_matches", "raw_deliveries", "raw_innings", "dim_teams", "dim_players", 
          "dim_venues", "dim_dates", "agg_player_batting", "agg_player_bowling", "agg_team_performance"]

profile_data = []
for table in tables:
    try:
        count = pd.read_sql(f"SELECT COUNT(*) as cnt FROM {table}", conn).iloc[0]["cnt"]
        cols = pd.read_sql(f"PRAGMA table_info({table})", conn)
        profile_data.append({
            "Table": table,
            "Row Count": count,
            "Columns": len(cols),
            "Layer": "Bronze" if "raw_" in table else ("Gold-Dim" if "dim_" in table else ("Gold-Agg" if "agg_" in table else "Gold-Fact")),
        })
    except Exception:
        profile_data.append({"Table": table, "Row Count": 0, "Columns": 0, "Layer": "N/A"})

profile_df = pd.DataFrame(profile_data)
st.dataframe(profile_df, use_container_width=True)

# Row count chart
fig = px.bar(
    profile_df, x="Table", y="Row Count", color="Layer",
    title="Row Counts by Table",
    color_discrete_map={
        "Bronze": "#f59e0b", "Gold-Dim": "#3b82f6", 
        "Gold-Agg": "#10b981", "Gold-Fact": "#8b5cf6", "N/A": "#6b7280"
    },
    template="plotly_dark",
)
fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ===================== NULL ANALYSIS =====================
st.markdown("## 🔍 Null Value Analysis")

selected_table = st.selectbox("Select table to profile:", tables)

try:
    df = pd.read_sql(f"SELECT * FROM {selected_table} LIMIT 10000", conn)
    
    col1, col2 = st.columns(2)
    
    with col1:
        null_counts = df.isnull().sum().reset_index()
        null_counts.columns = ["Column", "Null Count"]
        null_counts["Null %"] = (null_counts["Null Count"] / len(df) * 100).round(2)
        st.dataframe(null_counts, use_container_width=True)
    
    with col2:
        null_fig = px.bar(
            null_counts[null_counts["Null Count"] > 0],
            x="Column", y="Null %",
            title=f"Null Percentage - {selected_table}",
            template="plotly_dark",
            color="Null %",
            color_continuous_scale="RdYlGn_r",
        )
        st.plotly_chart(null_fig, use_container_width=True)

    # Numeric distributions
    st.markdown("### 📊 Numeric Column Distributions")
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    if numeric_cols:
        selected_col = st.selectbox("Select numeric column:", numeric_cols)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                df, x=selected_col, nbins=50,
                title=f"Distribution of {selected_col}",
                template="plotly_dark",
                color_discrete_sequence=["#3b82f6"],
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(
                df, y=selected_col,
                title=f"Box Plot - {selected_col}",
                template="plotly_dark",
                color_discrete_sequence=["#8b5cf6"],
            )
            st.plotly_chart(fig, use_container_width=True)

    # Sample data
    st.markdown("### 📄 Sample Data")
    st.dataframe(df.head(20), use_container_width=True)

except Exception as e:
    st.error(f"Error profiling table: {e}")

conn.close()
