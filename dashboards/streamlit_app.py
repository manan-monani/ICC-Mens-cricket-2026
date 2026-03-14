"""
ICC T20 Predictor - Main Streamlit Dashboard
Unified multi-page dashboard application.
"""

import streamlit as st

st.set_page_config(
    page_title="ICC T20 Cricket Predictor 🏏",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium look
st.markdown("""
<style>
    /* Main theme */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Gradient header */
    .stApp > header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Custom metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border: 1px solid #475569;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
    }
    
    div[data-testid="metric-container"] label {
        color: #94a3b8 !important;
    }
    
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
        font-size: 1.8rem !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #cbd5e1;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
    }
    
    /* Plotly chart containers */
    .stPlotlyChart {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Headers */
    h1 {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }
    
    h2, h3 {
        color: #e2e8f0 !important;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)


# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown("## 🏏 ICC T20 Predictor")
    st.markdown("---")
    st.markdown("""
    **Navigate to:**
    - 📊 **Data Quality** - DQ checks & profiling
    - 📈 **EDA Dashboard** - Exploratory analysis
    - 🎯 **Persona KPIs** - Coach & Analyst views  
    - 🤖 **AI Chatbot** - Ask cricket questions
    - 🔮 **Live Predictor** - Real-time predictions
    """)
    st.markdown("---")
    st.markdown("*Built for ICC T20 WC 2026 Hackathon*")
    st.markdown(f"*v1.0.0*")


# ===================== HOME PAGE =====================
st.markdown("# 🏏 ICC Men's T20 World Cup 2026 Prediction Platform")
st.markdown("### Real-time Cricket Analytics & AI-Powered Predictions")

st.markdown("---")

# Quick stats
import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "processed" / "icc_cricket.db"

if DB_PATH.exists():
    conn = sqlite3.connect(str(DB_PATH))
    
    try:
        matches = pd.read_sql("SELECT COUNT(*) as cnt FROM raw_matches", conn).iloc[0]["cnt"]
        deliveries = pd.read_sql("SELECT COUNT(*) as cnt FROM raw_deliveries", conn).iloc[0]["cnt"]
        players = pd.read_sql("SELECT COUNT(*) as cnt FROM agg_player_batting", conn).iloc[0]["cnt"]
        teams = pd.read_sql("SELECT COUNT(*) as cnt FROM agg_team_performance", conn).iloc[0]["cnt"]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Matches", f"{matches:,}")
        with col2:
            st.metric("Ball Events", f"{deliveries:,}")
        with col3:
            st.metric("Players Tracked", f"{players:,}")
        with col4:
            st.metric("Teams", f"{teams:,}")
            
    except Exception as e:
        st.warning(f"Database tables not yet populated. Run the ETL pipeline first.")
    
    conn.close()
else:
    st.warning("⚠️ Database not found. Please run the ETL pipeline first:")
    st.code("python etl/batch_etl.py", language="bash")

st.markdown("---")

# Feature cards
st.markdown("## 🌟 Platform Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📊 Data Analytics
    - Star schema data warehouse
    - Real-time data ingestion
    - Automated quality checks
    - 3200+ men's T20 matches analyzed
    """)

with col2:
    st.markdown("""
    ### 🤖 AI & ML Models
    - Win probability prediction (XGBoost + LightGBM)
    - Score regression (Random Forest + GBR)
    - Player clustering (K-Means + PCA)
    - RAG-powered AI chatbot
    """)

with col3:
    st.markdown("""
    ### 📈 Interactive Dashboards
    - Data quality monitoring
    - Exploratory data analysis
    - Coach & Analyst persona views
    - Live match prediction
    """)

st.markdown("---")
st.markdown("### 🚀 Quick Start")
st.markdown("""
1. **Run ETL Pipeline**: `python etl/batch_etl.py`
2. **Train ML Models**: `python ml/train_win_predictor.py`
3. **Start API Server**: `python api/main.py`
4. **Launch Dashboard**: `streamlit run dashboards/streamlit_app.py`
""")
