# 🏏 ICC Men's T20 World Cup 2026 Prediction Platform

An end-to-end data platform and AI intelligence system built for cricket analytics, match outcome prediction, and tournament analysis. This platform satisfies the requirements of predicting ICC Men's T20 outcomes using a robust data pipeline, machine learning, and GenAI.

## 🌟 Key Features

### 1. Robust Data Pipeline 
- **ETL:** Comprehensive batch ETL processing data from structured raw sources into a Star Schema data warehouse using Python and SQLite/PostgreSQL.
- **Data Simulator:** A highly configurable tool to simulate live T20 ball-by-ball matches for real-time inference and analysis.
- **Medallion Architecture:** Data transitions structurally through Bronze (raw), Silver (cleansed), and Gold (fact/dim/aggregated) layers with strict Data Quality (DQ) validations.

### 2. Machine Learning Suite 🧠
- **Win Probability Predictor (Classification):** An XGBoost + LightGBM ensemble determining live match win probability at a given phase.
- **Score Projections (Regression):** Random Forest + Gradient Boosting Regressor predicting the final innings score based on breakpoints (e.g., at Over 6, 10, 15).
- **Player Archetype Clustering (Unsupervised):** K-Means model clustering players into dynamic roles: *Anchor, Power Hitter, All-Rounder, Specialist Bowler, Impact Player*.

### 3. Generative AI Cricket Chatbot 🤖
- **RAG Architecture:** Leverages Google Gemini and local context from the Data Warehouse to answer complex cricket queries (e.g., stats, head-to-head records, predictions).
- **Graceful Fallback:** Capable of running in an offline data-only querying mode if an API Key is not presented.

### 4. Interactive Dashboards 📊
Streamlit-based UI split into distinct areas:
- **Data Quality Dashboard (Developer):** Tracks pipeline health and runs completeness checks.
- **EDA Dashboard:** Exploratory analysis of historic team, player, and venue performance.
- **Persona KPIs Dashboard:** Distinct lenses tailored to *Team Coach*, *Data Analyst*, and *Tournament Admin*.
- **Live Predictor:** Simulates playing state to query win probability and score regressors in real-time.
- **AI Chatbot UI:** Interactive front-end to talk to the cricket GenAI.

### 5. Production Ready 🚀
- **FastAPI Backend:** Fully decoupled API serving model inference and stats query endpoints.
- **Docker Compose:** Easily runnable multi-container architecture.
- **Airflow DAGs:** Directed Acyclic Graphs planned out for daily ETL schedules and weekly model re-training logic.

---

## 📂 Project Structure

```
├── airflow/                    # Airflow DAGs for orchestration
│   └── dags/
├── api/                        # FastAPI application backend
│   └── main.py
├── dashboards/                 # Streamlit application
│   ├── pages/                  # Multi-page dashboard layouts
│   └── streamlit_app.py
├── data/                       # Local data storage 
│   ├── raw/                    # Raw source dataset (CSVs)
│   └── processed/              # Data Warehouse (icc_cricket.db) & Models
├── docker/                     # Container configurations
│   ├── docker-compose.yml
│   └── Dockerfiles/
├── etl/                        # Batch processing pipeline layer
│   └── batch_etl.py
├── genai/                      # RAG Chatbot architecture
│   └── rag_pipeline.py
├── ml/                         # Feature Engineering & Model Training
│   ├── feature_engineering.py
│   ├── train_win_predictor.py
│   ├── train_score_regressor.py
│   └── train_player_clusters.py
├── simulator/                  # Real-time data simulator 
│   └── data_simulator.py
└── config.py                   # Centralized Configuration
```

---

## ⚙️ Quickstart (Local Development)

### 1. Environment Setup
Install dependencies and configure your `.env` file for API keys.
```bash
python -m venv venv
source venv/Scripts/activate      # Windows
# source venv/bin/activate        # Mac/Linux

pip install -r requirements.txt
cp .env.example .env
```

### 2. Run the Initial Pipeline
Execute the ETL pipeline to build the data warehouse and validate data quality.
```bash
python etl/batch_etl.py
```

### 3. Prepare Machine Learning Models
Generate ML features and train the prediction models.
```bash
python ml/feature_engineering.py

python ml/train_win_predictor.py
python ml/train_score_regressor.py
python ml/train_player_clusters.py
```

### 4. Start the Application
You need to run the **FastAPI Backend** and the **Streamlit Dashboard** simultaneously.

**Terminal 1 (Backend API):**
```bash
python api/main.py
# API available at http://localhost:8000
# Swagger Docs at http://localhost:8000/docs
```

**Terminal 2 (Frontend UI):**
```bash
streamlit run dashboards/streamlit_app.py
# UI available at http://localhost:8501
```

*(Optional)* Run the match simulator to generate live ball-by-ball events for the Live Predictor dashboard to ingest.
```bash
python simulator/data_simulator.py --fast
```

---

## 🐳 Docker Deployment

To spin up the entire pre-configured stack (PostgreSQL, MLflow backend, FastAPI, Streamlit, ChromaDB) via Docker Compose:

```bash
docker-compose -f docker/docker-compose.yml up --build -d
```
- **Streamlit App:** `http://localhost:8501`
- **FastAPI Service:** `http://localhost:8000`
- **MLflow Tracking:** `http://localhost:5000`

---

## 🔗 Architecture Notes

- **Data Simulation:** Events from the script write to `.jsonl` simulating streaming message brokers and ingest directly into the SQLite/Postgres backend.
- **AI/LLM Support:** The `rag_pipeline` depends on `google-generativeai`. Ensure you input your `GOOGLE_API_KEY` in the `.env` file or directly in the UI settings for enhanced AI query augmentation.
