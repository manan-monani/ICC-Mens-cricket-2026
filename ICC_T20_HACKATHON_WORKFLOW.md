# ICC Men's T20 World Cup 2026 — Outcome Prediction Platform
## Complete Agent Workflow for Antigravity IDE (Claude Opus 4.6)

---

## HOW TO USE THIS FILE
This document is the **single source of truth** for the agent. Each section is labeled:
- `[ONCE]` — Execute exactly one time during project setup
- `[RECURRING]` — Executes repeatedly on a schedule or trigger
- `[ON-DEMAND]` — Executes when triggered by new data, model retraining, or user request

Work top-to-bottom. Never skip a layer. Each layer's output is the next layer's input.

---

## TECH STACK — FULL REFERENCE

| Layer | Tool/Technology | Purpose | Run Frequency |
|---|---|---|---|
| Data Simulation | Python (`faker`, `random`, `schedule`) | Generate synthetic ball-by-ball events | RECURRING (every 5s) |
| Message Broker | Apache Kafka + Zookeeper | Real-time event streaming | RECURRING (always on) |
| Stream Processing | Apache Spark Structured Streaming | Real-time ETL, aggregations | RECURRING (always on) |
| Batch ETL | Apache Spark (PySpark) + dbt | Full historical load, transformations | ONCE + ON-DEMAND |
| Data Warehouse | PostgreSQL (local) or Snowflake (cloud) | Store fact/dimension tables | ONCE (schema) + RECURRING (inserts) |
| Data Quality | Great Expectations | Profile, validate, assert rules | RECURRING (per batch) |
| Orchestration | Apache Airflow | Schedule batch ETL, model retraining | RECURRING |
| ML Training | Scikit-learn, XGBoost, LightGBM | Win prediction, player clustering | ONCE + ON-DEMAND (retrain) |
| Model Registry | MLflow | Track experiments, version models | ON-DEMAND |
| API Backend | FastAPI | Serve predictions, chatbot, dashboard APIs | RECURRING (always on) |
| Dashboards | Metabase + Grafana + Streamlit | EDA, persona KPI, unified view | ONCE (setup) + RECURRING (refresh) |
| GenAI / RAG | LangChain + OpenAI/Claude API | Chatbot, NL queries, doc extraction | ON-DEMAND |
| Vector Store | ChromaDB | Store embeddings for RAG | ONCE (index) + ON-DEMAND (update) |
| Frontend | Streamlit (or React) | Unified interface | ONCE (build) |
| Containerization | Docker + Docker Compose | Package all services | ONCE (build) + RECURRING (run) |
| Secrets/Config | `.env` file + `python-dotenv` | Manage API keys, DB credentials | ONCE |

---

## STEP 1 — PROJECT SCAFFOLD [ONCE]

### 1.1 Directory Structure
Create the following folder tree:

```
icc_t20_predictor/
├── data/
│   ├── raw/                    # Original Kaggle CSV files (already have these)
│   ├── processed/              # Cleaned outputs
│   └── embeddings/             # ChromaDB vector store
├── simulator/
│   ├── data_simulator.py       # Real-time event generator
│   └── kafka_producer.py       # Publishes events to Kafka topics
├── etl/
│   ├── spark_streaming.py      # Reads from Kafka, writes to Bronze
│   ├── spark_batch.py          # Historical full load
│   ├── dbt/                    # dbt project for Silver→Gold transforms
│   │   ├── models/
│   │   │   ├── silver/         # Cleaned models
│   │   │   └── gold/           # Aggregated fact/dim models
│   │   └── dbt_project.yml
│   └── great_expectations/     # Data quality suite
├── warehouse/
│   ├── schema.sql              # DDL for all tables
│   └── seed_data.sql           # Static dimension tables (teams, venues)
├── ml/
│   ├── feature_engineering.py
│   ├── train_win_predictor.py
│   ├── train_player_clusters.py
│   ├── train_score_regressor.py
│   └── model_registry.py       # MLflow logging helpers
├── api/
│   ├── main.py                 # FastAPI app
│   ├── routers/
│   │   ├── predictions.py      # ML prediction endpoints
│   │   ├── chat.py             # GenAI chatbot endpoints
│   │   └── dashboard.py        # Dashboard data endpoints
│   └── dependencies.py
├── genai/
│   ├── rag_pipeline.py         # LangChain RAG setup
│   ├── document_loader.py      # PDF/web scraping
│   ├── vector_store.py         # ChromaDB operations
│   └── nl_query.py             # Natural language → SQL
├── dashboards/
│   ├── streamlit_app.py        # Unified Streamlit dashboard
│   ├── pages/
│   │   ├── 1_data_quality.py
│   │   ├── 2_eda.py
│   │   ├── 3_coach_dashboard.py
│   │   ├── 4_analyst_dashboard.py
│   │   ├── 5_predictor.py
│   │   └── 6_chatbot.py
│   └── metabase/               # Metabase dashboard JSON exports
├── airflow/
│   └── dags/
│       ├── batch_etl_dag.py
│       └── model_retrain_dag.py
├── docker/
│   ├── docker-compose.yml
│   └── Dockerfiles/
│       ├── Dockerfile.api
│       ├── Dockerfile.etl
│       └── Dockerfile.dashboard
├── .env.example
├── requirements.txt
└── README.md
```

### 1.2 Install Dependencies [ONCE]
```bash
pip install pyspark kafka-python dbt-postgres great_expectations \
            scikit-learn xgboost lightgbm mlflow fastapi uvicorn \
            langchain langchain-openai chromadb streamlit pandas \
            sqlalchemy psycopg2-binary python-dotenv schedule faker \
            plotly altair
```

### 1.3 Environment File [ONCE]
Create `.env`:
```
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=icc_cricket
POSTGRES_USER=admin
POSTGRES_PASSWORD=secret
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
OPENAI_API_KEY=sk-...         # or ANTHROPIC_API_KEY
MLFLOW_TRACKING_URI=http://localhost:5000
```

---

## STEP 2 — DATA WAREHOUSE DESIGN [ONCE]

### 2.1 Schema Design (Star Schema)

**Fact Tables:**
- `fact_ball_events` — one row per ball (match_id, inning, over, ball, batsman_id, bowler_id, runs, wicket, extras, venue_id, date_id)
- `fact_match_results` — one row per match (match_id, team1_id, team2_id, winner_id, margin, venue_id, date_id, toss_winner_id, toss_decision)

**Dimension Tables:**
- `dim_players` — player_id, name, country, role (batsman/bowler/allrounder), batting_style, bowling_style
- `dim_teams` — team_id, name, country, captain_id
- `dim_venues` — venue_id, name, city, country, pitch_type, avg_first_innings_score
- `dim_dates` — date_id, date, year, month, day, tournament_phase (group/SF/final)

**Agent instruction:** Run `warehouse/schema.sql` against PostgreSQL to create all tables. Use foreign keys and indexes on match_id, player_id.

### 2.2 Medallion Architecture Mapping
- **Bronze** = raw Kafka events landed as-is into PostgreSQL `raw_*` tables
- **Silver** = dbt models in `etl/dbt/models/silver/` — deduplication, null handling, type casting
- **Gold** = dbt models in `etl/dbt/models/gold/` — aggregated fact/dim tables used by ML and dashboards

---

## STEP 3 — DATA SIMULATION [RECURRING — every 5 seconds]

### 3.1 Simulator Logic
File: `simulator/data_simulator.py`

The simulator should:
1. Load the static match schedule from `data/raw/`
2. Pick the "current" in-progress match
3. Every 5 seconds, generate one ball event:
   - Random batsman from playing XI
   - Random bowler eligible for that over
   - Outcome: dot(40%), single(25%), 2-3 runs(15%), 4(10%), 6(5%), wicket(5%)
   - Include extras (wides, no-balls) with 5% probability
4. Maintain match state: score, wickets, run rate, required rate
5. Publish the event as a JSON message to Kafka topic `ball_events`

### 3.2 Kafka Topics to Create [ONCE]
```
ball_events        — live ball-by-ball
match_state        — current score/wickets after each ball
player_updates     — player stats after each innings
venue_conditions   — pitch/weather updates (every 30 mins)
```

### 3.3 Kafka Producer [RECURRING]
File: `simulator/kafka_producer.py`
- Connect to Kafka at `KAFKA_BOOTSTRAP_SERVERS`
- Serialize events as JSON
- Produce to appropriate topic based on event type
- Log production errors and retry with backoff

---

## STEP 4 — ETL PIPELINE

### 4.1 Spark Streaming (Bronze Layer) [RECURRING — always running]
File: `etl/spark_streaming.py`

```python
# Pseudocode structure:
spark = SparkSession.builder.appName("CricketStreaming").getOrCreate()
df = spark.readStream.format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
    .option("subscribe", "ball_events,match_state") \
    .load()
# Parse JSON → apply schema → write to PostgreSQL raw tables
df.writeStream.foreachBatch(write_to_bronze).start().awaitTermination()
```

Key transformations in streaming:
- Parse JSON payload
- Add `ingestion_timestamp`
- Validate required fields (match_id, over, ball — must not be null)
- Write to `raw_ball_events` with status = 'pending'

### 4.2 Spark Batch (Historical Load) [ONCE + ON-DEMAND on new file]
File: `etl/spark_batch.py`

Steps:
1. Read all CSV files from `data/raw/`
2. Apply the same schema as streaming
3. Deduplicate on (match_id, inning, over, ball)
4. Write to Bronze tables
5. Trigger dbt run after completion

### 4.3 dbt Silver Models [RECURRING — runs after each batch]
Each dbt model in `etl/dbt/models/silver/`:
- `stg_ball_events.sql` — cast types, handle nulls, rename columns
- `stg_match_results.sql` — normalize team names, fill missing toss data
- `stg_players.sql` — deduplicate player records, standardize roles

### 4.4 dbt Gold Models [RECURRING]
- `fact_ball_events.sql` — join with dim tables, add computed columns (run_rate, phase_of_game)
- `fact_match_results.sql` — add win margin category, venue stats
- `agg_player_batting.sql` — batting average, strike rate, boundary %, consistency score
- `agg_player_bowling.sql` — economy rate, wickets per match, dot ball %, death over economy
- `agg_team_performance.sql` — win rate by venue, by phase, head-to-head records

### 4.5 Data Quality with Great Expectations [RECURRING — per batch]
File: `etl/great_expectations/`

Expectations to implement:
- `match_id` is never null
- `runs_off_bat` is between 0 and 6
- `over` is between 0 and 19
- Player IDs exist in `dim_players`
- No duplicate (match_id, inning, over, ball) combinations
- Win/loss column has only valid team IDs

Output: HTML data quality report stored in `data/processed/dq_reports/`

---

## STEP 5 — PERSONAS AND KPIs

### Persona 1: Team Coach / Captain
**Goal:** Optimize playing XI selection and in-match strategy

KPIs:
- Win probability (live, updated per ball)
- Player form index (last 5 matches weighted)
- Bowling economy in death overs (overs 16-20)
- Boundary % in powerplay (overs 1-6)
- Head-to-head record vs today's opponent
- Best batting position for each player
- Optimal bowling rotation recommendation

### Persona 2: Data Analyst / Scout
**Goal:** Deep statistical analysis and talent identification

KPIs:
- Player consistency score (mean + std dev of runs)
- All-rounder impact index (batting + bowling combined)
- Venue-specific batting average
- Cluster assignment (player type: aggressive, accumulator, anchor, big-hitter)
- Partnership run rates by batting position combo
- Bowling match-up stats (left-hand vs right-hand batsmen)

### Persona 3: Tournament Administrator
**Goal:** Fair scheduling, venue allocation, forecasting

KPIs:
- Average scores by venue and pitch type
- Toss advantage win rate by venue
- Player availability and fatigue index
- Match outcome predictability score

---

## STEP 6 — DASHBOARDS

### 6.1 Data Quality Dashboard [Page 1 — Streamlit]
Built from Great Expectations output + DW profiling:
- Table: row counts per table, null %, duplicate %
- Chart: data freshness timeline (last ingestion per topic)
- Alert panel: failed expectations with severity
- Profiling: distribution plots for key numeric columns

### 6.2 EDA Dashboard [Page 2 — Streamlit]
- Score distribution by innings position
- Run rate progression chart (over-by-over)
- Correlation heatmap (venue type vs avg score)
- Top 10 batsmen and bowlers (filterable by tournament phase)
- Win percentage by batting first vs chasing

### 6.3 Coach Dashboard [Page 3 — Streamlit]
- Live win probability gauge (updates every 5s from simulator)
- Required run rate vs current run rate
- Player fitness/form heatmap
- Recommended batting order
- Bowling plan optimizer

### 6.4 Analyst Dashboard [Page 4 — Streamlit]
- Player cluster scatter plot (Plotly, interactive)
- Deep drill-down: click any player → full stat card
- Head-to-head comparison tool (pick 2 players)
- Partnership matrix
- Predictive batting position optimizer

### 6.5 Unified Dashboard Integration
All pages share:
- Common sidebar: tournament selector, team filter, date range
- Shared PostgreSQL connection via `@st.cache_resource`
- Real-time refresh: `st.experimental_rerun()` every 10s on live pages
- Export button: download any chart as PNG or CSV

**Metabase** (optional BI layer):
- Connect Metabase to PostgreSQL
- Build SQL-based questions for each KPI
- Embed Metabase iframes into Streamlit using `st.components.v1.iframe()`

**Grafana** (real-time monitoring):
- Connect to PostgreSQL or InfluxDB
- Live panels: events/second, Kafka lag, model prediction latency

---

## STEP 7 — MACHINE LEARNING

### 7.1 Feature Engineering [ONCE + ON-DEMAND]
File: `ml/feature_engineering.py`

Features to generate:
```
Match context:     match_phase, venue_type, toss_winner_factor, day_night
Team features:     team_win_rate_last5, head_to_head_wins, avg_score_at_venue
Batting state:     current_score, wickets_fallen, run_rate, balls_remaining, required_rate
Player features:   batsman_avg, batsman_sr_last5, batsman_vs_bowler_type_sr
                   bowler_economy_last5, bowler_wickets_last5
Derived:           pressure_index = (required_rate - run_rate) / balls_remaining
                   momentum_score = runs_last_3_overs / (wickets_last_3_overs + 1)
```

### 7.2 Use Case 1: Win Probability Prediction (Classification) [ONCE + RETRAIN]
File: `ml/train_win_predictor.py`
- Algorithm: XGBoost Classifier + LightGBM (ensemble)
- Target: `match_winner` (binary: team1=1, team2=0)
- Input: features at each over break
- Train/test split: 80/20, stratified by tournament phase
- Metrics: AUC-ROC, F1 score, calibration curve
- Live inference: called every ball by FastAPI `/predict/win_probability`

### 7.3 Use Case 2: Final Score Regression [ONCE + RETRAIN]
File: `ml/train_score_regressor.py`
- Algorithm: Random Forest Regressor + Ridge Regression
- Target: `final_innings_score`
- Input: score at over 6, over 10, over 15
- Metrics: RMSE, MAE, R²

### 7.4 Use Case 3: Player Clustering (Unsupervised) [ONCE + ON-DEMAND]
File: `ml/train_player_clusters.py`
- Algorithm: K-Means (k=5) + PCA for visualization
- Features: batting_avg, strike_rate, boundary_pct, consistency, bowling_economy
- Output: cluster label per player (Anchor / Aggressive / All-Rounder / Specialist / Death Hitter)
- Visualize: 2D PCA scatter plot in Streamlit

### 7.5 Use Case 4: Player Performance Regression [ONCE + RETRAIN]
- Algorithm: Gradient Boosting Regressor
- Target: predicted runs / wickets in next match
- Used by Coach Dashboard for team selection recommendation

### 7.6 MLflow Model Registry [ON-DEMAND]
File: `ml/model_registry.py`

```python
import mlflow

with mlflow.start_run():
    mlflow.log_params({"model": "xgboost", "n_estimators": 200})
    mlflow.log_metrics({"auc": 0.87, "f1": 0.81})
    mlflow.sklearn.log_model(model, "win_predictor")
    mlflow.register_model("runs:/<run_id>/win_predictor", "WinPredictor_v1")
```

Model promotion flow:
1. Train → log to MLflow (`Staging`)
2. Evaluate on holdout → if AUC > 0.80 → promote to `Production`
3. Old `Production` → moved to `Archived`
4. FastAPI always loads model tagged `Production`

Model retraining trigger:
- Airflow DAG: `model_retrain_dag.py` runs every Sunday 02:00 UTC
- Also triggered: if data drift detected (PSI > 0.2 on key features)

---

## STEP 8 — GENAI / RAG CHATBOT

### 8.1 Document Indexing [ONCE + ON-DEMAND when new docs added]
File: `genai/document_loader.py`

Documents to index:
- ICC rulebook PDF
- Match reports (scraped from Cricinfo)
- Player profile pages (web scraping with `BeautifulSoup`)
- Tournament statistics PDFs
- Team strategy documents

Steps:
1. Load documents using `LangChain.document_loaders` (PDFLoader, WebBaseLoader)
2. Split into chunks: `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)`
3. Embed using `OpenAIEmbeddings` or `HuggingFaceEmbeddings`
4. Store in ChromaDB: `Chroma.from_documents(docs, embeddings, persist_directory="data/embeddings")`

### 8.2 RAG Chatbot [ON-DEMAND — per user query]
File: `genai/rag_pipeline.py`

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma

vectorstore = Chroma(persist_directory="data/embeddings", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = ChatOpenAI(model="gpt-4o")  # or claude-sonnet-4-6

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# Example query:
result = qa_chain.invoke({"query": "What is Virat Kohli's average against left-arm pace?"})
```

### 8.3 Natural Language to SQL [ON-DEMAND]
File: `genai/nl_query.py`

Use LangChain's `SQLDatabaseChain`:
```python
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

db = SQLDatabase.from_uri("postgresql://admin:secret@localhost/icc_cricket")
chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
result = chain.run("Which team has the highest win rate at Eden Gardens in T20s?")
```

### 8.4 Chatbot Use Cases
The chatbot must handle:
1. Statistical queries: "Show me Rohit Sharma's performance in powerplay overs"
2. Rules queries: "What happens if rain interrupts in the 12th over?"
3. Strategy queries: "Who should bowl the death overs against Australia?"
4. Document queries: "What does the ICC playing condition say about DLS method?"
5. Prediction queries: "What is the predicted score if India bats first at Wankhede?"

---

## STEP 9 — FASTAPI BACKEND [RECURRING — always running]

### 9.1 Endpoints
File: `api/main.py`

```
GET  /health                          — health check
POST /predict/win_probability         — real-time win prob
POST /predict/final_score             — score prediction
GET  /players/{player_id}/stats       — player stat card
GET  /match/{match_id}/live           — live match state
POST /chat/query                      — RAG chatbot query
POST /chat/nl_sql                     — NL to SQL query
GET  /dashboard/eda/{metric}          — dashboard data APIs
GET  /dashboard/dq/report             — data quality summary
```

### 9.2 Model Loading
- Load MLflow `Production` model at startup via `mlflow.pyfunc.load_model()`
- Cache model in memory; reload on `PUT /admin/reload_model`

---

## STEP 10 — AIRFLOW ORCHESTRATION [RECURRING]

### DAG 1: Batch ETL (daily at midnight)
```
start → spark_batch_load → dbt_run_silver → dbt_run_gold → 
great_expectations_validate → notify_slack → end
```

### DAG 2: Model Retraining (weekly Sunday 02:00)
```
start → feature_engineering → train_win_predictor → 
evaluate_model → if_auc_gt_threshold → promote_to_production → 
update_fastapi → end
```

---

## STEP 11 — DOCKER COMPOSE DEPLOYMENT [ONCE]

### Services in `docker-compose.yml`:
```yaml
services:
  zookeeper:       image: confluentinc/cp-zookeeper:7.4.0
  kafka:           image: confluentinc/cp-kafka:7.4.0
  spark-master:    image: bitnami/spark:3.5
  spark-worker:    image: bitnami/spark:3.5
  postgres:        image: postgres:15
  dbt:             custom build from ./docker/Dockerfiles/Dockerfile.etl
  mlflow:          image: ghcr.io/mlflow/mlflow:v2.10.0
  airflow:         image: apache/airflow:2.8.0
  fastapi:         custom build from ./docker/Dockerfiles/Dockerfile.api
  streamlit:       custom build from ./docker/Dockerfiles/Dockerfile.dashboard
  chromadb:        image: chromadb/chroma:latest
  metabase:        image: metabase/metabase:latest
  grafana:         image: grafana/grafana:latest
```

Startup order (depends_on):
```
zookeeper → kafka → postgres → dbt → spark → mlflow → airflow → fastapi → streamlit
```

### Run the full stack:
```bash
docker-compose up --build -d
```

### Access points:
| Service | URL |
|---|---|
| Streamlit App | http://localhost:8501 |
| FastAPI Docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| Airflow UI | http://localhost:8080 |
| Metabase | http://localhost:3000 |
| Grafana | http://localhost:3001 |

---

## EXECUTION ORDER SUMMARY

### ONE-TIME SETUP (run in order):
1. `docker-compose up zookeeper kafka postgres` — start infra
2. Run `warehouse/schema.sql` — create DB schema
3. `spark_batch.py` — load historical CSV data
4. `dbt run` — build Silver and Gold tables
5. `great_expectations checkpoint run` — validate data
6. `document_loader.py` — index documents into ChromaDB
7. `feature_engineering.py` — generate ML features
8. `train_win_predictor.py` — train and register models
9. `docker-compose up --build -d` — start full stack

### CONTINUOUSLY RUNNING SERVICES:
- Kafka + Zookeeper
- `data_simulator.py` (produces events every 5s)
- `spark_streaming.py` (consumes from Kafka, writes to Bronze)
- FastAPI server
- Streamlit app

### RECURRING SCHEDULED JOBS (Airflow):
- Daily midnight: `batch_etl_dag` (dbt silver→gold + DQ)
- Weekly Sunday: `model_retrain_dag` (retrain + promote if better)

---

## HACKATHON JUDGING ALIGNMENT

| Criterion | Implementation |
|---|---|
| Real-time data | Kafka + Spark Streaming + live simulator |
| Data warehouse | Star schema in PostgreSQL, Medallion architecture |
| ETL pipeline | Spark batch + streaming, dbt, Great Expectations |
| Dashboards | Streamlit multi-page + Metabase + Grafana |
| Personas + KPIs | Coach, Analyst, Admin — 7+ KPIs each |
| ML use cases | Classification, Regression, Clustering |
| GenAI use cases | RAG chatbot, NL→SQL, PDF extraction |
| Model registry | MLflow with staging/production/archived |
| Dockerized | Full docker-compose with 12+ services |
| End-to-end | All layers connected, data flows from source to insight |

---

*Generated for ICC T20 WC 2026 Hackathon — Antigravity IDE + Claude Opus 4.6*
