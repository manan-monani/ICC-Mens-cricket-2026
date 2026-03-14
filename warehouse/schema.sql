-- ============================================================
-- ICC T20 Cricket World Cup 2026 - Data Warehouse Schema
-- Star Schema Design with Medallion Architecture
-- ============================================================

-- ===================== BRONZE LAYER =====================
-- Raw ingested data - as-is from source/Kafka

CREATE TABLE IF NOT EXISTS raw_ball_events (
    id                  SERIAL PRIMARY KEY,
    match_id            INTEGER NOT NULL,
    innings_number      INTEGER,
    over_number         INTEGER,
    ball_in_over        INTEGER,
    batting_team        VARCHAR(100),
    batter              VARCHAR(100),
    bowler              VARCHAR(100),
    non_striker         VARCHAR(100),
    runs_batter         INTEGER,
    runs_extras         INTEGER,
    runs_total          INTEGER,
    extras_byes         INTEGER DEFAULT 0,
    extras_legbyes      INTEGER DEFAULT 0,
    extras_noballs      INTEGER DEFAULT 0,
    extras_wides        INTEGER DEFAULT 0,
    extras_penalty      INTEGER DEFAULT 0,
    is_wicket           BOOLEAN DEFAULT FALSE,
    wicket_kind         VARCHAR(50),
    wicket_player_out   VARCHAR(100),
    wicket_fielders     VARCHAR(200),
    ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source              VARCHAR(20) DEFAULT 'batch',  -- 'batch' or 'stream'
    status              VARCHAR(20) DEFAULT 'pending' -- 'pending', 'processed', 'failed'
);

CREATE TABLE IF NOT EXISTS raw_matches (
    id                  SERIAL PRIMARY KEY,
    match_id            INTEGER NOT NULL UNIQUE,
    data_version        VARCHAR(20),
    created             VARCHAR(20),
    revision            INTEGER,
    match_date          DATE,
    season              VARCHAR(20),
    event_name          VARCHAR(200),
    event_match_number  INTEGER,
    match_type          VARCHAR(10),
    match_type_number   INTEGER,
    gender              VARCHAR(10),
    team_type           VARCHAR(20),
    venue               VARCHAR(200),
    city                VARCHAR(100),
    overs               INTEGER,
    balls_per_over      INTEGER,
    toss_winner         VARCHAR(100),
    toss_decision       VARCHAR(10),
    winner              VARCHAR(100),
    result_type         VARCHAR(20),
    result_margin       FLOAT,
    result_text         VARCHAR(300),
    method              VARCHAR(50),
    ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS raw_innings (
    id                  SERIAL PRIMARY KEY,
    match_id            INTEGER NOT NULL,
    innings_number      INTEGER,
    team                VARCHAR(100),
    total_runs          INTEGER,
    total_wickets       INTEGER,
    total_balls         INTEGER,
    extras_byes         INTEGER DEFAULT 0,
    extras_legbyes      INTEGER DEFAULT 0,
    extras_noballs      INTEGER DEFAULT 0,
    extras_wides        INTEGER DEFAULT 0,
    extras_penalty      INTEGER DEFAULT 0,
    ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ===================== DIMENSION TABLES (GOLD) =====================

CREATE TABLE IF NOT EXISTS dim_teams (
    team_id     SERIAL PRIMARY KEY,
    team_name   VARCHAR(100) NOT NULL UNIQUE,
    country     VARCHAR(100),
    team_type   VARCHAR(20),   -- 'international', 'associate'
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dim_players (
    player_id       SERIAL PRIMARY KEY,
    player_name     VARCHAR(100) NOT NULL,
    team_name       VARCHAR(100),
    role            VARCHAR(30),  -- 'batter', 'bowler', 'allrounder', 'wicketkeeper'
    batting_style   VARCHAR(30),  -- 'right-hand', 'left-hand'
    bowling_style   VARCHAR(50),  -- 'right-arm fast', 'left-arm spin', etc.
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(player_name, team_name)
);

CREATE TABLE IF NOT EXISTS dim_venues (
    venue_id                SERIAL PRIMARY KEY,
    venue_name              VARCHAR(200) NOT NULL,
    city                    VARCHAR(100),
    country                 VARCHAR(100),
    avg_first_innings_score FLOAT,
    avg_second_innings_score FLOAT,
    matches_played          INTEGER DEFAULT 0,
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(venue_name, city)
);

CREATE TABLE IF NOT EXISTS dim_dates (
    date_id         SERIAL PRIMARY KEY,
    full_date       DATE NOT NULL UNIQUE,
    year            INTEGER,
    month           INTEGER,
    day             INTEGER,
    day_of_week     VARCHAR(10),
    quarter         INTEGER,
    season          VARCHAR(20),
    is_weekend      BOOLEAN
);

-- ===================== FACT TABLES (GOLD) =====================

CREATE TABLE IF NOT EXISTS fact_ball_events (
    ball_event_id       SERIAL PRIMARY KEY,
    match_id            INTEGER NOT NULL,
    innings_number      INTEGER NOT NULL,
    over_number         INTEGER NOT NULL,
    ball_in_over        INTEGER NOT NULL,
    batting_team_id     INTEGER REFERENCES dim_teams(team_id),
    bowling_team_id     INTEGER REFERENCES dim_teams(team_id),
    batter_id           INTEGER REFERENCES dim_players(player_id),
    bowler_id           INTEGER REFERENCES dim_players(player_id),
    non_striker_id      INTEGER REFERENCES dim_players(player_id),
    venue_id            INTEGER REFERENCES dim_venues(venue_id),
    date_id             INTEGER REFERENCES dim_dates(date_id),
    runs_batter         INTEGER DEFAULT 0,
    runs_extras         INTEGER DEFAULT 0,
    runs_total          INTEGER DEFAULT 0,
    is_boundary_four    BOOLEAN DEFAULT FALSE,
    is_boundary_six     BOOLEAN DEFAULT FALSE,
    is_dot_ball         BOOLEAN DEFAULT FALSE,
    is_wicket           BOOLEAN DEFAULT FALSE,
    wicket_kind         VARCHAR(50),
    extras_type         VARCHAR(20),
    match_phase         VARCHAR(20),  -- 'powerplay', 'middle', 'death'
    cumulative_score    INTEGER,
    cumulative_wickets  INTEGER,
    current_run_rate    FLOAT,
    UNIQUE(match_id, innings_number, over_number, ball_in_over)
);

CREATE TABLE IF NOT EXISTS fact_match_results (
    match_result_id     SERIAL PRIMARY KEY,
    match_id            INTEGER NOT NULL UNIQUE,
    team1_id            INTEGER REFERENCES dim_teams(team_id),
    team2_id            INTEGER REFERENCES dim_teams(team_id),
    winner_id           INTEGER REFERENCES dim_teams(team_id),
    venue_id            INTEGER REFERENCES dim_venues(venue_id),
    date_id             INTEGER REFERENCES dim_dates(date_id),
    season              VARCHAR(20),
    event_name          VARCHAR(200),
    toss_winner_id      INTEGER REFERENCES dim_teams(team_id),
    toss_decision       VARCHAR(10),
    result_type         VARCHAR(20),
    result_margin       FLOAT,
    team1_score         INTEGER,
    team1_wickets       INTEGER,
    team1_overs         FLOAT,
    team2_score         INTEGER,
    team2_wickets       INTEGER,
    team2_overs         FLOAT,
    first_bat_team_id   INTEGER REFERENCES dim_teams(team_id),
    first_bat_won       BOOLEAN,
    method              VARCHAR(50),
    match_phase         VARCHAR(30)  -- 'group', 'super8', 'semifinal', 'final'
);

-- ===================== AGGREGATION TABLES (GOLD) =====================

CREATE TABLE IF NOT EXISTS agg_player_batting (
    id                  SERIAL PRIMARY KEY,
    player_id           INTEGER REFERENCES dim_players(player_id),
    player_name         VARCHAR(100),
    team_name           VARCHAR(100),
    matches_played      INTEGER DEFAULT 0,
    innings_batted      INTEGER DEFAULT 0,
    total_runs          INTEGER DEFAULT 0,
    balls_faced         INTEGER DEFAULT 0,
    batting_average     FLOAT,
    strike_rate         FLOAT,
    highest_score       INTEGER DEFAULT 0,
    fifties             INTEGER DEFAULT 0,
    hundreds            INTEGER DEFAULT 0,
    fours               INTEGER DEFAULT 0,
    sixes               INTEGER DEFAULT 0,
    dot_ball_pct        FLOAT,
    boundary_pct        FLOAT,
    powerplay_sr        FLOAT,  -- strike rate in powerplay
    death_sr            FLOAT,  -- strike rate in death overs
    consistency_score   FLOAT,  -- std dev based consistency
    form_index          FLOAT,  -- weighted recent performance
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(player_id)
);

CREATE TABLE IF NOT EXISTS agg_player_bowling (
    id                  SERIAL PRIMARY KEY,
    player_id           INTEGER REFERENCES dim_players(player_id),
    player_name         VARCHAR(100),
    team_name           VARCHAR(100),
    matches_played      INTEGER DEFAULT 0,
    innings_bowled      INTEGER DEFAULT 0,
    balls_bowled        INTEGER DEFAULT 0,
    runs_conceded       INTEGER DEFAULT 0,
    wickets_taken       INTEGER DEFAULT 0,
    economy_rate        FLOAT,
    bowling_average     FLOAT,
    bowling_strike_rate FLOAT,
    best_figures        VARCHAR(20),
    dot_ball_pct        FLOAT,
    boundary_conceded_pct FLOAT,
    powerplay_economy   FLOAT,
    death_economy       FLOAT,
    four_wicket_hauls   INTEGER DEFAULT 0,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(player_id)
);

CREATE TABLE IF NOT EXISTS agg_team_performance (
    id                  SERIAL PRIMARY KEY,
    team_id             INTEGER REFERENCES dim_teams(team_id),
    team_name           VARCHAR(100),
    matches_played      INTEGER DEFAULT 0,
    matches_won         INTEGER DEFAULT 0,
    matches_lost        INTEGER DEFAULT 0,
    no_results          INTEGER DEFAULT 0,
    win_pct             FLOAT,
    avg_score_batting   FLOAT,
    avg_score_bowling   FLOAT,
    avg_run_rate        FLOAT,
    toss_win_pct        FLOAT,
    bat_first_win_pct   FLOAT,
    chase_win_pct       FLOAT,
    powerplay_avg_score FLOAT,
    death_overs_avg_score FLOAT,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(team_id)
);

-- ===================== ML FEATURE STORE =====================

CREATE TABLE IF NOT EXISTS ml_match_features (
    id                      SERIAL PRIMARY KEY,
    match_id                INTEGER NOT NULL,
    innings_number          INTEGER,
    over_number             INTEGER,
    -- Match context
    venue_id                INTEGER,
    team_batting_id         INTEGER,
    team_bowling_id         INTEGER,
    toss_winner_batting     BOOLEAN,
    -- Current state
    current_score           INTEGER,
    current_wickets         INTEGER,
    current_run_rate        FLOAT,
    required_run_rate       FLOAT,
    balls_remaining         INTEGER,
    -- Team features
    team_bat_win_rate       FLOAT,
    team_bowl_win_rate      FLOAT,
    head_to_head_wins       INTEGER,
    team_avg_score_venue    FLOAT,
    -- Phase features
    powerplay_score         INTEGER,
    powerplay_wickets       INTEGER,
    -- Derived
    pressure_index          FLOAT,
    momentum_score          FLOAT,
    -- Target
    match_winner            INTEGER,  -- 1 = batting team wins, 0 = bowling team wins
    final_score             INTEGER,
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ===================== INDEXES =====================

CREATE INDEX IF NOT EXISTS idx_raw_ball_match ON raw_ball_events(match_id);
CREATE INDEX IF NOT EXISTS idx_raw_ball_status ON raw_ball_events(status);
CREATE INDEX IF NOT EXISTS idx_fact_ball_match ON fact_ball_events(match_id);
CREATE INDEX IF NOT EXISTS idx_fact_ball_batter ON fact_ball_events(batter_id);
CREATE INDEX IF NOT EXISTS idx_fact_ball_bowler ON fact_ball_events(bowler_id);
CREATE INDEX IF NOT EXISTS idx_fact_ball_venue ON fact_ball_events(venue_id);
CREATE INDEX IF NOT EXISTS idx_fact_match_winner ON fact_match_results(winner_id);
CREATE INDEX IF NOT EXISTS idx_fact_match_venue ON fact_match_results(venue_id);
CREATE INDEX IF NOT EXISTS idx_fact_match_date ON fact_match_results(date_id);
CREATE INDEX IF NOT EXISTS idx_ml_features_match ON ml_match_features(match_id);

-- ============================================================
-- END OF SCHEMA
-- ============================================================
