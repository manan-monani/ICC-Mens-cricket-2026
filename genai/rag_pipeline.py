"""
ICC T20 Predictor - RAG Chatbot Pipeline
Uses Google Gemini + ChromaDB for conversational cricket intelligence.

Handles:
1. Statistical queries (player stats, team records)
2. Prediction queries (win probability, score prediction)
3. Strategic queries (bowling plans, batting order)
4. General cricket knowledge
"""

import os
import sys
import json
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DATA_DIR, GOOGLE_API_KEY

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RAG_CHATBOT")

DB_PATH = PROCESSED_DATA_DIR / "icc_cricket.db"
EMBEDDINGS_DIR = PROCESSED_DATA_DIR.parent / "embeddings"


class CricketDataContext:
    """Provides cricket data context for the chatbot without requiring ChromaDB."""

    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH

    def get_player_batting_stats(self, player_name: str) -> Optional[dict]:
        """Get batting stats for a player."""
        conn = sqlite3.connect(str(self.db_path))
        df = pd.read_sql(
            "SELECT * FROM agg_player_batting WHERE player_name LIKE ?",
            conn, params=[f"%{player_name}%"]
        )
        conn.close()
        if len(df) > 0:
            return df.iloc[0].to_dict()
        return None

    def get_player_bowling_stats(self, player_name: str) -> Optional[dict]:
        """Get bowling stats for a player."""
        conn = sqlite3.connect(str(self.db_path))
        df = pd.read_sql(
            "SELECT * FROM agg_player_bowling WHERE player_name LIKE ?",
            conn, params=[f"%{player_name}%"]
        )
        conn.close()
        if len(df) > 0:
            return df.iloc[0].to_dict()
        return None

    def get_team_stats(self, team_name: str) -> Optional[dict]:
        """Get team performance stats."""
        conn = sqlite3.connect(str(self.db_path))
        df = pd.read_sql(
            "SELECT * FROM agg_team_performance WHERE team_name LIKE ?",
            conn, params=[f"%{team_name}%"]
        )
        conn.close()
        if len(df) > 0:
            return df.iloc[0].to_dict()
        return None

    def get_top_batters(self, n: int = 10) -> List[dict]:
        """Get top N batters by total runs."""
        conn = sqlite3.connect(str(self.db_path))
        df = pd.read_sql(
            f"SELECT player_name, team_name, total_runs, batting_average, strike_rate, matches_played "
            f"FROM agg_player_batting ORDER BY total_runs DESC LIMIT {n}",
            conn
        )
        conn.close()
        return df.to_dict("records")

    def get_top_bowlers(self, n: int = 10) -> List[dict]:
        """Get top N bowlers by wickets."""
        conn = sqlite3.connect(str(self.db_path))
        df = pd.read_sql(
            f"SELECT player_name, team_name, wickets_taken, economy_rate, bowling_average, matches_played "
            f"FROM agg_player_bowling ORDER BY wickets_taken DESC LIMIT {n}",
            conn
        )
        conn.close()
        return df.to_dict("records")

    def get_venue_stats(self, venue_name: str = None) -> List[dict]:
        """Get venue statistics."""
        conn = sqlite3.connect(str(self.db_path))
        if venue_name:
            df = pd.read_sql(
                "SELECT * FROM dim_venues WHERE venue_name LIKE ?",
                conn, params=[f"%{venue_name}%"]
            )
        else:
            df = pd.read_sql("SELECT * FROM dim_venues ORDER BY matches_played DESC LIMIT 10", conn)
        conn.close()
        return df.to_dict("records")

    def get_head_to_head(self, team1: str, team2: str) -> dict:
        """Get head-to-head record between two teams."""
        conn = sqlite3.connect(str(self.db_path))
        df = pd.read_sql("""
            SELECT winner, COUNT(*) as wins
            FROM raw_matches 
            WHERE match_id IN (
                SELECT DISTINCT m.match_id FROM raw_matches m
                WHERE (m.toss_winner LIKE ? OR m.winner LIKE ?)
            )
            AND (winner LIKE ? OR winner LIKE ?)
            GROUP BY winner
        """, conn, params=[f"%{team1}%", f"%{team1}%", f"%{team1}%", f"%{team2}%"])
        conn.close()

        result = {"team1": team1, "team2": team2, "team1_wins": 0, "team2_wins": 0}
        for _, row in df.iterrows():
            if team1.lower() in str(row["winner"]).lower():
                result["team1_wins"] = int(row["wins"])
            elif team2.lower() in str(row["winner"]).lower():
                result["team2_wins"] = int(row["wins"])
        return result

    def get_recent_match_context(self) -> dict:
        """Get context about recent/simulated matches."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            df = pd.read_sql("""
                SELECT match_id, batting_team, 
                       SUM(runs_total) as total_runs,
                       SUM(is_wicket) as total_wickets,
                       MAX(over_number) as last_over
                FROM raw_deliveries
                WHERE source = 'stream'
                GROUP BY match_id, batting_team
                ORDER BY match_id DESC
                LIMIT 4
            """, conn)
            conn.close()
            if len(df) > 0:
                return {"live_matches": df.to_dict("records")}
        except Exception:
            conn.close()
        return {"live_matches": []}

    def build_context(self, query: str) -> str:
        """Build relevant context for the query."""
        context_parts = []
        query_lower = query.lower()

        # Detect entity types in query
        # Team detection
        conn = sqlite3.connect(str(self.db_path))
        teams = pd.read_sql("SELECT DISTINCT team_name FROM agg_team_performance", conn)
        conn.close()

        mentioned_teams = []
        for _, row in teams.iterrows():
            team = row["team_name"]
            if team.lower() in query_lower:
                mentioned_teams.append(team)
                team_stats = self.get_team_stats(team)
                if team_stats:
                    context_parts.append(f"Team Stats for {team}: {json.dumps(team_stats, default=str)}")

        # Head-to-head
        if len(mentioned_teams) >= 2:
            h2h = self.get_head_to_head(mentioned_teams[0], mentioned_teams[1])
            context_parts.append(f"Head-to-Head: {json.dumps(h2h)}")

        # Player detection  
        conn = sqlite3.connect(str(self.db_path))
        players = pd.read_sql("SELECT DISTINCT player_name FROM agg_player_batting", conn)
        conn.close()

        for _, row in players.iterrows():
            player = row["player_name"]
            # Check if player name (or last name) is in query
            name_parts = player.split()
            for part in name_parts:
                if len(part) > 2 and part.lower() in query_lower:
                    bat_stats = self.get_player_batting_stats(player)
                    if bat_stats:
                        context_parts.append(f"Batting Stats for {player}: {json.dumps(bat_stats, default=str)}")
                    bowl_stats = self.get_player_bowling_stats(player)
                    if bowl_stats:
                        context_parts.append(f"Bowling Stats for {player}: {json.dumps(bowl_stats, default=str)}")
                    break

        # General queries
        if any(word in query_lower for word in ["top", "best", "highest", "most"]):
            if any(word in query_lower for word in ["bat", "run", "score"]):
                top = self.get_top_batters(5)
                context_parts.append(f"Top 5 Batters: {json.dumps(top, default=str)}")
            if any(word in query_lower for word in ["bowl", "wicket"]):
                top = self.get_top_bowlers(5)
                context_parts.append(f"Top 5 Bowlers: {json.dumps(top, default=str)}")

        # Venue queries
        if any(word in query_lower for word in ["venue", "stadium", "ground", "pitch"]):
            venues = self.get_venue_stats()
            context_parts.append(f"Venue Stats: {json.dumps(venues, default=str)}")

        # Live match context
        if any(word in query_lower for word in ["live", "current", "now", "today"]):
            live = self.get_recent_match_context()
            if live["live_matches"]:
                context_parts.append(f"Live Match Data: {json.dumps(live, default=str)}")

        return "\n\n".join(context_parts) if context_parts else "No specific data found. Please provide general cricket knowledge."


class CricketChatbot:
    """RAG-powered cricket chatbot using Google Gemini."""

    def __init__(self, api_key: str = None, db_path=None):
        self.api_key = api_key or GOOGLE_API_KEY
        self.data_context = CricketDataContext(db_path)
        self.model = None
        self.chat_history = []
        self._init_model()

    def _init_model(self):
        """Initialize the Gemini model."""
        if not self.api_key:
            logger.warning("No API key provided. Chatbot will work in offline mode.")
            return

        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel("gemini-2.0-flash")
            logger.info("Gemini model initialized")
        except ImportError:
            logger.warning("google-generativeai not installed. Using offline mode.")
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini: {e}")

    def query(self, user_query: str) -> dict:
        """Process a user query and return a response."""
        logger.info(f"Processing query: {user_query}")

        # Build context from data
        context = self.data_context.build_context(user_query)

        # Build prompt
        system_prompt = """You are an expert ICC T20 Cricket analyst and statistician. 
You have access to comprehensive cricket data and statistics.
Answer questions accurately using the provided data context.
If the data doesn't contain the exact answer, use the available statistics to provide the best possible answer.
Always cite specific numbers and statistics when available.
Be concise but thorough.
Format your responses in a user-friendly way with bullet points and key insights."""

        full_prompt = f"""
{system_prompt}

DATA CONTEXT:
{context}

USER QUESTION: {user_query}

Please provide a detailed, data-driven answer:"""

        # Get response
        if self.model:
            try:
                response = self.model.generate_content(full_prompt)
                answer = response.text
            except Exception as e:
                logger.error(f"Gemini API error: {e}")
                answer = self._offline_response(user_query, context)
        else:
            answer = self._offline_response(user_query, context)

        result = {
            "query": user_query,
            "answer": answer,
            "context_used": context[:500] + "..." if len(context) > 500 else context,
            "source": "gemini" if self.model else "offline",
        }

        self.chat_history.append(result)
        return result

    def _offline_response(self, query: str, context: str) -> str:
        """Generate response without LLM (data-only mode)."""
        if context and context != "No specific data found. Please provide general cricket knowledge.":
            return f"Based on the available data:\n\n{context}\n\nNote: This is a data-only response. Configure a Gemini API key for AI-enhanced answers."
        return "I don't have enough data to answer this question. Please try asking about specific players, teams, or match statistics."

    def get_history(self) -> List[dict]:
        """Get chat history."""
        return self.chat_history


# Singleton instance for the API
_chatbot_instance = None


def get_chatbot(api_key: str = None) -> CricketChatbot:
    """Get or create the chatbot singleton."""
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = CricketChatbot(api_key=api_key)
    return _chatbot_instance


if __name__ == "__main__":
    # Interactive mode
    chatbot = CricketChatbot()
    print("\n🏏 ICC T20 Cricket Chatbot 🏏")
    print("Type 'quit' to exit\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ["quit", "exit", "q"]:
            break
        if not query:
            continue

        result = chatbot.query(query)
        print(f"\n🤖 {result['answer']}\n")
