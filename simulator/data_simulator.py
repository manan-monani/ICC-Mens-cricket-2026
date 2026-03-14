"""
ICC T20 Predictor - Real-Time Data Simulator
Generates ball-by-ball events simulating a live T20 match.

Features:
- Picks from actual match data to create realistic simulations
- Generates ball events every 5 seconds
- Maintains match state (score, wickets, run rate)
- Outputs to both file and optionally to Kafka
- Stores events in the database for real-time processing
"""

import json
import random
import time
import sqlite3
import logging
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATASET_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT,
    SIMULATION_INTERVAL_SECONDS, MATCH_OVERS, BALLS_PER_OVER, BALL_OUTCOMES,
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("SIMULATOR")

DB_PATH = PROCESSED_DATA_DIR / "icc_cricket.db"
SIMULATION_LOG = PROCESSED_DATA_DIR / "simulation_events.jsonl"


class MatchState:
    """Tracks the current state of a simulated match."""

    def __init__(self, match_id: int, team1: str, team2: str, venue: str, toss_winner: str, toss_decision: str):
        self.match_id = match_id
        self.team1 = team1
        self.team2 = team2
        self.venue = venue
        self.toss_winner = toss_winner
        self.toss_decision = toss_decision

        # Determine batting order
        if toss_decision == "bat":
            self.batting_first = toss_winner
            self.batting_second = team2 if toss_winner == team1 else team1
        else:
            self.batting_first = team2 if toss_winner == team1 else team1
            self.batting_second = toss_winner

        # Current state
        self.innings = 1
        self.current_batting_team = self.batting_first
        self.current_bowling_team = self.batting_second
        self.score = 0
        self.wickets = 0
        self.overs = 0
        self.balls_in_over = 0
        self.target = None  # Set after first innings
        self.is_complete = False

        # Per-innings tracking
        self.innings1_score = 0
        self.innings1_wickets = 0
        self.innings2_score = 0
        self.innings2_wickets = 0

        # Players
        self.batters = []
        self.bowlers = []
        self.current_batter_idx = 0
        self.current_bowler_idx = 0

    @property
    def run_rate(self):
        balls = self.overs * 6 + self.balls_in_over
        return (self.score / balls * 6) if balls > 0 else 0

    @property
    def required_run_rate(self):
        if self.innings == 2 and self.target:
            remaining = self.target - self.score
            balls_rem = (MATCH_OVERS * BALLS_PER_OVER) - (self.overs * 6 + self.balls_in_over)
            return (remaining / balls_rem * 6) if balls_rem > 0 else 999
        return 0

    @property
    def balls_remaining(self):
        return (MATCH_OVERS * BALLS_PER_OVER) - (self.overs * 6 + self.balls_in_over)

    @property
    def match_phase(self):
        if self.overs < 6:
            return "powerplay"
        elif self.overs >= 16:
            return "death"
        else:
            return "middle"

    def to_dict(self):
        return {
            "match_id": self.match_id,
            "innings": self.innings,
            "batting_team": self.current_batting_team,
            "bowling_team": self.current_bowling_team,
            "score": self.score,
            "wickets": self.wickets,
            "overs": self.overs,
            "balls_in_over": self.balls_in_over,
            "over_display": f"{self.overs}.{self.balls_in_over}",
            "run_rate": round(self.run_rate, 2),
            "required_run_rate": round(self.required_run_rate, 2),
            "balls_remaining": self.balls_remaining,
            "target": self.target,
            "match_phase": self.match_phase,
            "is_complete": self.is_complete,
            "innings1_score": self.innings1_score,
            "innings1_wickets": self.innings1_wickets,
        }


class CricketSimulator:
    """Simulates a live T20 cricket match ball by ball."""

    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH
        self.players_data = {}
        self.match_state: Optional[MatchState] = None
        self.events = []

    def load_player_data(self):
        """Load real player data from the database."""
        logger.info("Loading player data from database...")
        try:
            conn = sqlite3.connect(str(self.db_path))

            # Load batters
            batters = pd.read_sql("""
                SELECT player_name, team_name, batting_average, strike_rate, 
                       matches_played, total_runs
                FROM agg_player_batting 
                WHERE matches_played >= 5
                ORDER BY total_runs DESC
            """, conn)

            # Load bowlers
            bowlers = pd.read_sql("""
                SELECT player_name, team_name, economy_rate, wickets_taken, 
                       matches_played
                FROM agg_player_bowling 
                WHERE matches_played >= 5
                ORDER BY wickets_taken DESC
            """, conn)

            # Load teams
            teams = pd.read_sql("""
                SELECT team_name, matches_played, win_pct
                FROM agg_team_performance
                WHERE matches_played >= 10
                ORDER BY win_pct DESC
            """, conn)

            conn.close()

            self.players_data = {
                "batters": batters,
                "bowlers": bowlers,
                "teams": teams,
            }
            logger.info(f"Loaded {len(batters)} batters, {len(bowlers)} bowlers, {len(teams)} teams")
            return True
        except Exception as e:
            logger.error(f"Failed to load player data: {e}")
            return False

    def setup_match(self, team1: str = None, team2: str = None) -> MatchState:
        """Set up a new match simulation."""
        teams = self.players_data.get("teams")

        if teams is None or len(teams) == 0:
            # Use default top teams if no data
            available_teams = ["India", "Australia", "England", "Pakistan", 
                             "New Zealand", "South Africa", "West Indies", "Sri Lanka"]
        else:
            available_teams = teams["team_name"].tolist()[:20]

        if team1 is None:
            team1, team2 = random.sample(available_teams[:10], 2)
        elif team2 is None:
            remaining = [t for t in available_teams if t != team1]
            team2 = random.choice(remaining[:10])

        # Random toss
        toss_winner = random.choice([team1, team2])
        toss_decision = random.choice(["bat", "field"])

        # Venue
        venues = ["Wankhede Stadium, Mumbai", "MCG, Melbourne", "Eden Gardens, Kolkata",
                  "Lord's, London", "SCG, Sydney", "Kensington Oval, Bridgetown",
                  "Dubai International Stadium, Dubai", "Newlands, Cape Town"]
        venue = random.choice(venues)

        match_id = int(datetime.now().timestamp())

        self.match_state = MatchState(
            match_id=match_id,
            team1=team1,
            team2=team2,
            venue=venue,
            toss_winner=toss_winner,
            toss_decision=toss_decision,
        )

        # Assign players
        self._assign_players(team1, team2)

        logger.info(f"Match Setup: {team1} vs {team2} at {venue}")
        logger.info(f"Toss: {toss_winner} won and chose to {toss_decision}")
        logger.info(f"Batting first: {self.match_state.batting_first}")

        return self.match_state

    def _assign_players(self, team1: str, team2: str):
        """Assign batters and bowlers for each team."""
        batters = self.players_data.get("batters", pd.DataFrame())
        bowlers = self.players_data.get("bowlers", pd.DataFrame())

        for team in [team1, team2]:
            # Get team batters (or generate placeholder names)
            team_batters = batters[batters["team_name"] == team].head(7)["player_name"].tolist()
            if len(team_batters) < 7:
                # Fill with generic names from data
                extra = batters.head(20)["player_name"].tolist()
                team_batters.extend([p for p in extra if p not in team_batters][:7 - len(team_batters)])

            team_bowlers = bowlers[bowlers["team_name"] == team].head(5)["player_name"].tolist()
            if len(team_bowlers) < 5:
                extra = bowlers.head(15)["player_name"].tolist()
                team_bowlers.extend([p for p in extra if p not in team_bowlers][:5 - len(team_bowlers)])

            if team == self.match_state.batting_first:
                self.match_state.batters = team_batters
            else:
                self.match_state.bowlers = team_bowlers

    def simulate_ball(self) -> dict:
        """Simulate a single ball delivery and return the event."""
        if self.match_state.is_complete:
            return None

        state = self.match_state

        # Determine outcome based on probabilities
        # Adjust probabilities based on match phase
        probs = dict(BALL_OUTCOMES)
        if state.match_phase == "powerplay":
            probs[4] = 0.12
            probs[6] = 0.06
            probs["wicket"] = 0.04
        elif state.match_phase == "death":
            probs[4] = 0.11
            probs[6] = 0.08
            probs[0] = 0.30
            probs["wicket"] = 0.06

        # If chasing and score close to target, adjust
        if state.innings == 2 and state.target:
            needed = state.target - state.score
            if needed <= 20 and state.wickets < 5:
                probs[4] = 0.13
                probs[6] = 0.07

        # Normalize probabilities
        total_prob = sum(v for v in probs.values())
        for k in probs:
            probs[k] = probs[k] / total_prob

        # Sample outcome
        outcomes = list(probs.keys())
        weights = [probs[k] for k in outcomes]
        outcome = random.choices(outcomes, weights=weights, k=1)[0]

        # Extras (5% chance of wide or no-ball)
        extras_type = None
        extras_runs = 0
        if random.random() < 0.05:
            extras_type = random.choice(["wide", "noball"])
            extras_runs = 1

        # Process outcome
        is_wicket = False
        wicket_kind = None
        runs_batter = 0

        if outcome == "wicket":
            is_wicket = True
            wicket_kind = random.choice(["bowled", "caught", "lbw", "run out", "stumped", "caught and bowled"])
            runs_batter = 0
            state.wickets += 1
        else:
            runs_batter = outcome

        state.score += runs_batter + extras_runs
        runs_total = runs_batter + extras_runs

        # Get current players
        batter = state.batters[min(state.current_batter_idx, len(state.batters) - 1)] if state.batters else "Batter"
        bowler = state.bowlers[min(state.current_bowler_idx, len(state.bowlers) - 1)] if state.bowlers else "Bowler"

        # Advance ball count (extras don't count as balls)
        if extras_type != "wide" and extras_type != "noball":
            state.balls_in_over += 1
            if state.balls_in_over >= BALLS_PER_OVER:
                state.overs += 1
                state.balls_in_over = 0
                state.current_bowler_idx = (state.current_bowler_idx + 1) % max(len(state.bowlers), 1)

        if is_wicket:
            state.current_batter_idx = min(state.current_batter_idx + 1, max(len(state.batters) - 1, 0))

        # Build event
        event = {
            "timestamp": datetime.now().isoformat(),
            "match_id": state.match_id,
            "innings_number": state.innings,
            "over_number": state.overs,
            "ball_in_over": state.balls_in_over,
            "batting_team": state.current_batting_team,
            "bowling_team": state.current_bowling_team,
            "batter": batter,
            "bowler": bowler,
            "runs_batter": runs_batter,
            "runs_extras": extras_runs,
            "runs_total": runs_total,
            "extras_type": extras_type,
            "is_wicket": is_wicket,
            "wicket_kind": wicket_kind,
            "match_phase": state.match_phase,
            "score": state.score,
            "wickets": state.wickets,
            "run_rate": round(state.run_rate, 2),
            "required_run_rate": round(state.required_run_rate, 2),
            "balls_remaining": state.balls_remaining,
            "target": state.target,
        }

        # Check innings/match end conditions
        if state.wickets >= 10 or (state.overs >= MATCH_OVERS and state.balls_in_over == 0):
            if state.innings == 1:
                # End of first innings
                state.innings1_score = state.score
                state.innings1_wickets = state.wickets
                state.target = state.score + 1
                state.innings = 2
                state.score = 0
                state.wickets = 0
                state.overs = 0
                state.balls_in_over = 0

                # Swap batting/bowling
                state.current_batting_team, state.current_bowling_team = (
                    state.current_bowling_team,
                    state.current_batting_team,
                )
                # Swap player lists
                old_batters = state.batters
                state.batters = state.bowlers
                state.bowlers = old_batters
                state.current_batter_idx = 0
                state.current_bowler_idx = 0

                event["innings_complete"] = True
                logger.info(f"First innings complete: {state.innings1_score}/{state.innings1_wickets}. Target: {state.target}")
            else:
                # End of match
                state.innings2_score = state.score
                state.innings2_wickets = state.wickets
                state.is_complete = True
                event["match_complete"] = True

        # Check if chase complete
        if state.innings == 2 and state.target and state.score >= state.target:
            state.innings2_score = state.score
            state.innings2_wickets = state.wickets
            state.is_complete = True
            event["match_complete"] = True

        if state.is_complete:
            if state.innings2_score >= (state.target or 999):
                winner = state.current_batting_team
                margin = f"{10 - state.wickets} wickets"
            else:
                winner = state.current_bowling_team
                margin = f"{(state.target or 0) - state.innings2_score - 1} runs"
            event["winner"] = winner
            event["win_margin"] = margin
            logger.info(f"Match Complete! {winner} won by {margin}")

        self.events.append(event)
        return event

    def save_event_to_file(self, event: dict):
        """Append event to JSONL file."""
        with open(SIMULATION_LOG, "a") as f:
            f.write(json.dumps(event) + "\n")

    def save_event_to_db(self, event: dict):
        """Save event to database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO raw_deliveries 
                (match_id, innings_number, over_number, ball_in_over, batting_team,
                 batter, bowler, runs_batter, runs_extras, runs_total,
                 is_wicket, wicket_kind, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'stream')
            """, (
                event["match_id"], event["innings_number"], event["over_number"],
                event["ball_in_over"], event["batting_team"],
                event["batter"], event["bowler"], event["runs_batter"],
                event["runs_extras"], event["runs_total"],
                1 if event["is_wicket"] else 0, event.get("wicket_kind"),
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to save event to DB: {e}")

    def run_simulation(self, team1: str = None, team2: str = None, interval: float = None):
        """Run a full match simulation."""
        interval = interval or SIMULATION_INTERVAL_SECONDS

        logger.info("=" * 60)
        logger.info("STARTING MATCH SIMULATION")
        logger.info("=" * 60)

        # Load data
        self.load_player_data()

        # Setup match
        self.setup_match(team1, team2)
        state = self.match_state

        logger.info(f"\n{'='*40}")
        logger.info(f"  {state.team1} vs {state.team2}")
        logger.info(f"  Venue: {state.venue}")
        logger.info(f"  Toss: {state.toss_winner} ({state.toss_decision})")
        logger.info(f"{'='*40}\n")

        ball_count = 0
        while not state.is_complete:
            event = self.simulate_ball()
            if event is None:
                break

            ball_count += 1

            # Log every ball
            wicket_str = f" WICKET! ({event.get('wicket_kind', '')})" if event["is_wicket"] else ""
            logger.info(
                f"  {event['over_number']}.{event['ball_in_over']} | "
                f"{event['batting_team']} {event['score']}/{event['wickets']} | "
                f"{event['batter']} → {event['runs_batter']} runs "
                f"(bowler: {event['bowler']}){wicket_str}"
            )

            # Save event
            self.save_event_to_file(event)
            self.save_event_to_db(event)

            # Sleep between balls
            time.sleep(interval)

        logger.info(f"\nSimulation complete. {ball_count} balls bowled.")
        return self.events


def main():
    """Main entry point for the simulator."""
    import argparse

    parser = argparse.ArgumentParser(description="ICC T20 Match Simulator")
    parser.add_argument("--team1", type=str, help="Team 1 name")
    parser.add_argument("--team2", type=str, help="Team 2 name")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between balls (default: 1.0)")
    parser.add_argument("--fast", action="store_true", help="Run instantly (no delay)")

    args = parser.parse_args()
    interval = 0.01 if args.fast else args.interval

    simulator = CricketSimulator()
    simulator.run_simulation(team1=args.team1, team2=args.team2, interval=interval)


if __name__ == "__main__":
    main()
