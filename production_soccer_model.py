import logging
from dataclasses import dataclass
from math import exp, log
from typing import Optional, Dict

import numpy as np

# sklearn is optional — only used if you later call train_calibration().
try:
    from sklearn.isotonic import IsotonicRegression  # type: ignore
except Exception:  # pragma: no cover
    IsotonicRegression = None  # graceful fallback

logger = logging.getLogger(__name__)
VERSION = "soccer_prod_1.0.0"


def clamp(x: float, a: float = 0.0, b: float = 1.0) -> float:
    return max(a, min(b, x))


def logit(p: float) -> float:
    p = clamp(p, 1e-6, 1 - 1e-6)
    return log(p / (1 - p))


def inv_logit(z: float) -> float:
    return 1 / (1 + exp(-z))


@dataclass
class SoccerGameState:
    """Soccer game state for win probability calculation"""
    home_score: int
    away_score: int
    minute: int
    half: int  # 1 or 2
    home_team: str = "HOME"
    away_team: str = "AWAY"

    # Advanced features
    home_xg: Optional[float] = None  # Expected goals
    away_xg: Optional[float] = None
    home_shots: Optional[int] = None
    away_shots: Optional[int] = None
    home_possession: Optional[float] = None  # 0.0 to 1.0
    red_cards_home: int = 0
    red_cards_away: int = 0
    injury_time: Optional[int] = None  # Additional minutes

    # Team strength (SPI-style ratings)
    home_spi: Optional[float] = None  # Soccer Power Index
    away_spi: Optional[float] = None
    home_attacking: Optional[float] = None
    home_defending: Optional[float] = None
    away_attacking: Optional[float] = None
    away_defending: Optional[float] = None


class ProductionSoccerModel:
    """Production Soccer Win Probability Model using SPI methodology"""

    def __init__(self):
        self.version = "1.0.0"
        self.calibrator = None
        self.is_calibrated = False

        # Base goal expectation rates (goals per 90 minutes)
        self.avg_goals_per_game = 2.6  # League average
        self.home_advantage = 0.15  # Home field advantage

        # Time-based goal scoring rates (minute to minute probability)
        self.scoring_rates_by_minute = self._initialize_scoring_rates()

        logger.info("Production Soccer model initialized")

    def _initialize_scoring_rates(self) -> Dict[int, float]:
        """Initialize minute-by-minute goal scoring rates"""
        rates = {}

        # Base rate (goals per minute in average game)
        base_rate = self.avg_goals_per_game / 90.0

        for minute in range(1, 121):  # Up to 120 minutes (extra time)
            if minute <= 45:
                # First half - consistent rate
                rates[minute] = base_rate
            elif minute <= 90:
                # Second half - slightly higher (fatigue, urgency)
                rates[minute] = base_rate * 1.1
            else:
                # Extra time - reduced rate (tired players)
                rates[minute] = base_rate * 0.8

        return rates

    def calculate_win_probability(self, state: SoccerGameState) -> float:
        """Calculate *home* win probability for given game state"""
        try:
            # Base calculation using goal difference and time
            z = logit(0.5 + self.home_advantage)  # Home advantage

            # Current score differential
            goal_diff = state.home_score - state.away_score

            # Time factor - how much time remains
            total_minutes = 90
            if state.minute > 90 and state.injury_time:
                total_minutes = 90 + state.injury_time
            elif state.minute > 90:
                total_minutes = min(state.minute + 5, 120)  # Estimate injury time

            time_remaining = max(0, total_minutes - state.minute)
            time_factor = time_remaining / 90.0

            # Score impact - diminishes as time runs out
            if time_factor > 0.1:
                score_impact = goal_diff * (0.8 + 0.6 * time_factor)
            else:
                # Very late in game - score becomes decisive
                score_impact = goal_diff * 2.5

            z += score_impact

            # Expected goals adjustment
            if state.home_xg is not None and state.away_xg is not None:
                xg_diff = state.home_xg - state.away_xg
                actual_diff = state.home_score - state.away_score

                # If xG suggests different story than scoreline
                xg_vs_actual = (xg_diff - actual_diff) * 0.15
                z += xg_vs_actual * time_factor  # More relevant with time remaining

            # Team strength adjustments (SPI-style)
            if state.home_spi is not None and state.away_spi is not None:
                spi_diff = (state.home_spi - state.away_spi) / 10.0  # Normalize
                z += spi_diff * 0.2 * time_factor

            # Red card adjustments
            red_card_diff = state.red_cards_away - state.red_cards_home
            if red_card_diff != 0:
                red_card_impact = red_card_diff * 0.3 * time_factor
                z += red_card_impact

            # Possession-based adjustment (if available)
            if state.home_possession is not None:
                possession_diff = state.home_possession - 0.5
                possession_impact = possession_diff * 0.1 * time_factor
                z += possession_impact

            # Late game dynamics
            if state.minute >= 80:
                late_game_urgency = (state.minute - 80) / 10.0

                # Losing team gets slight boost (desperation)
                if goal_diff < 0:
                    z += 0.05 * late_game_urgency
                elif goal_diff > 0:
                    # Leading team gets boost (game management)
                    z += 0.03 * late_game_urgency

            # Injury time effects
            if state.minute > 90:
                injury_time_factor = min((state.minute - 90) / 10.0, 1.0)

                # Trailing team desperation
                if goal_diff < 0:
                    z += 0.08 * injury_time_factor
                elif goal_diff > 0:
                    # Leading team advantage (time wasting)
                    z += 0.04 * injury_time_factor

            # Convert to probability
            prob = clamp(inv_logit(z), 0.02, 0.98)

            # Apply calibration if available
            if self.is_calibrated and self.calibrator:
                try:
                    calibrated_prob = self.calibrator.predict([prob])[0]
                    return clamp(float(calibrated_prob), 0.02, 0.98)
                except Exception:
                    pass

            return prob

        except Exception as e:
            logger.error(f"Error calculating Soccer win probability: {e}")
            return 0.5

    def calculate_draw_probability(self, state: SoccerGameState) -> float:
        """Calculate probability of draw (important for soccer)"""
        try:
            # Base draw probability decreases as goal difference increases
            goal_diff = abs(state.home_score - state.away_score)

            # Time factor
            time_remaining = max(0, 90 - state.minute) / 90.0

            if goal_diff == 0:
                # Currently tied
                base_draw = 0.28  # ~28% draw probability when tied
                # Increases slightly with less time remaining
                draw_prob = base_draw + (1 - time_remaining) * 0.05
            elif goal_diff == 1:
                # One goal difference
                base_draw = 0.15
                draw_prob = base_draw * time_remaining
            else:
                # Multiple goal difference
                draw_prob = max(0.05 * time_remaining, 0.01)

            return clamp(draw_prob, 0.01, 0.45)

        except Exception as e:
            logger.error(f"Error calculating draw probability: {e}")
            return 0.25

    def calculate_three_way_probabilities(self, state: SoccerGameState) -> Dict[str, float]:
        """Calculate home win, draw, away win probabilities"""
        try:
            home_win_prob = self.calculate_win_probability(state)
            draw_prob = self.calculate_draw_probability(state)
            away_win_prob = 1 - home_win_prob - draw_prob

            # Ensure probabilities sum to 1 and are positive
            total = home_win_prob + draw_prob + away_win_prob
            if total > 0:
                home_win_prob /= total
                draw_prob /= total
                away_win_prob /= total

            return {
                "home_win": max(0.01, home_win_prob),
                "draw": max(0.01, draw_prob),
                "away_win": max(0.01, away_win_prob),
            }

        except Exception as e:
            logger.error(f"Error calculating three-way probabilities: {e}")
            return {"home_win": 0.33, "draw": 0.34, "away_win": 0.33}

    def train_calibration(self, predictions: list, actual_outcomes: list):
        """Train isotonic calibration on historical data"""
        try:
            if IsotonicRegression is None:
                logger.warning("sklearn not available; skipping calibration.")
                return
            self.calibrator = IsotonicRegression(y_min=0.02, y_max=0.98, out_of_bounds="clip")
            self.calibrator.fit(predictions, actual_outcomes)
            self.is_calibrated = True
            logger.info("Soccer model calibration training complete")
        except Exception as e:
            logger.error(f"Soccer calibration training failed: {e}")

    # Convenience method (not used by adapter directly)
    def predict(self, game_data: Dict) -> float:
        """Return *home* win probability only (compat/convenience)"""
        try:
            state = _dict_to_state(game_data)
            return self.calculate_win_probability(state)
        except Exception as e:
            logger.error(f"Error in Soccer predict method: {e}")
            return 0.5


# -------- Adapter-facing shims (this is what your bot calls) --------

def _dict_to_state(game_data: Dict) -> SoccerGameState:
    home_team = (game_data.get("home_team") or {})
    away_team = (game_data.get("away_team") or {})

    return SoccerGameState(
        home_score=int(home_team.get("score", game_data.get("home_score", 0) or 0)),
        away_score=int(away_team.get("score", game_data.get("away_score", 0) or 0)),
        minute=int(game_data.get("minute", 0)),
        half=int(game_data.get("half", 1)),
        home_team=str(home_team.get("abbreviation") or home_team.get("name") or "HOME"),
        away_team=str(away_team.get("abbreviation") or away_team.get("name") or "AWAY"),
        home_xg=game_data.get("home_xg"),
        away_xg=game_data.get("away_xg"),
        red_cards_home=int(game_data.get("red_cards_home", 0)),
        red_cards_away=int(game_data.get("red_cards_away", 0)),
        home_possession=game_data.get("home_possession"),
        injury_time=game_data.get("injury_time"),
        home_spi=game_data.get("home_spi"),
        away_spi=game_data.get("away_spi"),
    )


_MODEL = ProductionSoccerModel()

def predict(game_data: Dict):
    """
    Return a dict with keys the adapter expects: {"home", "draw", "away"}.
    The adapter will pick the right one based on the market outcome (home/away/TIE).
    """
    state = _dict_to_state(game_data)
    probs = _MODEL.calculate_three_way_probabilities(state)
    return {"home": probs["home_win"], "draw": probs["draw"], "away": probs["away_win"]}

def predict_proba(game_data: Dict):
    """Alias for predict()."""
    return predict(game_data)

# Optional class-style API (the adapter will also detect this)
class Model(ProductionSoccerModel):
    def predict(self, game_data: Dict):
        state = _dict_to_state(game_data)
        probs = self.calculate_three_way_probabilities(state)
        return {"home": probs["home_win"], "draw": probs["draw"], "away": probs["away_win"]}
    predict_proba = predict


# --------- Quick self-test (run this file directly) ---------
def _test_soccer_model():
    model = ProductionSoccerModel()
    scenarios = [
        {"name": "Tied 1-1, 70'", "state": SoccerGameState(1, 1, 70, 2)},
        {"name": "Home 2-1, 85'", "state": SoccerGameState(2, 1, 85, 2)},
        {"name": "Away 0-1, 93'+5", "state": SoccerGameState(0, 1, 93, 2, injury_time=5)},
        {"name": "Home down 1-2, away red", "state": SoccerGameState(1, 2, 75, 2, red_cards_away=1)},
    ]
    print("Testing Soccer Model:")
    print("=" * 50)
    for s in scenarios:
        probs = model.calculate_three_way_probabilities(s["state"])
        print(f"{s['name']}: home={probs['home_win']:.3f}, draw={probs['draw']:.3f}, away={probs['away_win']:.3f}")
    print("Adapter-facing predict() sample:",
          predict({"home_team": {"abbreviation": "LAG", "score": 0},
                   "away_team": {"abbreviation": "CIN", "score": 0},
                   "minute": 0, "half": 1}))

if __name__ == "__main__":
    _test_soccer_model()
PY

# optional smoke test (uses the adapter-style predict())
python - <<'PY'
import production_soccer_model as m
print("predict() ->", m.predict({"home_team":{"abbreviation":"LAG","score":0},
                                 "away_team":{"abbreviation":"CIN","score":0},
                                 "minute":0,"half":1}))