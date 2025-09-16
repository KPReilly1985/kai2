#!/usr/bin/env python3
"""
Production MLB Win Probability Model
Based on successful validation patterns from previous development
"""

import numpy as np
from dataclasses import dataclass
from math import exp, log
from typing import Optional, Dict
from sklearn.isotonic import IsotonicRegression
import logging

logger = logging.getLogger(__name__)

def clamp(x: float, a: float = 0.0, b: float = 1.0) -> float:
    return max(a, min(b, x))

def logit(p: float) -> float:
    p = clamp(p, 1e-6, 1-1e-6)
    return log(p/(1-p))

def inv_logit(z: float) -> float:
    return 1/(1+exp(-z))

@dataclass
class MLBGameState:
    """MLB game state for win probability calculation"""
    home_score: int
    away_score: int
    inning: int
    top_bottom: str  # "top" or "bottom"
    outs: int
    runners_on_base: Dict[str, bool]  # {"1": True, "2": False, "3": True}
    home_team: str = "HOME"
    away_team: str = "AWAY"
    
    # Advanced features
    pitcher_handedness: Optional[str] = None  # "L" or "R"
    batter_handedness: Optional[str] = None   # "L" or "R" 
    pitch_count: Optional[Dict[str, int]] = None  # {"balls": 2, "strikes": 1}
    leverage_index: Optional[float] = None
    weather_factor: Optional[float] = None  # -0.1 to 0.1
    
    def __post_init__(self):
        # Ensure runners_on_base has all bases
        if not isinstance(self.runners_on_base, dict):
            self.runners_on_base = {"1": False, "2": False, "3": False}
        for base in ["1", "2", "3"]:
            if base not in self.runners_on_base:
                self.runners_on_base[base] = False

class ProductionMLBModel:
    """Production MLB Win Probability Model"""
    
    def __init__(self):
        self.version = "1.0.0"
        self.calibrator = None
        self.is_calibrated = False
        
        # Base-out run expectancy matrix (MLB average)
        self.run_expectancy = {
            (0, 0): 0.481, (0, 1): 0.254, (0, 2): 0.095,  # 0 outs
            (1, 0): 0.254, (1, 1): 0.140, (1, 2): 0.049,  # 1 out
            (2, 0): 0.095, (2, 1): 0.049, (2, 2): 0.024   # 2 outs
        }
        
        # Base states (binary representation: 1B, 2B, 3B)
        self.base_states = {
            (False, False, False): 0,  # Bases empty
            (True, False, False): 1,   # Runner on 1st
            (False, True, False): 2,   # Runner on 2nd
            (True, True, False): 3,    # Runners on 1st & 2nd
            (False, False, True): 4,   # Runner on 3rd
            (True, False, True): 5,    # Runners on 1st & 3rd
            (False, True, True): 6,    # Runners on 2nd & 3rd
            (True, True, True): 7      # Bases loaded
        }
        
        # Run expectancy by base-out state
        self.base_out_matrix = {
            # (base_state, outs): run_expectancy
            (0, 0): 0.481, (0, 1): 0.254, (0, 2): 0.095,
            (1, 0): 0.831, (1, 1): 0.489, (1, 2): 0.214,
            (2, 0): 1.068, (2, 1): 0.644, (2, 2): 0.305,
            (3, 0): 1.373, (3, 1): 0.908, (3, 2): 0.471,
            (4, 0): 1.426, (4, 1): 0.897, (4, 2): 0.382,
            (5, 0): 1.798, (5, 1): 1.140, (5, 2): 0.573,
            (6, 0): 2.052, (6, 1): 1.377, (6, 2): 0.736,
            (7, 0): 2.282, (7, 1): 1.541, (7, 2): 0.815
        }
        
        logger.info("Production MLB model initialized")
    
    def calculate_win_probability(self, state: MLBGameState) -> float:
        """Calculate win probability for given game state"""
        try:
            # Base logit calculation
            z = logit(0.54)  # Home field advantage (~54%)
            
            # Core scoring differential
            run_diff = state.home_score - state.away_score
            
            # Time factor - how much of game remains
            innings_remaining = 9.5 - (state.inning - 0.5)
            if state.top_bottom == "bottom":
                innings_remaining -= 0.5
            
            time_factor = max(0.1, innings_remaining / 9.0)
            
            # Score impact decreases as time factor decreases
            score_impact = (0.25 + 0.35 * time_factor) * (run_diff / (1 + 2.5 * (1 - time_factor)))
            z += score_impact
            
            # Base-out situation impact
            base_out_value = self._calculate_base_out_value(state)
            z += base_out_value * 0.15  # Scale factor for base-out impact
            
            # Late-game adjustments
            if state.inning >= 7:
                late_game_factor = (state.inning - 6) / 3.0
                z += late_game_factor * 0.03 * (1 if run_diff >= 0 else -1)
            
            # Extra innings
            if state.inning > 9:
                extra_innings_factor = min((state.inning - 9) * 0.02, 0.08)
                z += extra_innings_factor * (1 if run_diff >= 0 else -1)
            
            # Bottom of 9th or later - walk-off potential
            if state.inning >= 9 and state.top_bottom == "bottom" and run_diff <= 0:
                walkoff_pressure = 0.05 if run_diff == 0 else 0.03
                z += walkoff_pressure
            
            # Leverage adjustments
            if state.leverage_index is not None:
                leverage_adjustment = min(state.leverage_index - 1.0, 2.0) * 0.01
                z += leverage_adjustment * (1 if run_diff >= 0 else -1)
            
            # Count-based adjustments
            if state.pitch_count:
                count_factor = self._calculate_count_factor(state.pitch_count)
                z += count_factor
            
            # Platoon advantage
            if state.pitcher_handedness and state.batter_handedness:
                platoon_factor = self._calculate_platoon_advantage(
                    state.pitcher_handedness, state.batter_handedness
                )
                z += platoon_factor
            
            # Weather factor
            if state.weather_factor is not None:
                z += state.weather_factor * 0.5
            
            # Convert to probability
            prob = clamp(inv_logit(z), 0.02, 0.98)
            
            # Apply calibration if available
            if self.is_calibrated and self.calibrator:
                try:
                    calibrated_prob = self.calibrator.predict([prob])[0]
                    return clamp(calibrated_prob, 0.02, 0.98)
                except:
                    pass
            
            return prob
            
        except Exception as e:
            logger.error(f"Error calculating MLB win probability: {e}")
            return 0.5
    
    def _calculate_base_out_value(self, state: MLBGameState) -> float:
        """Calculate run expectancy value from base-out state"""
        try:
            # Convert runners to base state
            runners = state.runners_on_base
            base_state_tuple = (
                runners.get("1", False),
                runners.get("2", False), 
                runners.get("3", False)
            )
            
            base_state = self.base_states.get(base_state_tuple, 0)
            outs = min(max(state.outs, 0), 2)
            
            run_expectancy = self.base_out_matrix.get((base_state, outs), 0.481)
            
            # Normalize around league average (empty bases, 0 outs)
            baseline = self.base_out_matrix[(0, 0)]
            return (run_expectancy - baseline) / 2.0  # Scale factor
            
        except Exception as e:
            logger.error(f"Error calculating base-out value: {e}")
            return 0.0
    
    def _calculate_count_factor(self, pitch_count: Dict[str, int]) -> float:
        """Calculate adjustment based on current count"""
        try:
            balls = pitch_count.get("balls", 0)
            strikes = pitch_count.get("strikes", 0)
            
            # Hitter-friendly counts
            if balls >= 2 and strikes <= 1:
                return 0.015  # Slight boost for batting team
            # Pitcher-friendly counts
            elif strikes >= 2 and balls <= 1:
                return -0.015  # Slight boost for pitching team
            else:
                return 0.0
                
        except:
            return 0.0
    
    def _calculate_platoon_advantage(self, pitcher_hand: str, batter_hand: str) -> float:
        """Calculate platoon advantage (same-handed vs opposite-handed)"""
        try:
            # Opposite-handed matchups favor the batter slightly
            if pitcher_hand != batter_hand:
                return 0.008  # Small boost for batting team
            else:
                return -0.005  # Small boost for pitching team
        except:
            return 0.0
    
    def train_calibration(self, predictions: list, actual_outcomes: list):
        """Train isotonic calibration on historical data"""
        try:
            self.calibrator = IsotonicRegression(
                y_min=0.02,
                y_max=0.98,
                out_of_bounds='clip'
            )
            self.calibrator.fit(predictions, actual_outcomes)
            self.is_calibrated = True
            logger.info("MLB model calibration training complete")
        except Exception as e:
            logger.error(f"MLB calibration training failed: {e}")
    
    def predict(self, game_data: Dict) -> float:
        """Compatibility method for enhanced model manager"""
        try:
            # Convert dict to MLBGameState
            runners = game_data.get("runners", {"1": False, "2": False, "3": False})
            
            state = MLBGameState(
                home_score=game_data.get("home_score", 0),
                away_score=game_data.get("away_score", 0),
                inning=game_data.get("inning", 1),
                top_bottom=game_data.get("top_bottom", "top"),
                outs=game_data.get("outs", 0),
                runners_on_base=runners
            )
            
            return self.calculate_win_probability(state)
            
        except Exception as e:
            logger.error(f"Error in MLB predict method: {e}")
            return 0.5


# Test the model
def test_mlb_model():
    """Test the MLB model with various game situations"""
    model = ProductionMLBModel()
    
    test_scenarios = [
        {
            "name": "Tied game, 7th inning, bases loaded",
            "state": MLBGameState(
                home_score=3,
                away_score=3,
                inning=7,
                top_bottom="bottom",
                outs=1,
                runners_on_base={"1": True, "2": True, "3": True}
            )
        },
        {
            "name": "Home team leading by 1, bottom 9th",
            "state": MLBGameState(
                home_score=5,
                away_score=4,
                inning=9,
                top_bottom="bottom",
                outs=2,
                runners_on_base={"1": False, "2": True, "3": False}
            )
        },
        {
            "name": "Away team leading, top 9th",
            "state": MLBGameState(
                home_score=2,
                away_score=4,
                inning=9,
                top_bottom="top",
                outs=0,
                runners_on_base={"1": True, "2": False, "3": False}
            )
        }
    ]
    
    print("Testing MLB Model:")
    print("=" * 50)
    
    for scenario in test_scenarios:
        wp = model.calculate_win_probability(scenario["state"])
        print(f"{scenario['name']}: {wp:.3f} ({wp*100:.1f}%)")
    
    print(f"\nModel version: {model.version}")

if __name__ == "__main__":
    test_mlb_model()