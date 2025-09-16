from dataclasses import dataclass
from math import exp, log
from typing import Optional


def clamp(x: float, a: float = 0.0, b: float = 1.0) -> float:  # FIX: Use float types
    return max(a, min(b, x))


def sigmoid(z: float) -> float:
    return 1/(1+exp(-z))


def logit(p: float) -> float:
    p = clamp(p, 1e-6, 1-1e-6)
    return log(p/(1-p))


def inv_logit(z: float) -> float:
    return sigmoid(z)


@dataclass
class MLBState:
    runs_for: int
    runs_against: int
    inning: int
    top: bool
    outs: int
    runner_on_1: int = 0
    runner_on_2: int = 0
    runner_on_3: int = 0
    pregame_wp: Optional[float] = None
    run_env: float = 8.6


def _mlb_base_out_value(o: int, r1: int, r2: int, r3: int) -> float:
    return {0: 0.35, 1: 0.18, 2: 0.04}.get(o, 0.02) + (0.18*r1 + 0.32*r2 + 0.45*r3)


def _mlb_time_frac(inning: int, top: bool) -> float:
    hi = (inning-1)*2 + (0 if top else 1)
    total = 18
    return clamp((total-1-hi)/(total-1), 0.0, 1.0)  # FIX: Use float literals


def mlb_win_probability(s: MLBState) -> float:
    z = logit(s.pregame_wp if s.pregame_wp is not None else 0.5)
    lead = s.runs_for - s.runs_against
    t = _mlb_time_frac(s.inning, s.top)
    z += (0.85*(1+1.2*(1-t))*(lead/(1+3*t)))
    z += 0.25*(_mlb_base_out_value(s.outs, s.runner_on_1, s.runner_on_2, s.runner_on_3) - 0.20)
    env = clamp((s.run_env-7.5)/4.0, -0.25, 0.25)
    z -= env*(lead/5.0)
    return clamp(inv_logit(z), 0.0, 1.0)  # FIX: Use float literals


@dataclass
class NFLState:
    points_for: int
    points_against: int
    quarter: int
    clock_seconds: int
    has_possession: Optional[bool] = None
    yard_line: Optional[int] = None
    down: Optional[int] = None
    to_go: Optional[int] = None
    timeouts_remaining: Optional[int] = None
    opp_timeouts_remaining: Optional[int] = None
    pregame_wp: Optional[float] = None


def _nfl_time_frac(q: int, clk: int) -> float:
    total = 3600
    elapsed = (q-1)*900 + (900-clk)
    return clamp((total-elapsed)/total, 0.0, 1.0)  # FIX: Use float literals


def _pos_boost(has_pos: Optional[bool], yl: Optional[int], down: Optional[int], to_go: Optional[int]) -> float:
    if has_pos is None: 
        return 0.0
    base = 0.08 if has_pos else -0.08
    if has_pos and yl is not None: 
        base += (yl-50)*0.0015
    if has_pos and down is not None and to_go is not None:
        if down == 3 and to_go >= 7: 
            base -= 0.02
        if down == 4 and to_go >= 1: 
            base -= 0.05
    return base


def nfl_win_probability(s: NFLState) -> float:
    z = logit(s.pregame_wp if s.pregame_wp is not None else 0.5)
    lead = s.points_for - s.points_against
    t = _nfl_time_frac(s.quarter, s.clock_seconds)
    z += (0.10 + 0.35*(1-t))*(lead/(1+3*t))
    z += _pos_boost(s.has_possession, s.yard_line, s.down, s.to_go)
    if s.timeouts_remaining is not None and s.opp_timeouts_remaining is not None:
        z += 0.01*(s.timeouts_remaining - s.opp_timeouts_remaining)
    if s.quarter >= 4: 
        z += (1-t)*0.05
    return clamp(inv_logit(z), 0.0, 1.0)  # FIX: Use float literals