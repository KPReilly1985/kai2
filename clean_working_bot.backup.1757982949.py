#!/usr/bin/env python3
"""
Real Kalshi API Trading Bot - FIXED VERSION
==========================================
UPDATED to handle actual Kalshi ticker formats and account balance issues
and to support:
  • Reading LIVE Kalshi markets in paper mode
  • A "listen-only" mode that logs signals but places no orders (toggle)
  • Collapsing multiple outcomes per match to a single best-side signal (toggle)
  • Optional ESPN-live-only filtering (toggle)
  • Proper tradeable price handling (exit on BID, not ASK)
  • Ignore untradeable quotes (0/100) + min-hold to prevent instant stops
"""

import asyncio
import logging
import time
import os
import sys
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

import numpy as np
import pandas as pd

from real_sports_feed import RealSportsFeed

# Optional external production models (if present on PYTHONPATH)
try:
    import production_nfl_model as _prod_nfl
except Exception:
    _prod_nfl = None

try:
    import production_soccer_model as _prod_soc
except Exception:
    _prod_soc = None


# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# UPDATED: Use working authentication
from kalshi_auth_fixed import get_auth, KalshiAuth

# Simple working config - no external dependencies
from types import SimpleNamespace


def get_config(config_path=None):
    return SimpleNamespace(
        paper_trading=True,                # paper mode, but still read live markets
        bankroll=1000.0,
        min_position_size=1,
        max_position_size=50,
        kelly_fraction=0.10,
        daily_loss_limit_pct=15.0,
        max_positions=20,
        edge_threshold=0.05,
        min_confidence=0.70,
        fee_buffer=0.01,
        take_profit_pct=20.0,
        stop_loss_pct=15.0,
        trailing_stop_pct=5.0,
        max_hold_hours=24,
        poll_interval_seconds=30,
        portfolio_risk_limit_pct=50.0,

        # READ real Kalshi markets even in paper mode (read-only)
        use_live_market_data_in_paper=True,
        # Place paper orders & manage PnL (set False to log-only)
        listen_only=False,
        # Optional: if trading live, use the real Kalshi balance as bankroll
        use_real_balance=False,

        # NEW toggles:
        collapse_outcomes_per_match=True,     # one signal/order per match per cycle
        only_markets_with_espn_live=False,    # only act on ESPN-live events

        # NEW execution/marking controls:
        min_hold_seconds=10,                  # don't exit before this many seconds
        skip_exit_on_untradeable=True,        # ignore 0/100 quotes for exit
        price_smoothing_alpha=0.35,           # EMA smoothing for marks (0..1), optional
    )


def setup_resilient_logging(config):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


InstitutionalConfig = SimpleNamespace

# Enhanced analytics imports
try:
    from textblob import TextBlob  # noqa: F401
    import talib  # noqa: F401
    ADVANCED_ANALYTICS_AVAILABLE = True
except ImportError:
    ADVANCED_ANALYTICS_AVAILABLE = False


# FIXED: Enhanced Sport and Ticker Parser
class Sport(Enum):
    NFL = "nfl"
    MLB = "mlb"
    NBA = "nba"
    SOCCER = "soccer"
    TENNIS = "tennis"
    MLS = "mls"
    EPL = "epl"
    WNBA = "wnba"
    UNKNOWN = "unknown"


@dataclass
class ParsedTicker:
    original: str
    sport: Sport
    team1: str
    team2: str
    market_type: str
    confidence: float
    outcome: str = ""

    def is_valid(self) -> bool:
        return (
            self.sport != Sport.UNKNOWN
            and self.team1 != "UNKNOWN"
            and self.team2 != "UNKNOWN"
            and self.confidence > 0.5
        )


class FixedKalshiTickerParser:
    """FIXED: Enhanced parser for actual Kalshi ticker formats"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.kalshi_patterns = [
            # Tennis: KXWTAMATCH-25SEP15BOIYEO-YEO/BOI
            (
                r"^KXWTAMATCH-\d{2}[A-Z]{3}\d{2}(?P<p1>[A-Z]{3})(?P<p2>[A-Z]{3})-(?P<outcome>[A-Z]{3})$",
                Sport.TENNIS,
            ),
            # EPL: KXEPLGAME-25SEP29EVEWHU-WHU/TIE/EVE
            (
                r"^KXEPLGAME-\d{2}[A-Z]{3}\d{2}(?P<team1>[A-Z]{3})(?P<team2>[A-Z]{3})-(?P<outcome>[A-Z]+)$",
                Sport.EPL,
            ),
            # MLS: KXMLSGAME-25SEP20ORLNSH-TIE/ORL/NSH
            (
                r"^KXMLSGAME-\d{2}[A-Z]{3}\d{2}(?P<team1>[A-Z]{3})(?P<team2>[A-Z]{3})-(?P<outcome>[A-Z]+)$",
                Sport.MLS,
            ),
            # Serie A: KXSERIEAGAME-25SEP22NAPPIS-TIE/PIS/NAP
            (
                r"^KXSERIEAGAME-\d{2}[A-Z]{3}\d{2}(?P<team1>[A-Z]{3})(?P<team2>[A-Z]{3})-(?P<outcome>[A-Z]+)$",
                Sport.SOCCER,
            ),
            # WNBA: KXWNBASERIES-25NYLPHX-PHX/NYL
            (
                r"^KXWNBASERIES-\d{2}(?P<team1>[A-Z]{3})(?P<team2>[A-Z]{3})-(?P<outcome>[A-Z]+)$",
                Sport.WNBA,
            ),
            # Legacy formats
            (
                r"^(?P<sport>NFL|MLB|NBA|NHL|SOCCER)[-_](?P<team1>[A-Z]{2,4})[-_](?P<team2>[A-Z]{2,4})[-_](?P<market>[A-Z0-9]+)$",
                None,
            ),
        ]

    def parse(self, ticker: str) -> ParsedTicker:
        ticker = ticker.strip().upper()
        try:
            if any(x in ticker for x in ["SPOTIFY", "REDISTRICTING", "CZECH", "PARTY"]):
                return self._create_unknown_ticker(ticker)

            for pattern, sport in self.kalshi_patterns:
                match = re.match(pattern, ticker)
                if match:
                    groups = match.groupdict()
                    if sport is None:
                        sport_str = groups.get("sport", "")
                        sport = self._parse_sport(sport_str)
                        team1 = groups.get("team1", "UNKNOWN")
                        team2 = groups.get("team2", "UNKNOWN")
                        market_type = groups.get("market", "H1")
                        outcome = ""
                    else:
                        if "p1" in groups and "p2" in groups:
                            team1 = groups["p1"]
                            team2 = groups["p2"]
                        else:
                            team1 = groups["team1"]
                            team2 = groups["team2"]
                        outcome = groups.get("outcome", "")
                        market_type = self._classify_market_type(outcome, sport)

                    return ParsedTicker(
                        original=ticker,
                        sport=sport,
                        team1=team1,
                        team2=team2,
                        market_type=market_type,
                        outcome=outcome,
                        confidence=0.95 if sport != Sport.UNKNOWN else 0.3,
                    )
            return self._create_unknown_ticker(ticker)
        except Exception as e:
            self.logger.error(f"Failed to parse ticker {ticker}: {e}")
            return self._create_unknown_ticker(ticker)

    def _classify_market_type(self, outcome: str, sport: Sport) -> str:
        outcome = outcome.upper()
        if outcome == "TIE":
            return "DRAW"
        elif len(outcome) == 3:
            return "MONEYLINE"
        else:
            return "OTHER"

    def _parse_sport(self, sport_str: str) -> Sport:
        sport_mapping = {
            "NFL": Sport.NFL,
            "MLB": Sport.MLB,
            "NBA": Sport.NBA,
            "SOCCER": Sport.SOCCER,
            "TENNIS": Sport.TENNIS,
            "MLS": Sport.MLS,
            "EPL": Sport.EPL,
        }
        return sport_mapping.get(sport_str.upper(), Sport.UNKNOWN)

    def _create_unknown_ticker(self, ticker: str) -> ParsedTicker:
        return ParsedTicker(
            original=ticker,
            sport=Sport.UNKNOWN,
            team1="UNKNOWN",
            team2="UNKNOWN",
            market_type="UNKNOWN",
            confidence=0.0,
        )


# Global parser instance
_parser = FixedKalshiTickerParser()


def parse_ticker(ticker: str) -> ParsedTicker:
    return _parser.parse(ticker)


def get_sport(ticker: str) -> Sport:
    return parse_ticker(ticker).sport


@dataclass
class Position:
    """Enhanced position tracking with institutional exit strategies"""

    ticker: str
    side: str  # 'yes' or 'no'
    quantity: int
    entry_price: float
    entry_time: datetime

    # Enhanced exit targets
    take_profit_price: float
    stop_loss_price: float
    trailing_stop_price: float
    time_exit: datetime

    # Position metadata
    edge: float
    confidence: float
    model_score: float
    institutional_score: float
    patterns_detected: int

    # Risk/marking
    max_favorable_price: float = field(default=0.0)
    unrealized_pnl: float = field(default=0.0)
    risk_adjusted_size: float = field(default=0.0)
    last_marked_price: float = field(default=0.0)   # EMA-marked price for smoothing
    last_marked_time: datetime = field(default_factory=datetime.now)

    def update_trailing_stop(self, current_price: float, trailing_pct: float) -> None:
        if self.side == "yes":
            if current_price > self.max_favorable_price:
                self.max_favorable_price = current_price
            new_trailing = current_price * (1 - trailing_pct / 100)
            self.trailing_stop_price = max(self.trailing_stop_price, new_trailing)
        else:
            # For NO, favorable is price moving down
            if (current_price < self.max_favorable_price) or (self.max_favorable_price == 0):
                self.max_favorable_price = current_price
            new_trailing = current_price * (1 + trailing_pct / 100)
            self.trailing_stop_price = min(self.trailing_stop_price, new_trailing)

    def should_exit(self, current_price: float, current_time: datetime) -> Tuple[bool, str]:
        # Take profit
        if self.side == "yes" and current_price >= self.take_profit_price:
            return True, "take_profit"
        if self.side == "no" and current_price <= self.take_profit_price:
            return True, "take_profit"

        # Stop loss
        if self.side == "yes" and current_price <= self.stop_loss_price:
            return True, "stop_loss"
        if self.side == "no" and current_price >= self.stop_loss_price:
            return True, "stop_loss"

        # Trailing stop
        if self.side == "yes" and current_price <= self.trailing_stop_price:
            return True, "trailing_stop"
        if self.side == "no" and current_price >= self.trailing_stop_price:
            return True, "trailing_stop"

        # Time exit
        if current_time >= self.time_exit:
            return True, "time_exit"

        return False, ""


class MockModel:
    """Mock model for development when real models not available"""

    def __init__(self, sport: str):
        self.sport = sport
        self.version = "mock_1.0.0"

    def predict_win_probability(self, game_data: Dict) -> Tuple[float, float]:
        base_prob = 0.5
        if "home_team" in game_data:
            base_prob += 0.03
        if "score_differential" in game_data:
            score_diff = game_data["score_differential"]
            base_prob += min(0.2, score_diff * 0.02)

        import random
        noise = random.gauss(0, 0.05)
        probability = max(0.1, min(0.9, base_prob + noise))
        confidence = random.uniform(0.7, 0.95)
        return probability, confidence


class ProductionModelWrapper:
    """Fallback wrapper around wp_models logic."""

    def __init__(self, sport: str):
        self.sport = sport
        self.version = f"{sport}_wrapper_1.0"
        self.logger = logging.getLogger(__name__)
        try:
            from wp_models import (  # noqa: F401
                nfl_win_probability,
                mlb_win_probability,
                NFLState,
                MLBState,
            )
            self.nfl_win_probability = nfl_win_probability
            self.mlb_win_probability = mlb_win_probability
            self.NFLState = NFLState
            self.MLBState = MLBState
            self.wp_models_available = True
            self.logger.info(f"Production {sport.upper()} wrapper loaded")
        except ImportError as e:
            self.wp_models_available = False
            self.logger.warning(f"Could not import wp_models: {e}")

    def predict_win_probability(self, game_data: Dict) -> Tuple[float, float]:
        if not self.wp_models_available:
            return MockModel(self.sport).predict_win_probability(game_data)
        try:
            if self.sport == "nfl":
                home_score = game_data.get("home_team", {}).get("score", 0)
                away_score = game_data.get("away_team", {}).get("score", 0)
                quarter = game_data.get("period", 1)
                nfl_state = self.NFLState(
                    points_for=home_score,
                    points_against=away_score,
                    quarter=quarter,
                    clock_seconds=900,
                    pregame_wp=0.5,
                )
                probability = self.nfl_win_probability(nfl_state)
                return probability, 0.85
            elif self.sport == "mlb":
                home_score = game_data.get("home_team", {}).get("score", 0)
                away_score = game_data.get("away_team", {}).get("score", 0)
                inning = game_data.get("period", 1)
                mlb_state = self.MLBState(
                    runs_for=home_score,
                    runs_against=away_score,
                    inning=inning,
                    top=True,
                    outs=0,
                    pregame_wp=0.5,
                )
                probability = self.mlb_win_probability(mlb_state)
                return probability, 0.85
            else:
                return MockModel(self.sport).predict_win_probability(game_data)
        except Exception as e:
            self.logger.error(f"Production wrapper failed for {self.sport}: {e}")
            return MockModel(self.sport).predict_win_probability(game_data)



class ExternalModelAdapter:

    def __init__(self, impl):
        self.impl = impl
        self.version = getattr(impl, "VERSION", getattr(impl, "__version__", "external_1.0"))

    def predict_win_probability(self, game_data: dict):
        # Support module with function, class instance, or simple predict
        # Module-level predict_win_probability
        if hasattr(self.impl, "predict_win_probability"):
            return self.impl.predict_win_probability(game_data)

        # Module-level predict
        if hasattr(self.impl, "predict"):
            return self.impl.predict(game_data)

        # Class-style API
        if hasattr(self.impl, "Model"):
            inst = getattr(self, "_instance", None)
            if inst is None:
                inst = self.impl.Model()
                setattr(self, "_instance", inst)
            if hasattr(inst, "predict_win_probability"):
                return inst.predict_win_probability(game_data)
            if hasattr(inst, "predict"):
                return inst.predict(game_data)

        raise RuntimeError("External model has no usable predict* interface")


class EnhancedModelManager:
    """Enhanced model manager with institutional-grade model routing"""

    def __init__(self, config: InstitutionalConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self._initialize_models()

    def _initialize_models(self):
        try:
            self.models["nfl"] = ProductionModelWrapper("nfl")
            self.logger.info("Production NFL model (wrapped) initialized")
            self.models["mlb"] = ProductionModelWrapper("mlb")
            self.logger.info("Production MLB model (wrapped) initialized")
            self.models["soccer"] = MockModel("soccer")
            self.logger.info("Mock Soccer model initialized")
            self.models["tennis"] = MockModel("tennis")
            self.logger.info("Mock Tennis model initialized")
            self.models["mls"] = MockModel("mls")
            self.logger.info("Mock MLS model initialized")
            self.models["epl"] = MockModel("epl")
            self.logger.info("Mock EPL model initialized")
            self.models["wnba"] = MockModel("wnba")
            self.logger.info("Mock WNBA model initialized")

            # === External production models override (patched) ===
            try:
                if '_prod_nfl' in globals() and _prod_nfl is not None:
                    self.models["nfl"] = ExternalModelAdapter(_prod_nfl)
                    self.logger.info("Production NFL model (external) initialized")

                if '_prod_soc' in globals() and _prod_soc is not None:
                    soccer_model = ExternalModelAdapter(_prod_soc)
                    # Use the same soccer model for league-specific variants
                    self.models["soccer"] = soccer_model
                    self.models["epl"] = soccer_model
                    self.models["mls"] = soccer_model
                    self.logger.info("Production Soccer model (external) initialized for soccer/epl/mls")
            except Exception as _ext_e:
                self.logger.warning(f"External model patch failed: {_ext_e}")
            # === end external override patch ===
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            for sport in ["nfl", "mlb", "soccer", "tennis", "mls", "epl", "wnba"]:
                self.models[sport] = MockModel(sport)
        self.logger.info(f"Enhanced Model Manager initialized with {len(self.models)} sport models")

    def get_prediction(self, ticker: str, game_data: Dict, sport_hint: str = None) -> Tuple[float, float, Dict]:
        try:
            parsed = parse_ticker(ticker)
            if not parsed.is_valid():
                self.logger.warning(f"Could not parse ticker: {ticker}")
                return 0.5, 0.0, {"error": "invalid_ticker"}

            sport = sport_hint or parsed.sport.value
            model = self.models.get(sport)
            if not model:
                self.logger.warning(f"No model available for sport: {sport}")
                return 0.5, 0.0, {"error": "no_model"}

            probability, confidence = model.predict_win_probability(game_data)
            metadata = {
                "sport": sport,
                "model_version": getattr(model, "version", "unknown"),
                "teams": f"{parsed.team1} vs {parsed.team2}",
                "market_type": parsed.market_type,
                "parser_confidence": parsed.confidence,
                "timestamp": datetime.now().isoformat(),
            }
            return probability, confidence, metadata
        except Exception as e:
            self.logger.error(f"Prediction failed for {ticker}: {e}")
            return 0.5, 0.0, {"error": str(e)}


class InstitutionalRiskManager:
    """FIXED: Institutional-grade risk management with micro-account support"""

    def __init__(self, config: InstitutionalConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.daily_pnl = 0.0
        self.total_exposure = 0.0
        self.position_count = 0
        self.daily_trades = 0
        self.day_start_time = datetime.now().date()
        self.drawdown_period = False
        self.consecutive_losses = 0
        self.volatility_scaling = 1.0
        self.correlation_matrix = defaultdict(float)
        self.logger.info(f"Institutional risk manager initialized: ${config.bankroll} bankroll")

    def calculate_position_size(self, edge: float, confidence: float, current_price: float, volatility: float = 0.2) -> float:
        try:
            kelly_fraction = edge / max(volatility**2, 0.01)
            confidence_adjustment = confidence**2
            vol_adjustment = max(0.5, 1.0 - volatility)
            drawdown_adjustment = 0.5 if self.drawdown_period else 1.0
            exposure_pct = self.total_exposure / max(self.config.bankroll, 1.0)
            exposure_adjustment = max(0.3, 1.0 - exposure_pct / self.config.portfolio_risk_limit_pct * 100)
            position_adjustment = max(0.5, 1.0 - self.position_count / self.config.max_positions)
            adjusted_kelly = (kelly_fraction * confidence_adjustment * vol_adjustment *
                              drawdown_adjustment * exposure_adjustment * position_adjustment)
            bounded_kelly = min(adjusted_kelly, self.config.kelly_fraction)
            position_value = bounded_kelly * self.config.bankroll
            quantity = position_value / max(current_price, 0.01)
            min_size = max(1, self.config.min_position_size)
            max_size = min(self.config.max_position_size, self.config.bankroll / current_price * 0.5)
            quantity = max(min_size, min(quantity, max_size))
            position_value = quantity * current_price
            if position_value > self.config.bankroll * 0.8:
                quantity = (self.config.bankroll * 0.8) / current_price
            self.logger.debug(
                f"Position sizing: edge={edge:.1%}, confidence={confidence:.1%}, "
                f"kelly={kelly_fraction:.3f}, adjusted={adjusted_kelly:.3f}, "
                f"quantity={quantity:.0f}, value=${position_value:.2f}"
            )
            return max(1, quantity)
        except Exception as e:
            self.logger.error(f"Position sizing failed: {e}")
            return max(1, self.config.min_position_size)

    def check_risk_limits(self, proposed_size: float, current_price: float) -> bool:
        position_value = proposed_size * current_price
        if position_value < 0.10:
            self.logger.debug("Position too small to execute")
            return False
        daily_loss_limit = self.config.daily_loss_limit_pct / 100 * self.config.bankroll
        if self.daily_pnl <= -daily_loss_limit:
            self.logger.warning("Daily loss limit reached")
            return False
        new_exposure = self.total_exposure + position_value
        exposure_limit = self.config.portfolio_risk_limit_pct / 100 * self.config.bankroll
        if new_exposure > exposure_limit:
            self.logger.warning("Portfolio exposure limit reached")
            return False
        if self.position_count >= self.config.max_positions:
            self.logger.warning("Maximum positions limit reached")
            return False
        if position_value > self.config.bankroll * 0.9:
            self.logger.warning("Insufficient balance for position")
            return False
        return True

    def update_risk_metrics(self, pnl_change: float, new_position: bool = False):
        current_date = datetime.now().date()
        if current_date != self.day_start_time:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.day_start_time = current_date
        self.daily_pnl += pnl_change
        if new_position:
            self.position_count += 1
            self.daily_trades += 1
        if self.daily_pnl < -0.05 * self.config.bankroll:
            self.drawdown_period = True
        elif self.daily_pnl > 0:
            self.drawdown_period = False


class RealKalshiTradingBot:
    """FIXED: Real Kalshi API trading bot with enhanced error handling"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        setup_resilient_logging(self.config)
        self.logger = logging.getLogger(__name__)
        self.model_manager = EnhancedModelManager(self.config)
        self.risk_manager = InstitutionalRiskManager(self.config)

        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.total_pnl = 0.0
        self.total_trades = 0
        self.wins = 0
        self.losses = 0

        self.kalshi_auth: Optional[KalshiAuth] = None
        self._initialize_kalshi_auth()

        self.logger.info("Real Kalshi trading bot initialized")
        self.logger.info(
            f"Configuration: {self.config.daily_loss_limit_pct}% daily limit, {self.config.max_positions} max positions"
        )

        self.sports_feed = RealSportsFeed()
        self.logger.info("✅ Real sports feed with ESPN integration loaded")
    # --- NEW: human-readable labels for logs ---------------------------------
    def _map_code_to_name(self, code: str) -> str:
        """Map 2-3 letter codes to nicer names when we know them; otherwise echo the code."""
        if not code:
            return ""
        try:
            return _parser.team_mappings.get(str(code).upper(), str(code).upper())
        except Exception:
            return str(code).upper()

    def _format_label(
        self,
        ticker: str,
        market: "Optional[Dict]" = None,
        side: "Optional[str]" = None,
        outcome: "Optional[str]" = None,
    ) -> str:
        """
        Build a readable label like: [TENNIS] Boulter vs Yeo — pick: Boulter
        Falls back to ticker parts if we don't have full names.
        """
        parsed = parse_ticker(ticker)
        sport = parsed.sport.value.upper() if parsed.sport != Sport.UNKNOWN else "SPORT"
        title = (market or {}).get("title") or ""

        if title:
            base = title
        else:
            t1 = self._map_code_to_name(parsed.team1)
            t2 = self._map_code_to_name(parsed.team2)
            base = f"{t1} vs {t2}"
        pick_code = (outcome or parsed.outcome or "").upper()
        pick = self._map_code_to_name(pick_code) if pick_code else ""
        return f"[{sport}] {base}{(' — pick: ' + pick) if pick else ''}"


    def _initialize_kalshi_auth(self):
        """
        Initialize Kalshi authentication.
        Even in paper mode we attempt auth so we can READ real markets (no orders placed in paper mode).
        """
        try:
            self.kalshi_auth = get_auth()
            balance_data = self.kalshi_auth.get_balance()
            balance = balance_data["balance"] / 100
            self.logger.info("Kalshi authentication successful")
            self.logger.info(f"Account balance: ${balance:.2f}")

            if self.config.paper_trading:
                self.logger.info("Paper trading mode: using REAL market data, NOT placing live orders.")
            else:
                if balance < 10.0:
                    self._adjust_for_micro_account(balance)
                elif balance < 100.0:
                    self._adjust_for_small_account(balance)
                if getattr(self.config, "use_real_balance", False):
                    self.config.bankroll = balance
                    self.logger.info(f"Updated bankroll to real balance: ${balance:.2f}")
        except Exception as e:
            if self.config.paper_trading:
                self.logger.info("Paper trading mode - no real authentication")
                self.logger.warning(f"Kalshi auth unavailable: {e}")
                self.kalshi_auth = None
            else:
                self.logger.error(f"Kalshi authentication failed (live mode): {e}")
                self.logger.info("Falling back to paper trading mode")
                self.kalshi_auth = None
                self.config.paper_trading = True

    def _adjust_for_micro_account(self, balance: float):
        self.logger.warning(f"Micro-account detected (${balance:.2f}). Adjusting settings...")
        self.config.min_position_size = 1
        self.config.max_position_size = max(2, balance * 0.20)
        self.config.kelly_fraction = 0.05
        self.config.daily_loss_limit_pct = 2.0
        self.config.max_positions = 10
        self.config.edge_threshold = 0.10
        self.config.min_confidence = 0.80
        self.logger.info("Micro-account adjustments applied:")
        self.logger.info(f" Min position: ${self.config.min_position_size}")
        self.logger.info(f" Max position: ${self.config.max_position_size:.2f}")
        self.logger.info(f" Kelly fraction: {self.config.kelly_fraction:.1%}")
        self.logger.info(f" Required edge: {self.config.edge_threshold:.1%}")

    def _adjust_for_small_account(self, balance: float):
        self.logger.warning(f"Small account detected (${balance:.2f}). Adjusting settings...")
        self.config.min_position_size = 1
        self.config.max_position_size = balance * 0.30
        self.config.kelly_fraction = 0.10
        self.config.daily_loss_limit_pct = 5.0
        self.config.edge_threshold = 0.08
        self.logger.info("Small account adjustments applied")

    # ---------- QUOTES / PRICES ----------

    def _fetch_market_quote(self, ticker: str) -> Optional[Dict[str, float]]:
        """
        Read the best bid/ask for both YES and NO from Kalshi (0..100 ints).
        Returns dict with yes_bid, yes_ask, no_bid, no_ask in 0..1 floats.
        """
        try:
            use_real = self.kalshi_auth is not None and (
                not self.config.paper_trading or self.config.use_live_market_data_in_paper
            )
            if use_real:
                data = self.kalshi_auth.get_markets(ticker=ticker)
                markets = data.get("markets", [])
                if not markets:
                    return None
                m = markets[0]
                return {
                    "yes_bid": (m.get("yes_bid", 50) / 100),
                    "yes_ask": (m.get("yes_ask", 50) / 100),
                    "no_bid": (m.get("no_bid", 50) / 100),
                    "no_ask": (m.get("no_ask", 50) / 100),
                }
            # mock
            return None
        except Exception as e:
            self.logger.error(f"Quote fetch failed for {ticker}: {e}")
            return None

    def _get_tradeable_price(self, ticker: str, side: str, action: str) -> Optional[float]:
        """
        Return the price you'd likely transact at *now*:
          - If holding YES and want to exit -> sell YES -> use yes_bid
          - If holding NO and want to exit  -> sell NO  -> use no_bid
          - For entries: buy YES -> yes_ask ; buy NO -> no_ask
        Falls back to mid if missing, ignores untradeable 0/1 when configured.
        """
        q = self._fetch_market_quote(ticker) or {}
        yes_bid = q.get("yes_bid", None)
        yes_ask = q.get("yes_ask", None)
        no_bid  = q.get("no_bid", None)
        no_ask  = q.get("no_ask", None)

        def mid(a: Optional[float], b: Optional[float]) -> Optional[float]:
            if a is None or b is None:
                return None
            return (a + b) / 2.0

        price = None
        if action == "sell":
            if side == "yes":
                price = yes_bid if yes_bid is not None else mid(yes_bid, yes_ask)
            else:
                price = no_bid if no_bid is not None else mid(no_bid, no_ask)
        else:  # action == "buy"
            if side == "yes":
                price = yes_ask if yes_ask is not None else mid(yes_bid, yes_ask)
            else:
                price = no_ask if no_ask is not None else mid(no_bid, no_ask)

        # If configured, treat hard 0/1 as untradeable (sparse book); return None
        if price is not None and getattr(self.config, "skip_exit_on_untradeable", True):
            if price <= 0.0 + 1e-9 or price >= 1.0 - 1e-9:
                return None

        return price

    # ---------- TRADING ----------

    def _execute_real_trade(self, ticker: str, decision_data: Dict) -> Optional[Position]:
        """Execute trade (live or paper). In listen-only mode, skip execution."""
        try:
            if getattr(self.config, "listen_only", False):
                self.logger.info(
                    f"LISTEN-ONLY: Skipping order for {ticker} "
                    f"{decision_data['side'].upper()} {int(decision_data['quantity'])} "
                    f"@ {decision_data['entry_price']:.2f}"
                )
                return None

            side = decision_data["side"]
            quantity = max(1, int(decision_data["quantity"]))
            entry_price = decision_data["entry_price"]

            position_value = quantity * entry_price
            if position_value > self.config.bankroll:
                self.logger.warning(
                    f"Position value ${position_value:.2f} exceeds bankroll ${self.config.bankroll:.2f}"
                )
                return None

            # Live order path (disabled in paper)
            if self.kalshi_auth and not self.config.paper_trading:
                try:
                    order_data = {
                        "ticker": ticker,
                        "type": "limit",
                        "side": side,
                        "action": "buy",
                        "count": quantity,
                        "yes_price": int(entry_price * 100) if side == "yes" else None,
                        "no_price": int(entry_price * 100) if side == "no" else None,
                        "expiration_ts": None,
                    }
                    order_response = self.kalshi_auth.place_order(order_data)
                    order_id = order_response.get("order", {}).get("order_id")
                    if order_id:
                        self.logger.info(
                            f"REAL ORDER PLACED: {order_id} - {ticker} {side.upper()} {quantity} @ {entry_price:.2f}"
                        )
                    else:
                        self.logger.error(f"Order placement failed: {order_response}")
                        return None
                except Exception as api_error:
                    if "404" in str(api_error):
                        self.logger.error(f"Market {ticker} not found or no longer available")
                    elif "insufficient" in str(api_error).lower():
                        self.logger.error("Insufficient balance for trade")
                    else:
                        self.logger.error(f"API error placing order: {api_error}")
                    return None
            else:

                # NEW: include sport + teams and pick label
                label = decision_data.get("label") or self._format_label(
                    ticker,
                    decision_data.get("market"),
                    side=side,
                    outcome=parse_ticker(ticker).outcome,
                )
                self.logger.info(
                    f"PAPER TRADE: {label} | {ticker} {side.upper()} {quantity} @ {entry_price:.2f}"
                )

            take_profit_price = (
                entry_price * (1 + self.config.take_profit_pct / 100) if side == "yes"
                else entry_price * (1 - self.config.take_profit_pct / 100)
            )
            stop_loss_price = (
                entry_price * (1 - self.config.stop_loss_pct / 100) if side == "yes"
                else entry_price * (1 + self.config.stop_loss_pct / 100)
            )
            trailing_stop_price = stop_loss_price
            time_exit = datetime.now() + timedelta(hours=getattr(self.config, "max_hold_hours", 24))

            parsed = parse_ticker(ticker)
            if parsed.is_valid():
                if parsed.sport == Sport.MLB:
                    time_exit = min(time_exit, datetime.now() + timedelta(hours=3))
                elif parsed.sport == Sport.NFL:
                    time_exit = min(time_exit, datetime.now() + timedelta(hours=4))

            position = Position(
                ticker=ticker,
                side=side,
                quantity=quantity,
                entry_price=entry_price,
                entry_time=datetime.now(),
                take_profit_price=take_profit_price,
                stop_loss_price=stop_loss_price,
                trailing_stop_price=trailing_stop_price,
                time_exit=time_exit,
                edge=decision_data["edge"],
                confidence=decision_data["confidence"],
                model_score=decision_data["model_prob"],
                institutional_score=decision_data.get("institutional_score", decision_data["edge"] * decision_data["confidence"]),
                patterns_detected=decision_data.get("patterns", 0),
                last_marked_price=entry_price,
            )

            self.positions[ticker] = position
            self.risk_manager.total_exposure += position_value
            self.risk_manager.update_risk_metrics(0, new_position=True)

            self.logger.info(f"Position added: {ticker} {side} {quantity} @ {entry_price:.2f}")
            # NEW: show game label on executed printout
            label = decision_data.get("label") or self._format_label(
                ticker, decision_data.get("market"), side=side, outcome=parse_ticker(ticker).outcome
            )
            self.logger.info(f" Game: {label}")
            self.logger.info(f"Exit targets: TP={take_profit_price:.2f}, Stop={stop_loss_price:.2f}, Time={time_exit}")
            return position
        except Exception as e:
            self.logger.error(f"Trade execution failed for {ticker}: {e}")
            return None

    def _should_take_position(self, ticker: str, market_data: Dict, game_data: Dict) -> Tuple[bool, Dict]:
        """Enhanced position decision with validation"""
        try:
            parsed = parse_ticker(ticker)
            if not parsed.is_valid():
                return False, {"reason": "invalid_ticker", "confidence": parsed.confidence}
            if parsed.confidence < 0.7:
                return False, {"reason": "low_parsing_confidence", "confidence": parsed.confidence}

            model_prob, confidence, metadata = self.model_manager.get_prediction(
                ticker, game_data, sport_hint=parsed.sport.value
            )

            yes_price = market_data.get("yes_ask", 50) / 100
            no_price = market_data.get("no_ask", 50) / 100

            if (yes_price <= 0.01 or yes_price >= 0.99 or no_price <= 0.01 or no_price >= 0.99):
                return False, {"reason": "unrealistic_prices", "yes_price": yes_price, "no_price": no_price}

            yes_edge = model_prob - yes_price - self.config.fee_buffer
            no_edge = (1 - model_prob) - no_price - self.config.fee_buffer

            max_edge = max(yes_edge, no_edge)
            if max_edge < self.config.edge_threshold:
                return False, {"reason": "insufficient_edge", "max_edge": max_edge}
            if confidence < self.config.min_confidence:
                return False, {"reason": "low_confidence", "confidence": confidence}

            side = "yes" if yes_edge > no_edge else "no"
            edge = yes_edge if side == "yes" else no_edge
            entry_price = yes_price if side == "yes" else no_price

            volatility = market_data.get("volatility", 0.2)
            quantity = self.risk_manager.calculate_position_size(edge, confidence, entry_price, volatility)

            # Allow small paper trades; align with risk check threshold (0.10)
            position_value = quantity * entry_price
            if position_value < 0.10:
                return False, {"reason": "position_too_small", "value": position_value}
            if position_value > self.config.bankroll * 0.8:
                return False, {"reason": "position_too_large", "value": position_value}

            if not self.risk_manager.check_risk_limits(quantity, entry_price):
                return False, {"reason": "risk_limits"}

            decision_data = {
                "side": side,
                "edge": edge,
                "confidence": confidence,
                "quantity": quantity,
                "entry_price": entry_price,
                "model_prob": model_prob,
                "institutional_score": edge * confidence,
                "patterns": 1,
                "metadata": metadata,
                "parsed_info": {
                    "sport": parsed.sport.value,
                    "team1": parsed.team1,
                    "team2": parsed.team2,
                    "parsing_confidence": parsed.confidence,
                },
            }
            # NEW: pretty label for logs + keep market snapshot for context
            decision_data["label"] = self._format_label(
                ticker, market_data, side=side, outcome=parsed.outcome
            )
            decision_data["market"] = market_data

            return True, decision_data
        except Exception as e:
            self.logger.error(f"Position decision failed for {ticker}: {e}")
            return False, {"reason": "error", "error": str(e)}

    # ---------- DATA HELPERS ----------

    async def get_enhanced_market_data(self, market):
        """Get enhanced market data with ESPN integration if available"""
        ticker = market.get("ticker")
        try:
            game_state = await self.sports_feed.get_game_state(ticker)
        except Exception:
            game_state = None

        if game_state:
            self.logger.info(
                f"📊 ESPN: {game_state['away_team']['name']} {game_state['away_team']['score']}-"
                f"{game_state['home_team']['score']} {game_state['home_team']['name']}"
            )
            return game_state
        else:
            return {
                "ticker": ticker,
                "yes_bid": market.get("yes_bid", 50),
                "no_bid": market.get("no_bid", 50),
            }

    def _base_match_id(self, ticker: str) -> str:
        """Collapse multiple outcomes for same match to one best-side per cycle."""
        if "-" in ticker:
            return ticker.rsplit("-", 1)[0]
        return ticker

    # ---------- POSITION LIFECYCLE ----------

    def _ema_mark(self, position: Position, new_price: float) -> float:
        """Apply simple EMA smoothing to avoid 0/100 quote whipsaws on marks."""
        alpha = float(getattr(self.config, "price_smoothing_alpha", 0.0))
        if alpha <= 0.0 or position.last_marked_price == 0.0:
            mark = new_price
        else:
            mark = alpha * new_price + (1.0 - alpha) * position.last_marked_price
        position.last_marked_price = mark
        position.last_marked_time = datetime.now()
        return mark

    def _manage_positions(self):
        """Manage existing positions with institutional exit logic"""
        positions_to_close = []
        now = datetime.now()
        min_hold = int(getattr(self.config, "min_hold_seconds", 0))

        for ticker, position in list(self.positions.items()):
            try:
                # Use tradeable exit price (sell to bid)
                exit_px = self._get_tradeable_price(ticker, position.side, action="sell")

                # If quote is untradeable and we're configured to skip, just carry the position
                if exit_px is None and getattr(self.config, "skip_exit_on_untradeable", True):
                    self.logger.debug(f"Untradeable/empty exit quote for {ticker}, skipping exit checks this cycle.")
                    continue

                # Fallback: if no quote at all, skip
                if exit_px is None:
                    continue

                # Smooth mark
                current_price = self._ema_mark(position, exit_px)

                # Update trailing stop
                trailing_pct = float(getattr(self.config, "trailing_stop_pct", 5.0))
                position.update_trailing_stop(current_price, trailing_pct)

                # Respect minimum hold time
                if (now - position.entry_time).total_seconds() < min_hold:
                    continue

                # Check exit conditions using tradeable price
                should_exit, exit_reason = position.should_exit(current_price, now)
                if should_exit:
                    positions_to_close.append((ticker, position, current_price, exit_reason))
            except Exception as e:
                self.logger.error(f"Position management failed for {ticker}: {e}")

        # Close positions
        for ticker, position, exit_price, exit_reason in positions_to_close:
            self._close_position(ticker, position, exit_price, exit_reason)

    def _close_position(self, ticker: str, position: Position, exit_price: float, reason: str):
        """Close position and calculate PnL at the *sell* price we can actually hit."""
        try:
            if position.side == "yes":
                pnl = position.quantity * (exit_price - position.entry_price)
            else:
                pnl = position.quantity * (position.entry_price - exit_price)

            self.total_pnl += pnl
            self.total_trades += 1
            if pnl > 0:
                self.wins += 1
            else:
                self.losses += 1

            position_value = position.quantity * position.entry_price
            self.risk_manager.total_exposure -= position_value
            self.risk_manager.update_risk_metrics(pnl)

            try:
                label = self._format_label(ticker)
            except Exception:
                label = ticker
            self.logger.info(f"POSITION CLOSED: {label} - {reason.upper()}")
            self.logger.info(f"Exit: {position.quantity} @ {exit_price:.2f}, PnL: ${pnl:.2f}")

            self.closed_positions.append(position)
            del self.positions[ticker]
        except Exception as e:
            self.logger.error(f"Position closure failed for {ticker}: {e}")

    def _get_game_data(self, ticker: str) -> Dict:
        """Get live game data for ticker (placeholder integrates with sports feed)"""
        try:
            parsed = parse_ticker(ticker)
            if not parsed.is_valid():
                return {}
            return {
                "home_team": {"abbreviation": parsed.team1, "score": 0},
                "away_team": {"abbreviation": parsed.team2, "score": 0},
                "period": 1,
                "status": "in_progress",
            }
        except Exception as e:
            self.logger.error(f"Failed to get game data for {ticker}: {e}")
            return {}

    # ---------- SESSION LOOP ----------

    async def run_trading_session(self, duration_minutes: int = 60):
        """Run trading session with enhanced error handling"""
        try:
            self.logger.info(f"Starting {duration_minutes}-minute REAL trading session")
            start_time = time.time()
            end_time = start_time + (duration_minutes * 60)
            poll_interval = getattr(self.config, "poll_interval_seconds", 30)

            while time.time() < end_time:
                try:
                    markets = self._get_live_markets()
                    self.logger.info(f"Analyzing {len(markets)} markets...")

                    # Optional: filter to ESPN live only
                    if getattr(self.config, "only_markets_with_espn_live", False):
                        filtered_markets = []
                        for m in markets:
                            t = m.get("ticker", "")
                            try:
                                gs = await self.sports_feed.get_game_state(t)
                            except Exception:
                                gs = None
                            state = (gs or {}).get("game_state", "")
                            if str(state).lower() in {"live", "in_progress", "in progress"}:
                                filtered_markets.append(m)
                        markets = filtered_markets
                        self.logger.info(f"After ESPN-live filter: {len(markets)} markets")

                    # Optional: collapse multiple outcomes per match to one best-side
                    chosen_decisions: Dict[str, Dict] = {}
                    if getattr(self.config, "collapse_outcomes_per_match", False):
                        best_by_match: Dict[str, Tuple[Dict, Dict]] = {}
                        for m in markets:
                            t = m.get("ticker", "")
                            if not t:
                                continue
                            if t in self.positions:
                                continue
                            game_data = self._get_game_data(t)
                            should_trade, decision = self._should_take_position(t, m, game_data)
                            if should_trade:
                                base_id = self._base_match_id(t)
                                prev = best_by_match.get(base_id)
                                if (prev is None) or (decision["edge"] > prev[1]["edge"]):
                                    best_by_match[base_id] = (m, decision)
                        markets = [{"ticker": mk.get("ticker", ""), **mk} for mk, _ in best_by_match.values()]
                        chosen_decisions = {mk.get("ticker", ""): dec for mk, dec in best_by_match.values()}

                    opportunities_found = 0
                    for market in markets:
                        ticker = market.get("ticker", "")
                        if not ticker:
                            continue
                        if ticker in self.positions:
                            continue

                        game_data = self._get_game_data(ticker)

                        if ticker in chosen_decisions:
                            should_trade, decision_data = True, chosen_decisions[ticker]
                        else:
                            should_trade, decision_data = self._should_take_position(ticker, market, game_data)

                        if getattr(self.config, "listen_only", False):
                            if should_trade:
                                self.logger.info(
                                    f"LISTEN-ONLY SIGNAL: {ticker} side={decision_data['side'].upper()} "
                                    f"edge={decision_data['edge']:.1%} conf={decision_data['confidence']:.1%} "
                                    f"price={decision_data['entry_price']:.3f}"
                                )
                            continue

                        if should_trade:
                            opportunities_found += 1
                            position = self._execute_real_trade(ticker, decision_data)
                            if position:
                                self.logger.info("TRADE EXECUTED:")
                                self.logger.info(f" Market: {ticker}")
                                self.logger.info(f" Side: {decision_data['side'].upper()}")
                                self.logger.info(f" Quantity: {int(decision_data['quantity'])}")
                                self.logger.info(f" Price: {decision_data['entry_price']:.3f}")
                                self.logger.info(f" Edge: {decision_data['edge']:.1%}")
                                self.logger.info(f" Confidence: {decision_data['confidence']:.1%}")
                                if self.config.bankroll < 10.0:
                                    self.logger.info("Micro-account: limiting to one trade per cycle")
                                    break
                        else:
                            reason = decision_data.get("reason", "unknown")
                            if reason not in ["insufficient_edge", "low_confidence"]:
                                self.logger.debug(f"Skipped {ticker}: {reason}")

                    if opportunities_found == 0 and not getattr(self.config, "listen_only", False):
                        self.logger.info("No trading opportunities found this cycle")

                    if self.positions:
                        self.logger.info(f"Managing {len(self.positions)} open positions...")
                        self._manage_positions()

                    await asyncio.sleep(poll_interval)
                except Exception as e:
                    self.logger.error(f"Trading loop error: {e}")
                    await asyncio.sleep(10)

            self._generate_session_summary(duration_minutes)
        except Exception as e:
            self.logger.error(f"Trading session failed: {e}")

    # ---------- SUMMARY ----------

    def _generate_session_summary(self, duration_minutes: int):
        self.logger.info("=" * 80)
        self.logger.info("KALSHI TRADING SESSION COMPLETE")
        self.logger.info("=" * 80)
        win_rate = (self.wins / max(1, self.total_trades)) * 100
        avg_pnl_per_trade = self.total_pnl / max(1, self.total_trades)
        self.logger.info("SESSION PERFORMANCE:")
        self.logger.info(f" Duration: {duration_minutes} minutes")
        self.logger.info(f" Total P&L: ${self.total_pnl:.2f}")
        self.logger.info(f" Total Trades: {self.total_trades}")
        self.logger.info(f" Win Rate: {win_rate:.1f}% ({self.wins}W/{self.losses}L)")
        self.logger.info(f" Avg P&L per Trade: ${avg_pnl_per_trade:.2f}")

        rs = self.risk_manager
        self.logger.info("RISK MANAGEMENT:")
        self.logger.info(f" Daily P&L: ${rs.daily_pnl:.2f} ({(rs.daily_pnl / max(self.config.bankroll, 1.0)) * 100:+.1f}%)")
        self.logger.info(f" Portfolio Exposure: ${rs.total_exposure:.2f} ({(rs.total_exposure / max(self.config.bankroll, 1.0)) * 100:.1f}%)")
        self.logger.info(f" Active Positions: {len(self.positions)}")
        self.logger.info(f" Daily Trades: {rs.daily_trades}")

        if self.positions:
            self.logger.info("ACTIVE POSITIONS:")
            for ticker, pos in self.positions.items():
                self.logger.info(f" {ticker}: {pos.side.upper()} {pos.quantity} @ {pos.entry_price:.3f}")

        auth_status = (
            "LIVE API"
            if (self.kalshi_auth and not self.config.paper_trading)
            else ("PAPER (Live market data)" if (self.kalshi_auth and self.config.use_live_market_data_in_paper) else "PAPER (Mock markets)")
        )
        self.logger.info("SYSTEM STATUS:")
        self.logger.info(f" Authentication: {auth_status}")
        self.logger.info(f" Models Available: {len(self.model_manager.models)}")
        self.logger.info(f" Bankroll: ${self.config.bankroll:.2f}")

        if self.kalshi_auth and not self.config.paper_trading:
            try:
                balance_data = self.kalshi_auth.get_balance()
                balance = balance_data["balance"] / 100
                self.logger.info(f" Current Balance: ${balance:.2f}")
            except Exception:
                pass

        self.logger.info("=" * 80)

    # ---------- MARKETS FETCH (last so helper funcs are defined) ----------

    def _get_live_markets(self) -> List[Dict]:
        """Get live markets with proper filtering"""
        try:
            use_real_markets = self.kalshi_auth is not None and (
                not self.config.paper_trading or self.config.use_live_market_data_in_paper
            )

            if use_real_markets:
                markets_data = self.kalshi_auth.get_markets(status="open")
                markets = markets_data.get("markets", [])
                sports_markets = []
                for market in markets:
                    ticker = market.get("ticker", "")
                    parsed = parse_ticker(ticker)
                    if (
                        parsed.is_valid()
                        and parsed.sport in [Sport.NFL, Sport.MLB, Sport.NBA, Sport.TENNIS, Sport.MLS, Sport.EPL, Sport.SOCCER, Sport.WNBA]
                        and parsed.confidence > 0.8
                    ):
                        sports_markets.append(
                            {
                                "ticker": ticker,
                                "title": market.get("title", ""),
                                "yes_bid": market.get("yes_bid", 50),
                                "yes_ask": market.get("yes_ask", 50),
                                "no_bid": market.get("no_bid", 50),
                                "no_ask": market.get("no_ask", 50),
                                "status": market.get("status", "open"),
                                "volume": market.get("volume", 0),
                                "open_interest": market.get("open_interest", 0),
                                "depth": market.get("volume", 100),
                                "parsed_sport": parsed.sport.value,
                                "parsed_team1": parsed.team1,
                                "parsed_team2": parsed.team2,
                                "parsing_confidence": parsed.confidence,
                            }
                        )
                self.logger.info(f"Retrieved {len(sports_markets)} valid sports markets from Kalshi API")
                if not sports_markets:
                    self.logger.warning("No valid sports markets found, using mock data")
                    return self._get_mock_markets()
                return sports_markets
            else:
                return self._get_mock_markets()
        except Exception as e:
            self.logger.error(f"Failed to get live markets: {e}")
            self.logger.info("Falling back to mock markets")
            return self._get_mock_markets()

    def _get_mock_markets(self) -> List[Dict]:
        return [
            {
                "ticker": "NFL-KC-BUF-H1",
                "title": "Will Chiefs beat Bills?",
                "yes_bid": 45,
                "yes_ask": 47,
                "no_bid": 53,
                "no_ask": 55,
                "status": "open",
                "depth": 100,
                "volume": 1500,
                "open_interest": 800,
                "parsed_sport": "nfl",
                "parsed_team1": "KC",
                "parsed_team2": "BUF",
                "parsing_confidence": 0.95,
            },
            {
                "ticker": "MLB-NYY-BOS-H1",
                "title": "Will Yankees beat Red Sox?",
                "yes_bid": 58,
                "yes_ask": 60,
                "no_bid": 40,
                "no_ask": 42,
                "status": "open",
                "depth": 150,
                "volume": 1200,
                "open_interest": 600,
                "parsed_sport": "mlb",
                "parsed_team1": "NYY",
                "parsed_team2": "BOS",
                "parsing_confidence": 0.95,
            },
        ]


# ---------- MAIN ----------

async def main():
    """Main entry point"""
    print("REAL KALSHI API TRADING BOT - FIXED VERSION")
    print("Enhanced ticker parsing and micro-account support")
    cfg = get_config()
    print(
        f"Listen-only: {'ON' if cfg.listen_only else 'OFF'} | "
        f"Paper live data: {'ON' if cfg.use_live_market_data_in_paper else 'OFF'} | "
        f"Collapse outcomes per match: {'ON' if cfg.collapse_outcomes_per_match else 'OFF'} | "
        f"ESPN-live-only: {'ON' if cfg.only_markets_with_espn_live else 'OFF'}"
    )
    print("=" * 80)

    try:
        bot = RealKalshiTradingBot()
        try:
            duration_input = input("Enter trading duration in minutes (default 60): ").strip()
            duration = int(duration_input) if duration_input else 60
        except ValueError:
            duration = 60

        print(f"Starting {duration}-minute trading session...")
        if bot.config.listen_only:
            print("NOTE: Listen-only mode is ON -> logging signals, not placing orders.")
        print("WARNING: This will place actual trades if not in paper trading mode and not in listen-only!")

        if not bot.config.paper_trading and bot.kalshi_auth and not bot.config.listen_only:
            confirm = input("REAL MONEY TRADING ENABLED. Continue? (yes/no): ").strip().lower()
            if confirm != "yes":
                print("Trading cancelled by user")
                return

        await bot.run_trading_session(duration)
    except KeyboardInterrupt:
        print("\nTrading session interrupted by user")
    except Exception as e:
        print(f"\nTrading session failed: {e}")
        logging.error(f"Main execution failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
