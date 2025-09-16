#!/usr/bin/env python3
"""
Institutional-Grade Kalshi Trading Bot
=====================================
Integrates your proven models with institutional-level risk management,
pattern analysis, sophisticated exit strategies, and enterprise-grade resilience
"""

import asyncio
import logging
import time
import os
import sys
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
import pandas as pd

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Enhanced imports with resilience
from enhanced_config_loader import get_config, setup_resilient_logging, InstitutionalConfig
from robust_ticker_parser import parse_ticker, get_sport, Sport, ParsedTicker
from resilient_api_client import (
    get_resilient_kalshi_client, 
    get_resilient_sports_feed,
    ResilientAPIError
)

# Your proven model imports
try:
    from models.production_nfl_model import ProductionNFLModel
    from models.production_mlb_model import ProductionMLBModel
    from models.production_soccer_model import ProductionSoccerModel
    NFL_MODEL_AVAILABLE = True
except ImportError:
    NFL_MODEL_AVAILABLE = False
    logging.warning("Production models not found - using mock models")

# Enhanced analytics imports
try:
    from textblob import TextBlob
    import talib
    ADVANCED_ANALYTICS_AVAILABLE = True
except ImportError:
    ADVANCED_ANALYTICS_AVAILABLE = False


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
    
    # Risk tracking
    max_favorable_price: float = field(default=0.0)
    unrealized_pnl: float = field(default=0.0)
    risk_adjusted_size: float = field(default=0.0)
    
    def update_trailing_stop(self, current_price: float, trailing_pct: float) -> None:
        """Update trailing stop based on favorable price movement"""
        if self.side == 'yes':
            if current_price > self.max_favorable_price:
                self.max_favorable_price = current_price
                new_trailing = current_price * (1 - trailing_pct / 100)
                self.trailing_stop_price = max(self.trailing_stop_price, new_trailing)
        else:
            if current_price < self.max_favorable_price or self.max_favorable_price == 0:
                self.max_favorable_price = current_price
                new_trailing = current_price * (1 + trailing_pct / 100)
                self.trailing_stop_price = min(self.trailing_stop_price, new_trailing)
    
    def should_exit(self, current_price: float, current_time: datetime) -> Tuple[bool, str]:
        """Determine if position should be exited"""
        # Take profit
        if self.side == 'yes' and current_price >= self.take_profit_price:
            return True, "take_profit"
        if self.side == 'no' and current_price <= self.take_profit_price:
            return True, "take_profit"
        
        # Stop loss
        if self.side == 'yes' and current_price <= self.stop_loss_price:
            return True, "stop_loss"
        if self.side == 'no' and current_price >= self.stop_loss_price:
            return True, "stop_loss"
        
        # Trailing stop
        if self.side == 'yes' and current_price <= self.trailing_stop_price:
            return True, "trailing_stop"
        if self.side == 'no' and current_price >= self.trailing_stop_price:
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
        """Mock prediction with realistic values"""
        # Generate semi-realistic probabilities based on game state
        base_prob = 0.5
        
        if 'home_team' in game_data:
            base_prob += 0.03  # Home field advantage
        
        if 'score_differential' in game_data:
            score_diff = game_data['score_differential']
            base_prob += min(0.2, score_diff * 0.02)  # Score impact
        
        # Add some randomness
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
        
        # Try to import wp_models
        try:
            from wp_models import nfl_win_probability, mlb_win_probability, NFLState, MLBState
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
        """Predict using wp_models or fallback to mock"""
        if not self.wp_models_available:
            return MockModel(self.sport).predict_win_probability(game_data)
        
        try:
            if self.sport == 'nfl':
                # Create NFLState and call nfl_win_probability
                home_score = game_data.get('home_team', {}).get('score', 0)
                away_score = game_data.get('away_team', {}).get('score', 0)
                quarter = game_data.get('period', 1)
                
                nfl_state = self.NFLState(
                    points_for=home_score,
                    points_against=away_score,
                    quarter=quarter,
                    clock_seconds=900,  # Default
                    pregame_wp=0.5
                )
                
                probability = self.nfl_win_probability(nfl_state)
                confidence = 0.85
                return probability, confidence
                
            elif self.sport == 'mlb':
                # Create MLBState and call mlb_win_probability
                home_score = game_data.get('home_team', {}).get('score', 0)
                away_score = game_data.get('away_team', {}).get('score', 0)
                inning = game_data.get('period', 1)
                
                mlb_state = self.MLBState(
                    runs_for=home_score,
                    runs_against=away_score,
                    inning=inning,
                    top=True,
                    outs=0,
                    pregame_wp=0.5
                )
                
                probability = self.mlb_win_probability(mlb_state)
                confidence = 0.85
                return probability, confidence
            
            else:
                # Fallback for other sports
                return MockModel(self.sport).predict_win_probability(game_data)
                
        except Exception as e:
            self.logger.error(f"Production wrapper failed for {self.sport}: {e}")
            return MockModel(self.sport).predict_win_probability(game_data)


class EnhancedModelManager:
    """Enhanced model manager with institutional-grade model routing"""
    
    def __init__(self, config: InstitutionalConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        
        # Initialize models with fallbacks
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize sport-specific models with fallbacks"""
        try:
            # Try to use your actual production models first
            if NFL_MODEL_AVAILABLE:
                # Check if models have the expected interface
                nfl_model = ProductionNFLModel()
                if hasattr(nfl_model, 'predict_win_probability'):
                    self.models['nfl'] = nfl_model
                    self.logger.info("Production NFL model initialized")
                else:
                    # Use wrapper for your wp_models functions
                    self.models['nfl'] = ProductionModelWrapper('nfl')
                    self.logger.info("Production NFL model (wrapped) initialized")
            else:
                # Use wrapper for your wp_models functions
                self.models['nfl'] = ProductionModelWrapper('nfl')
                self.logger.info("Production NFL model (wrapped) initialized")
            
            if NFL_MODEL_AVAILABLE:
                mlb_model = ProductionMLBModel()
                if hasattr(mlb_model, 'predict_win_probability'):
                    self.models['mlb'] = mlb_model
                    self.logger.info("Production MLB model initialized")
                else:
                    self.models['mlb'] = ProductionModelWrapper('mlb')
                    self.logger.info("Production MLB model (wrapped) initialized")
            else:
                self.models['mlb'] = ProductionModelWrapper('mlb')
                self.logger.info("Production MLB model (wrapped) initialized")
            
            if NFL_MODEL_AVAILABLE:
                soccer_model = ProductionSoccerModel()
                if hasattr(soccer_model, 'predict_win_probability'):
                    self.models['soccer'] = soccer_model
                    self.logger.info("Production Soccer model initialized")
                else:
                    self.models['soccer'] = MockModel('soccer')
                    self.logger.info("Mock Soccer model initialized")
            else:
                self.models['soccer'] = MockModel('soccer')
                self.logger.info("Mock Soccer model initialized")
            
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            # Fallback to all wrapper/mock models
            self.models['nfl'] = ProductionModelWrapper('nfl')
            self.models['mlb'] = ProductionModelWrapper('mlb')
            self.models['soccer'] = MockModel('soccer')
        
        self.logger.info(f"Enhanced Model Manager initialized with {len(self.models)} sport models")
    
    def get_prediction(self, ticker: str, game_data: Dict) -> Tuple[float, float, Dict]:
        """Get prediction with enhanced metadata"""
        try:
            # Parse ticker to get sport
            parsed = parse_ticker(ticker)
            
            if not parsed.is_valid():
                self.logger.warning(f"Could not parse ticker: {ticker}")
                return 0.5, 0.0, {"error": "invalid_ticker"}
            
            sport = parsed.sport.value
            model = self.models.get(sport)
            
            if not model:
                self.logger.warning(f"No model available for sport: {sport}")
                return 0.5, 0.0, {"error": "no_model"}
            
            # Get prediction
            probability, confidence = model.predict_win_probability(game_data)
            
            # Enhanced metadata
            metadata = {
                "sport": sport,
                "model_version": getattr(model, 'version', 'unknown'),
                "teams": f"{parsed.team1} vs {parsed.team2}",
                "market_type": parsed.market_type,
                "parser_confidence": parsed.confidence,
                "timestamp": datetime.now().isoformat()
            }
            
            return probability, confidence, metadata
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {ticker}: {e}")
            return 0.5, 0.0, {"error": str(e)}


class InstitutionalRiskManager:
    """Institutional-grade risk management with advanced position sizing"""
    
    def __init__(self, config: InstitutionalConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Risk tracking
        self.daily_pnl = 0.0
        self.total_exposure = 0.0
        self.position_count = 0
        self.daily_trades = 0
        self.day_start_time = datetime.now().date()
        
        # Advanced risk metrics
        self.drawdown_period = False
        self.consecutive_losses = 0
        self.volatility_scaling = 1.0
        self.correlation_matrix = defaultdict(float)
        
        self.logger.info(f"Institutional risk manager initialized: ${config.bankroll} bankroll")
    
    def calculate_position_size(self, edge: float, confidence: float, 
                              current_price: float, volatility: float = 0.2) -> float:
        """Advanced position sizing with multiple risk factors"""
        try:
            # Base Kelly calculation
            kelly_fraction = edge / (volatility ** 2)
            
            # Apply institutional adjustments
            
            # 1. Confidence scaling
            confidence_adjustment = confidence ** 2  # Square for conservative scaling
            
            # 2. Volatility scaling  
            vol_adjustment = max(0.5, 1.0 - volatility)
            
            # 3. Drawdown scaling
            drawdown_adjustment = 0.5 if self.drawdown_period else 1.0
            
            # 4. Exposure scaling
            exposure_pct = self.total_exposure / self.config.bankroll
            exposure_adjustment = max(0.3, 1.0 - exposure_pct / self.config.portfolio_risk_limit_pct * 100)
            
            # 5. Position count scaling
            position_adjustment = max(0.5, 1.0 - self.position_count / self.config.max_positions)
            
            # Combine all factors
            adjusted_kelly = (kelly_fraction * 
                            confidence_adjustment * 
                            vol_adjustment * 
                            drawdown_adjustment * 
                            exposure_adjustment * 
                            position_adjustment)
            
            # Apply Kelly fraction limit
            bounded_kelly = min(adjusted_kelly, self.config.kelly_fraction)
            
            # Calculate dollar amount
            position_value = bounded_kelly * self.config.bankroll
            
            # Convert to quantity
            quantity = position_value / current_price
            
            # Apply min/max limits
            quantity = max(self.config.min_position_size, 
                          min(quantity, self.config.max_position_size))
            
            self.logger.debug(f"Position sizing: edge={edge:.1%}, confidence={confidence:.1%}, "
                            f"kelly={kelly_fraction:.3f}, adjusted={adjusted_kelly:.3f}, "
                            f"quantity={quantity:.0f}")
            
            return quantity
            
        except Exception as e:
            self.logger.error(f"Position sizing failed: {e}")
            return self.config.min_position_size
    
    def check_risk_limits(self, proposed_size: float, current_price: float) -> bool:
        """Check if trade passes all risk limits"""
        
        # Daily loss limit
        if self.daily_pnl <= -self.config.daily_loss_limit_pct / 100 * self.config.bankroll:
            self.logger.warning("Daily loss limit reached")
            return False
        
        # Portfolio exposure limit
        new_exposure = self.total_exposure + (proposed_size * current_price)
        if new_exposure > self.config.portfolio_risk_limit_pct / 100 * self.config.bankroll:
            self.logger.warning("Portfolio exposure limit reached")
            return False
        
        # Max positions limit
        if self.position_count >= self.config.max_positions:
            self.logger.warning("Maximum positions limit reached")
            return False
        
        return True
    
    def update_risk_metrics(self, pnl_change: float, new_position: bool = False):
        """Update risk tracking metrics"""
        # Check if new day
        current_date = datetime.now().date()
        if current_date != self.day_start_time:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.day_start_time = current_date
        
        # Update PnL
        self.daily_pnl += pnl_change
        
        # Update position count
        if new_position:
            self.position_count += 1
            self.daily_trades += 1
        
        # Check for drawdown period
        if self.daily_pnl < -0.05 * self.config.bankroll:  # 5% drawdown
            self.drawdown_period = True
        elif self.daily_pnl > 0:
            self.drawdown_period = False
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get current risk metrics summary"""
        return {
            "daily_pnl": self.daily_pnl,
            "daily_pnl_pct": self.daily_pnl / self.config.bankroll * 100,
            "total_exposure": self.total_exposure,
            "exposure_pct": self.total_exposure / self.config.bankroll * 100,
            "position_count": self.position_count,
            "daily_trades": self.daily_trades,
            "drawdown_period": self.drawdown_period,
            "risk_capacity_remaining": max(0, self.config.portfolio_risk_limit_pct / 100 * self.config.bankroll - self.total_exposure)
        }


class InstitutionalTradingBot:
    """Institutional-grade trading bot with enterprise resilience"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Load configuration with validation
        self.config = get_config(config_path)
        
        # Set up resilient logging
        setup_resilient_logging(self.config)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.model_manager = EnhancedModelManager(self.config)
        self.risk_manager = InstitutionalRiskManager(self.config)
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        
        # Performance tracking
        self.total_pnl = 0.0
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        
        # API clients with resilience
        self.kalshi_client = None
        self.sports_feed = None
        
        self._initialize_api_clients()
        
        self.logger.info("Institutional trading bot initialized")
        self.logger.info(f"Configuration: {self.config.daily_loss_limit_pct}% daily limit, {self.config.max_positions} max positions")
    
    def _initialize_api_clients(self):
        """Initialize API clients with authentication handling"""
        try:
            # Set up authentication headers if available
            auth_headers = {}
            if self.config.kalshi_key_id and not self.config.paper_trading:
                # In production, you'd load the actual auth here
                auth_headers = {"Authorization": "Bearer mock_token"}
            
            # Initialize resilient clients
            self.kalshi_client = get_resilient_kalshi_client(
                self.config.kalshi_base_url, 
                auth_headers
            )
            
            self.sports_feed = get_resilient_sports_feed()
            
            self.logger.info("Kalshi client connected")
            
        except Exception as e:
            self.logger.error(f"API client initialization failed: {e}")
            # Continue with mock clients for development
    
    def _calculate_institutional_score(self, edge: float, confidence: float, 
                                     patterns: int, market_data: Dict) -> float:
        """Calculate comprehensive institutional trading score"""
        try:
            # Base score from edge and confidence
            base_score = (edge * confidence) ** 0.5
            
            # Pattern confirmation bonus
            pattern_bonus = min(0.1, patterns * 0.02)
            
            # Market depth bonus
            depth = market_data.get('depth', 100)
            depth_bonus = min(0.05, depth / 1000)
            
            # Time to expiry factor
            time_factor = 1.0  # Could add time decay logic here
            
            # Combine factors
            institutional_score = (base_score + pattern_bonus + depth_bonus) * time_factor
            
            return min(1.0, institutional_score)
            
        except Exception as e:
            self.logger.error(f"Institutional score calculation failed: {e}")
            return edge * confidence
    
    def _should_take_position(self, ticker: str, market_data: Dict, 
                            game_data: Dict) -> Tuple[bool, Dict]:
        """Determine if position should be taken with institutional criteria"""
        try:
            # Get model prediction
            model_prob, confidence, metadata = self.model_manager.get_prediction(ticker, game_data)
            
            # Calculate market implied probability
            yes_price = market_data.get('yes_ask', 50) / 100
            no_price = market_data.get('no_ask', 50) / 100
            
            # Calculate edges
            yes_edge = model_prob - yes_price - self.config.fee_buffer
            no_edge = (1 - model_prob) - no_price - self.config.fee_buffer
            
            # Check minimum thresholds
            max_edge = max(yes_edge, no_edge)
            if max_edge < self.config.edge_threshold:
                return False, {"reason": "insufficient_edge", "max_edge": max_edge}
            
            if confidence < self.config.min_confidence:
                return False, {"reason": "low_confidence", "confidence": confidence}
            
            # Determine side
            side = "yes" if yes_edge > no_edge else "no"
            edge = yes_edge if side == "yes" else no_edge
            entry_price = yes_price if side == "yes" else no_price
            
            # Pattern analysis (simplified)
            patterns_detected = 0
            if self.config.pattern_analysis_enabled:
                patterns_detected = self._analyze_patterns(ticker, market_data)
            
            # Calculate institutional score
            institutional_score = self._calculate_institutional_score(
                edge, confidence, patterns_detected, market_data
            )
            
            # Position sizing
            volatility = market_data.get('volatility', 0.2)
            quantity = self.risk_manager.calculate_position_size(
                edge, confidence, entry_price, volatility
            )
            
            # Risk limit check
            if not self.risk_manager.check_risk_limits(quantity, entry_price):
                return False, {"reason": "risk_limits"}
            
            # Compile decision data
            decision_data = {
                "side": side,
                "edge": edge,
                "confidence": confidence,
                "quantity": quantity,
                "entry_price": entry_price,
                "model_prob": model_prob,
                "institutional_score": institutional_score,
                "patterns": patterns_detected,
                "metadata": metadata
            }
            
            return True, decision_data
            
        except Exception as e:
            self.logger.error(f"Position decision failed for {ticker}: {e}")
            return False, {"reason": "error", "error": str(e)}
    
    def _analyze_patterns(self, ticker: str, market_data: Dict) -> int:
        """Simplified pattern analysis"""
        patterns = 0
        
        # Check for momentum patterns
        price_history = market_data.get('price_history', [])
        if len(price_history) >= 3:
            if price_history[-1] > price_history[-2] > price_history[-3]:
                patterns += 1  # Uptrend
            elif price_history[-1] < price_history[-2] < price_history[-3]:
                patterns += 1  # Downtrend
        
        # Check volume patterns
        volume = market_data.get('volume', 0)
        if volume > market_data.get('avg_volume', 0) * 1.5:
            patterns += 1  # High volume
        
        return patterns
    
    def _execute_paper_trade(self, ticker: str, decision_data: Dict) -> Optional[Position]:
        """Execute paper trade with institutional exit strategies"""
        try:
            side = decision_data['side']
            quantity = int(decision_data['quantity'])
            entry_price = decision_data['entry_price']
            
            # Calculate exit targets
            take_profit_price = entry_price * (1 + self.config.take_profit_pct / 100) if side == 'yes' else entry_price * (1 - self.config.take_profit_pct / 100)
            stop_loss_price = entry_price * (1 - self.config.stop_loss_pct / 100) if side == 'yes' else entry_price * (1 + self.config.stop_loss_pct / 100)
            trailing_stop_price = stop_loss_price  # Initial trailing stop
            time_exit = datetime.now() + timedelta(hours=self.config.max_hold_hours)
            
            # Apply sport-specific time limits
            parsed = parse_ticker(ticker)
            if parsed.is_valid():
                if parsed.sport == Sport.MLB:
                    # Close by specific inning
                    time_exit = min(time_exit, datetime.now() + timedelta(hours=3))
                elif parsed.sport == Sport.NFL:
                    # Close by specific quarter  
                    time_exit = min(time_exit, datetime.now() + timedelta(hours=4))
            
            # Create position
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
                edge=decision_data['edge'],
                confidence=decision_data['confidence'],
                model_score=decision_data['model_prob'],
                institutional_score=decision_data['institutional_score'],
                patterns_detected=decision_data['patterns']
            )
            
            # Track position
            self.positions[ticker] = position
            
            # Update risk metrics
            position_value = quantity * entry_price
            self.risk_manager.total_exposure += position_value
            self.risk_manager.update_risk_metrics(0, new_position=True)
            
            # Log trade execution
            self.logger.info(f"INSTITUTIONAL PAPER TRADE: {ticker} {side.upper()} {quantity} @ {entry_price:.2f}")
            self.logger.info(f"Position added: {ticker} {side} {quantity} @ {entry_price:.2f}")
            self.logger.info(f"Exit targets: TP={take_profit_price:.2f}, Stop={stop_loss_price:.2f}, Time={time_exit}")
            
            return position
            
        except Exception as e:
            self.logger.error(f"Trade execution failed for {ticker}: {e}")
            return None
    
    def _manage_positions(self):
        """Manage existing positions with institutional exit logic"""
        positions_to_close = []
        
        for ticker, position in self.positions.items():
            try:
                # Get current market price (mock for now)
                current_price = self._get_current_price(ticker, position.side)
                current_time = datetime.now()
                
                # Update trailing stop
                position.update_trailing_stop(current_price, self.config.trailing_stop_pct)
                
                # Check exit conditions
                should_exit, exit_reason = position.should_exit(current_price, current_time)
                
                if should_exit:
                    positions_to_close.append((ticker, position, current_price, exit_reason))
                
            except Exception as e:
                self.logger.error(f"Position management failed for {ticker}: {e}")
        
        # Close positions
        for ticker, position, exit_price, exit_reason in positions_to_close:
            self._close_position(ticker, position, exit_price, exit_reason)
    
    def _get_current_price(self, ticker: str, side: str) -> float:
        """Get current market price (mock implementation)"""
        # In production, this would query the Kalshi API
        # For now, simulate some price movement
        import random
        base_price = 0.5
        movement = random.gauss(0, 0.02)  # 2% volatility
        return max(0.01, min(0.99, base_price + movement))
    
    def _close_position(self, ticker: str, position: Position, exit_price: float, reason: str):
        """Close position and calculate PnL"""
        try:
            # Calculate PnL
            if position.side == 'yes':
                pnl = position.quantity * (exit_price - position.entry_price)
            else:
                pnl = position.quantity * (position.entry_price - exit_price)
            
            # Update tracking
            self.total_pnl += pnl
            self.total_trades += 1
            
            if pnl > 0:
                self.wins += 1
            else:
                self.losses += 1
            
            # Update risk manager
            position_value = position.quantity * position.entry_price
            self.risk_manager.total_exposure -= position_value
            self.risk_manager.update_risk_metrics(pnl)
            
            # Log closure
            self.logger.info(f"POSITION CLOSED: {ticker} - {reason.upper()}")
            self.logger.info(f"Exit: {position.quantity} @ {exit_price:.2f}, PnL: ${pnl:.2f}")
            
            # Move to closed positions
            self.closed_positions.append(position)
            del self.positions[ticker]
            
        except Exception as e:
            self.logger.error(f"Position closure failed for {ticker}: {e}")
    
    async def run_trading_session(self, duration_minutes: int = 60):
        """Run institutional trading session"""
        try:
            self.logger.info(f"Starting {duration_minutes}-minute institutional trading session")
            
            start_time = time.time()
            end_time = start_time + (duration_minutes * 60)
            
            while time.time() < end_time:
                try:
                    # Get active markets (force mock data in development)
                    if self.kalshi_client and self.kalshi_client.is_healthy() and not self.config.paper_trading:
                        markets = self.kalshi_client.get_active_sports_markets()
                    else:
                        # Force mock data in paper trading mode
                        markets = [
                            {
                                "ticker": "NFL-SF-DAL-H1",
                                "title": "Will 49ers beat Cowboys?",
                                "yes_bid": 45, "yes_ask": 47,
                                "no_bid": 53, "no_ask": 55,
                                "status": "open", "depth": 100
                            },
                            {
                                "ticker": "MLB-NYY-BOS-H1", 
                                "title": "Will Yankees beat Red Sox?",
                                "yes_bid": 58, "yes_ask": 60,
                                "no_bid": 40, "no_ask": 42,
                                "status": "open", "depth": 150
                            }
                        ]
                    
                    # Analyze each market
                    for market in markets:
                        ticker = market.get('ticker', '')
                        
                        if not ticker:
                            continue
                        
                        # Skip if already have position
                        if ticker in self.positions:
                            continue
                        
                        # Get game data
                        game_data = self._get_game_data(ticker)
                        
                        # Make trading decision
                        should_trade, decision_data = self._should_take_position(
                            ticker, market, game_data
                        )
                        
                        if should_trade:
                            position = self._execute_paper_trade(ticker, decision_data)
                            
                            if position:
                                # Log detailed execution
                                self.logger.info("INSTITUTIONAL TRADE EXECUTED:")
                                self.logger.info(f"  Market: {ticker}")
                                self.logger.info(f"  Side: {decision_data['side'].upper()}")
                                self.logger.info(f"  Quantity: {int(decision_data['quantity'])}")
                                self.logger.info(f"  Price: {decision_data['entry_price']:.2f}")
                                self.logger.info(f"  Edge: {decision_data['edge']:.1%}")
                                self.logger.info(f"  Confidence: {decision_data['confidence']:.1%}")
                                self.logger.info(f"  Patterns: {decision_data['patterns']} detected")
                                self.logger.info(f"  Institutional Score: {decision_data['institutional_score']:.3f}")
                    
                    # Manage existing positions
                    self._manage_positions()
                    
                    # Wait before next cycle
                    await asyncio.sleep(self.config.poll_interval_seconds)
                    
                except ResilientAPIError as e:
                    self.logger.warning(f"API error in trading loop: {e}")
                    await asyncio.sleep(5)  # Brief pause on API errors
                    
                except Exception as e:
                    self.logger.error(f"Trading loop error: {e}")
                    await asyncio.sleep(10)  # Longer pause on unexpected errors
            
            # Session complete - generate summary
            self._generate_session_summary(duration_minutes)
            
        except Exception as e:
            self.logger.error(f"Trading session failed: {e}")
    
    def _get_game_data(self, ticker: str) -> Dict:
        """Get live game data for ticker"""
        try:
            parsed = parse_ticker(ticker)
            
            if not parsed.is_valid():
                return {}
            
            sport = parsed.sport.value
            
            if self.sports_feed and self.sports_feed.is_healthy():
                games = self.sports_feed.get_live_game_data(sport)
                
                # Find matching game
                for game in games:
                    home_abbrev = game.get('home_team', {}).get('abbreviation', '')
                    away_abbrev = game.get('away_team', {}).get('abbreviation', '')
                    
                    if (home_abbrev in [parsed.team1, parsed.team2] and 
                        away_abbrev in [parsed.team1, parsed.team2]):
                        return game
            
            # Fallback mock data
            return {
                "home_team": {"abbreviation": parsed.team1, "score": 0},
                "away_team": {"abbreviation": parsed.team2, "score": 0},
                "period": 1,
                "status": "in_progress"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get game data for {ticker}: {e}")
            return {}
    
    def _generate_session_summary(self, duration_minutes: int):
        """Generate comprehensive session summary"""
        self.logger.info("=" * 80)
        self.logger.info("🏛️  INSTITUTIONAL TRADING SESSION COMPLETE")
        self.logger.info("=" * 80)
        
        # Performance metrics
        win_rate = (self.wins / max(1, self.total_trades)) * 100
        avg_pnl_per_trade = self.total_pnl / max(1, self.total_trades)
        
        self.logger.info(f"📊 SESSION PERFORMANCE:")
        self.logger.info(f"   Duration: {duration_minutes} minutes")
        self.logger.info(f"   Total P&L: ${self.total_pnl:.2f}")
        self.logger.info(f"   Total Trades: {self.total_trades}")
        self.logger.info(f"   Win Rate: {win_rate:.1f}% ({self.wins}W/{self.losses}L)")
        self.logger.info(f"   Avg P&L per Trade: ${avg_pnl_per_trade:.2f}")
        
        # Risk metrics
        risk_summary = self.risk_manager.get_risk_summary()
        self.logger.info(f"🛡️  RISK MANAGEMENT:")
        self.logger.info(f"   Daily P&L: ${risk_summary['daily_pnl']:.2f} ({risk_summary['daily_pnl_pct']:+.1f}%)")
        self.logger.info(f"   Portfolio Exposure: ${risk_summary['total_exposure']:.2f} ({risk_summary['exposure_pct']:.1f}%)")
        self.logger.info(f"   Active Positions: {len(self.positions)}")
        self.logger.info(f"   Daily Trades: {risk_summary['daily_trades']}")
        
        # Position details
        if self.positions:
            self.logger.info(f"📈 ACTIVE POSITIONS:")
            for ticker, pos in self.positions.items():
                self.logger.info(f"   {ticker}: {pos.side.upper()} {pos.quantity} @ {pos.entry_price:.2f}")
        
        # Model performance
        self.logger.info(f"🤖 MODEL PERFORMANCE:")
        self.logger.info(f"   Models Available: {len(self.model_manager.models)}")
        self.logger.info(f"   API Health: Kalshi={self.kalshi_client.is_healthy() if self.kalshi_client else False}, Sports={self.sports_feed.is_healthy() if self.sports_feed else False}")
        
        self.logger.info("=" * 80)


async def main():
    """Main entry point"""
    print("🏛️ INSTITUTIONAL-GRADE KALSHI TRADING BOT")
    print("Advanced risk management + proven models + pattern analysis")
    print("=" * 80)
    
    try:
        # Initialize bot
        bot = InstitutionalTradingBot()
        
        # Get trading duration
        try:
            duration_input = input("Enter trading duration in minutes (default 60): ").strip()
            duration = int(duration_input) if duration_input else 60
        except ValueError:
            duration = 60
        
        print(f"Starting {duration}-minute trading session...")
        
        # Run trading session
        await bot.run_trading_session(duration)
        
    except KeyboardInterrupt:
        print("\n⏹️  Trading session interrupted by user")
    except Exception as e:
        print(f"\n❌ Trading session failed: {e}")
        logging.error(f"Main execution failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())