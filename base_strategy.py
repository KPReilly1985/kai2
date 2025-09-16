# strategies/base_strategy.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any


class Signal(Enum):
    """Trading signal types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"  # Exit position


@dataclass
class TradingSignal:
    """Complete trading signal with sizing and metadata"""
    ticker: str
    signal: Signal
    confidence: float  # 0.0 to 1.0
    edge: float  # Expected edge percentage
    kelly_fraction: float  # Position size from Kelly
    size: int  # Actual contracts to trade
    reason: str  # Why this signal was generated
    metadata: Dict[str, Any]  # Additional data (prices, indicators, etc.)


class Strategy(ABC):
    """Base strategy interface - your edge logic goes in subclasses"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.positions: Dict[str, int] = {}  # Track open positions
        
    @abstractmethod
    def evaluate(self, quote: Dict[str, Any]) -> Optional[TradingSignal]:
        """
        Evaluate a quote and return a trading signal if one exists.
        This is where your edge calculation logic lives.
        """
        pass
    
    @abstractmethod
    def calculate_edge(self, quote: Dict[str, Any]) -> float:
        """Calculate expected edge for this opportunity"""
        pass
    
    @abstractmethod
    def calculate_kelly(self, edge: float, win_prob: float, odds: float) -> float:
        """
        Kelly Criterion calculation
        f* = (p*b - q) / b
        where:
        - p = probability of winning
        - q = probability of losing (1-p)
        - b = odds (amount won per dollar bet)
        """
        pass
    
    def update_position(self, ticker: str, size: int, side: str) -> None:
        """Update position tracking after a fill"""
        if side == "buy":
            self.positions[ticker] = self.positions.get(ticker, 0) + size
        else:  # sell
            self.positions[ticker] = self.positions.get(ticker, 0) - size
            if self.positions[ticker] == 0:
                del self.positions[ticker]


# strategies/enhanced_edge_strategy.py
class EnhancedEdgeStrategy(Strategy):
    """
    Your actual edge calculation and Kelly formula implementation.
    This preserves all your existing logic, just wrapped in the pattern.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Extract your edge parameters from config
        edge_cfg = config.get("edge", {})
        self.threshold = edge_cfg.get("threshold", 0.05)
        self.fee_buffer = edge_cfg.get("fee_buffer", 0.01)
        self.min_price_cents = edge_cfg.get("min_price_cents", 5)
        self.max_price_cents = edge_cfg.get("max_price_cents", 95)
        
        # Enhanced edge parameters
        enhanced_cfg = config.get("enhanced_edge", {})
        self.min_confidence = enhanced_cfg.get("min_confidence", 0.4)
        self.advanced_weight = enhanced_cfg.get("advanced_weight", 0.25)
        
        # Kelly parameters
        sizing_cfg = config.get("sizing", {})
        self.kelly_fraction = sizing_cfg.get("kelly_fraction", 0.33)
        self.max_per_trade_risk = sizing_cfg.get("max_per_trade_risk", 0.02)
        
    def evaluate(self, quote: Dict[str, Any]) -> Optional[TradingSignal]:
        """
        Main evaluation logic - determines if there's a trading opportunity
        """
        ticker = quote.get("ticker")
        if not ticker:
            return None
            
        # Calculate edge using your existing logic
        edge = self.calculate_edge(quote)
        
        # Check if edge meets threshold
        if abs(edge) < self.threshold:
            return None
            
        # Determine signal direction
        signal = Signal.BUY if edge > 0 else Signal.SELL
        
        # Calculate confidence (you can enhance this)
        confidence = min(1.0, abs(edge) / 0.10)  # Scale edge to confidence
        
        # Calculate Kelly sizing
        win_prob = 0.5 + (edge / 2)  # Simple conversion, enhance as needed
        odds = 1.0  # Even money for now, adjust based on your model
        kelly = self.calculate_kelly(edge, win_prob, odds)
        
        # Apply Kelly fraction and risk limits
        position_fraction = min(
            kelly * self.kelly_fraction,
            self.max_per_trade_risk
        )
        
        # Calculate actual size (simplified - add your bankroll logic)
        size = int(position_fraction * 1000)  # Placeholder sizing
        
        return TradingSignal(
            ticker=ticker,
            signal=signal,
            confidence=confidence,
            edge=edge,
            kelly_fraction=position_fraction,
            size=size,
            reason=f"Edge {edge:.2%} exceeds threshold",
            metadata={
                "bid": quote.get("best_bid"),
                "ask": quote.get("best_ask"),
                "mid": quote.get("mid"),
                "spread": quote.get("spread")
            }
        )
    
    def calculate_edge(self, quote: Dict[str, Any]) -> float:
        """
        YOUR EDGE CALCULATION LOGIC GOES HERE
        This is where you implement your actual edge formula
        """
        # Example implementation - replace with your actual logic
        bid = quote.get("best_bid", 0)
        ask = quote.get("best_ask", 0)
        mid = (bid + ask) / 2
        
        # Your complex edge calculation
        # This is just a placeholder - use your actual formula
        fair_value = self._calculate_fair_value(quote)
        edge = (fair_value - mid) / mid if mid > 0 else 0
        
        # Apply fee adjustments
        edge -= self.fee_buffer
        
        return edge
    
    def calculate_kelly(self, edge: float, win_prob: float, odds: float) -> float:
        """
        Kelly Criterion: f* = (p*b - q) / b
        
        Args:
            edge: Expected edge as a fraction
            win_prob: Probability of winning (0 to 1)
            odds: Payout odds (e.g., 1.0 for even money)
        
        Returns:
            Optimal fraction of bankroll to bet
        """
        if odds <= 0 or win_prob <= 0 or win_prob >= 1:
            return 0.0
            
        q = 1 - win_prob  # Probability of losing
        
        # Standard Kelly formula
        kelly = (win_prob * odds - q) / odds
        
        # Cap at reasonable levels
        return max(0, min(kelly, 0.25))  # Never bet more than 25%
    
    def _calculate_fair_value(self, quote: Dict[str, Any]) -> float:
        """
        YOUR PROPRIETARY FAIR VALUE CALCULATION
        This is where your secret sauce lives
        """
        # Placeholder - implement your actual model here
        # Could incorporate:
        # - Pattern analysis
        # - Network effects
        # - External factors
        # - Historical data
        # - Sports odds
        # - Sentiment analysis
        # etc.
        
        mid = (quote.get("best_bid", 0) + quote.get("best_ask", 0)) / 2
        
        # Example: adjust based on some factors
        # Replace this with your actual logic
        adjustment = 0.01  # 1 cent adjustment as example
        
        return mid + adjustment