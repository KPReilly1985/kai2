import math
from core.config_loader import load_config

CFG = load_config()
EDGE_CFG = CFG["edge"]

class EdgeEngine:
    """Compute expected edge on trades."""

    def __init__(self):
        self.min_edge_bp = EDGE_CFG["min_edge_bp"]
        self.slippage_bp = EDGE_CFG["slippage_bp"]
        self.fee_bp = EDGE_CFG["fee_bp"]
        self.min_book_depth = EDGE_CFG["min_book_depth"]
        self.spread_bp_max = EDGE_CFG["spread_bp_max"]

    def compute_edge(self, fair_price: float, yes_price: float, book_depth: int) -> float:
        """Return edge in basis points (bps)."""
        if book_depth < self.min_book_depth:
            return -9999  # treat as untradeable

        spread_bp = abs(fair_price - yes_price) * 10000
        if spread_bp > self.spread_bp_max:
            return -9999

        raw_edge = (fair_price - yes_price) * 10000
        adjusted_edge = raw_edge - self.slippage_bp - self.fee_bp
        return adjusted_edge

    def is_tradeable(self, fair_price: float, yes_price: float, book_depth: int) -> bool:
        edge = self.compute_edge(fair_price, yes_price, book_depth)
        return edge >= self.min_edge_bp
