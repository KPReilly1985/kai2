"""
TrailingExit helper for managing exits.
Supports:
- Trailing stop (% from peak)
- Hard stop (% loss from entry)
"""

class TrailingExit:
    def __init__(self, trailing_stop_pct: float, hard_stop_pct: float):
        self.trailing_stop_pct = trailing_stop_pct
        self.hard_stop_pct = hard_stop_pct

    def init(self, entry_price: float) -> dict:
        """
        Initialize state for a new position.
        """
        return {"peak": entry_price}

    def check_exit(self, entry_price: float, current_bid: float, state: dict):
        """
        Update position state and decide if exit is needed.

        Returns:
          - updated state dict if still open
          - "exit" string if exit condition met
        """
        if current_bid <= 0:
            return state

        # Update peak price if new high
        if current_bid > state["peak"]:
            state["peak"] = current_bid

        # Calculate drawdown (trailing stop)
        drawdown = (state["peak"] - current_bid) / state["peak"]
        # Calculate hard stop (absolute loss from entry)
        hard_loss = (entry_price - current_bid) / entry_price

        if drawdown >= self.trailing_stop_pct or hard_loss >= self.hard_stop_pct:
            return "exit"

        return state
