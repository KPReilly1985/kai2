# Simplified config loader that works with your existing config
from types import SimpleNamespace

class ValidatedConfigLoader:
    def load_config(self):
        return SimpleNamespace(
            paper_trading=True,
            bankroll=1000.0,
            min_position_size=1,
            max_position_size=50,
            kelly_fraction=0.10,
            daily_loss_limit_pct=15.0,
            max_positions=8,
            edge_threshold=0.05,
            min_confidence=0.70,
            fee_buffer=0.01,
            take_profit_pct=20.0,
            stop_loss_pct=15.0,
            trailing_stop_pct=5.0,
            max_hold_hours=24,
            poll_interval_seconds=30,
            portfolio_risk_limit_pct=50.0
        )

def load_validated_config():
    loader = ValidatedConfigLoader()
    return loader.load_config()