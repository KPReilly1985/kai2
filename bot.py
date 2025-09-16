# bots/bot.py
from typing import Any, Mapping

class Bot:
    def __init__(self, *, config: Mapping[str, Any]) -> None:
        self.config = dict(config)

    # Optional hooks; EnhancedBot can override any subset
    def start(self) -> None: ...
    def shutdown(self) -> None: ...
    def run_once(self) -> None: ...
    def run_cycle(self) -> None: ...
    def portfolio_value(self) -> float: return 0.0
    def pnl(self) -> float: return 0.0
    def last_pnl(self) -> float: return 0.0
    def stats(self) -> Mapping[str, Any]: return {}
