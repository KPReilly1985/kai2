import asyncio
import logging
from typing import Dict, Optional

from core.repricing_engine import RepricingEngine


async def route_entry_async(
    client,
    market: Dict,
    fair: float,
    edge: float,
    bankroll: float,
    cfg: Dict,
    notify_hook,
    trade_logger,
    state,
    repricing_engine: Optional[RepricingEngine] = None,
):
    """
    Async entry order routing logic.
    Places orders using available bankroll, risk config, and market edge.
    Supports async repricing engine for amendments/cancels.
    """

    try:
        ticker = market["ticker"]
        side = "yes"  # TODO: strategy logic could flip this based on inputs
        contracts = _calculate_position_size(market, bankroll, cfg)

        if contracts <= 0:
            logging.debug(f"[ROUTER_ASYNC] Skipping {ticker}: position size=0")
            return None

        yes_price = int(market.get("last_price", 0))

        order_data = {
            "ticker": ticker,
            "type": "limit",
            "side": side,
            "action": "buy",
            "count": contracts,
            "yes_price": yes_price,
        }

        logging.info(f"[ROUTER_ASYNC] Placing order: {order_data}")

        # Place the order
        resp = await client._call_async("POST", "/portfolio/orders", json_body=order_data)
        order_id = (resp.get("order") or resp).get("id")

        if not order_id:
            logging.warning(f"[ROUTER_ASYNC] No order_id returned for {ticker}")
            return None

        # Repricing management (if enabled)
        if repricing_engine and repricing_engine.enabled:
            asyncio.create_task(
                repricing_engine.manage_repricing(order_id, ticker, yes_price)
            )

        # Log trade
        if trade_logger:
            trade_logger.log(
                ticker=ticker,
                side=side,
                action="BUY",
                qty=contracts,
                price=yes_price / 100.0,
                pnl=0.0,
            )

        # Update simulated state (if paper trading)
        if cfg.get("modes", {}).get("paper_trading", False):
            state.sim_add(ticker, side, contracts, yes_price / 100.0)

        # Send notification
        notify_hook(
            "Entry Order Placed",
            f"{ticker} {contracts}x @ {yes_price}c (async)",
        )

        return order_id

    except Exception as e:
        logging.error(f"[ROUTER_ASYNC] Failed to route entry for {market.get('ticker')}: {e}")
        return None


def _calculate_position_size(market: Dict, bankroll: float, cfg: Dict) -> int:
    """
    Determine position size based on bankroll, edge config, and market price.
    """
    try:
        price = market.get("last_price", 50)
        risk_fraction = cfg["edge"].get("risk_fraction", 0.01)

        max_risk_usd = bankroll * risk_fraction
        cost_per_contract = price / 100.0
        contracts = int(max_risk_usd / cost_per_contract)

        return max(0, contracts)

    except Exception as e:
        logging.error(f"[ROUTER_ASYNC] Position size calculation failed: {e}")
        return 0
