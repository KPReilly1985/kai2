import logging, uuid, time
from models.sizing import bounded_stake
from .edge_engine import market_prob_from_price_cents
import logging
from typing import Dict, Any


def _best_yes_price_from_orderbook(book):
    try:
        bids = book.get("bids_yes") or book.get("bids") or []
        if bids: return int(bids[0]["price"])
    except Exception:
        pass
    return None

def route_entry(client, m, fair, edge, bankroll, cfg, notify_hook, trade_logger=None, state=None):
    cents = m["last_price"]; mkt_prob = market_prob_from_price_cents(cents)
    stake = bounded_stake(edge, mkt_prob, bankroll,
                          cfg["bankroll"]["kelly_fraction"],
                          cfg["bankroll"]["min_stake_usd"],
                          cfg["bankroll"]["max_stake_usd"],
                          cfg["bankroll"]["risk_fraction_per_trade"])
    if stake <= 0: return
    contracts = max(1, int(stake // max(1, cents)))

    paper = cfg.get("modes", {}).get("paper_trading", False)
    reprice_cfg = cfg.get("reprice", {})
    use_reprice = bool(reprice_cfg.get("enabled", True))

    if paper:
        logging.info(f"[PAPER ENTRY] {m['ticker']} {m['title']} stake=${stake} -> {contracts} @ {cents}c")
        if trade_logger:
            trade_logger.log(m["ticker"], "yes", "BUY", contracts, cents/100.0)
        if state:
            state.sim_add(m["ticker"], "yes", contracts, cents/100.0)
        notify_hook("Paper Entry", f"{m['ticker']} {m['title']} @ {cents}c edge={edge:.3f} fair={fair:.3f}")
        return

    # Real order with simple reprice policy
    try:
        order = client.create_order(
            ticker=m["ticker"],
            action="buy",
            side="yes",
            count=contracts,
            type_="limit",
            yes_price=int(cents),
            client_order_id=str(uuid.uuid4()),
            buy_max_cost=int(contracts * cents),
            post_only=False,
        )
        order_id = order.get("order", {}).get("id") or order.get("id")
        logging.info(f"[ENTRY] {m['ticker']} placed order_id={order_id} {contracts}@{cents}c")
        if trade_logger:
            trade_logger.log(m["ticker"], "yes", "BUY", contracts, cents/100.0)
        notify_hook("Entry", f"{m['ticker']} {m['title']} placed {contracts}@{cents}c")
    except Exception as e:
        logging.error(f"entry failed: {e}")
        return

    # Reprice loop
    if not use_reprice or not order_id:
        return

    attempts = 0
    placed_price = int(cents)
    bumped_total = 0
    check_after = int(reprice_cfg.get("check_after_seconds", 8))
    price_bump = int(reprice_cfg.get("price_bump_cents", 1))
    max_bump = int(reprice_cfg.get("max_bump_total_cents", 4))
    spread_cancel = int(reprice_cfg.get("cancel_if_spread_wider_than_cents", 15))

    while attempts < int(reprice_cfg.get("max_attempts", 3)):
        time.sleep(check_after)
        # check order status and book
        try:
            od = client.get_order(order_id)
            status = (od.get("order") or od).get("status","")
            if status in ("filled","cancelled","rejected","expired"):
                logging.info(f"[REPRICE] order {order_id} status={status} -> stop")
                break
            book = client.get_orderbook(m["ticker"]) or {}
            best_yes = _best_yes_price_from_orderbook(book) or placed_price
            ask = (book.get("asks_yes") or [{}])[0].get("price", placed_price)
            bid = (book.get("bids_yes") or [{}])[0].get("price", placed_price)
            spread = abs(int(ask) - int(bid))

            if spread > spread_cancel:
                client.cancel_order(order_id)
                logging.info(f"[REPRICE] Cancel order {order_id} due to wide spread={spread}")
                break

            # If best_yes > placed_price, bump up a cent (within cap)
            bump = min(price_bump, max_bump - bumped_total)
            if bump <= 0:
                logging.info(f"[REPRICE] reached max bump for {order_id}")
                break

            new_price = placed_price + bump
            resp = client.amend_order(order_id, {"yes_price": int(new_price)})
            attempts += 1
            placed_price = new_price
            bumped_total += bump
            logging.info(f"[REPRICE] amended {order_id} -> {new_price}c (attempt {attempts})")
        except Exception as e:
            logging.warning(f"[REPRICE] error: {e}")
            break
