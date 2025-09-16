#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inspect_feeds.py  —  Kalshi sports market inspector with ESPN enrichment

- Works with your existing kalshi_auth_fixed.get_auth().
- If envs are missing, auto-fills from ./kalshi_private_key.pem and ./kalshi_key_id.txt.
- Robust ticker parsing so NFL/MLB spreads/totals/etc. are recognized as sports.
- ESPN scoreboard lookups (public site API) to mark live/pre/final.
- Clear fallbacks & diagnostics when ESPN is unavailable.

Examples:
  python inspect_feeds.py
  python inspect_feeds.py --live-only --kalshi-live-fallback --espn-debug --sports nfl,mlb,tennis
  python inspect_feeds.py --today-only --no-espn --sports nfl,mlb --min-sports 60 --max-pages 30
"""

import os
import re
import sys
import asyncio
import logging
from datetime import datetime, timezone, timedelta, date
from typing import Dict, Any, List, Optional, Tuple

# =========================
# Small helpers and display
# =========================

def month_str_to_num(mon: str) -> int:
    m = mon.upper()
    return {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
            "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}.get(m, 1)

def parse_kalshi_date(ticker: str) -> Optional[date]:
    """
    Parse YY-MON-DD patterns like ...-25SEP16... -> date(2025, 9, 16)
    """
    m = re.search(r'(\d{2})([A-Z]{3})(\d{2})', ticker.upper())
    if not m:
        return None
    yy, mon, dd = m.groups()
    try:
        year = 2000 + int(yy)
        month = month_str_to_num(mon)
        day = int(dd)
        return date(year, month, day)
    except Exception:
        return None

def fmt_time(ts: Optional[int]) -> str:
    if not ts:
        return ""
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")
    except Exception:
        return ""

def short(s: Any, n: int=34) -> str:
    s = "" if s is None else str(s)
    return (s[: n-1] + "…") if len(s) > n else s

def pad(s: str, n: int) -> str:
    return (s or "").ljust(n)[:n]

def live_flag(es_state: str) -> bool:
    s = (es_state or "").lower()
    return s in {"live", "in_progress", "in progress", "playing"}

def base_match_id(ticker: str) -> str:
    return ticker.rsplit('-', 1)[0] if '-' in ticker else ticker

def yyyymmdd_from_token(yy_mon_dd: str) -> str:
    # "25SEP16" -> "20250916"
    yy = int(yy_mon_dd[:2])
    mon = month_str_to_num(yy_mon_dd[2:5])
    dd = int(yy_mon_dd[5:7])
    year = 2000 + yy
    return f"{year:04d}{mon:02d}{dd:02d}"

def is_today_ish(ev_date: Optional[date], now_local: datetime, cutoff_hours: int) -> bool:
    if not ev_date:
        return False
    delta_days = (ev_date - now_local.date()).days
    # exact today
    if delta_days == 0:
        return True
    # very late previous day counts as today-ish if we're within cutoff from midnight
    if delta_days == -1 and now_local.hour < cutoff_hours:
        return True
    # very early next day counts as today-ish if we're within cutoff to midnight
    if delta_days == 1 and (24 - now_local.hour) < cutoff_hours:
        return True
    return False

# =====================
# Sport / league detect
# =====================

LEAGUE_TO_SPORT = {
    # Big four
    "NFL":"nfl","MLB":"mlb","NBA":"nba","NHL":"nhl","WNBA":"wnba",
    # Tennis
    "ATP":"tennis","WTA":"tennis","TENNIS":"tennis",
    # NCAA (optional)
    "NCAAF":"ncaaf","NCAAB":"ncaab",
    # Soccer families
    "MLS":"mls","EPL":"soccer","LALIGA":"soccer","SERIEA":"soccer","BUNDES":"soccer","LIGUE1":"soccer",
    "UCL":"soccer","UEL":"soccer","USOC":"soccer","FA":"soccer","COPA":"soccer",
}

KNOWN_LEAGUES = sorted(LEAGUE_TO_SPORT.keys(), key=len, reverse=True)

# Examples of Kalshi tickers:
# KXNFLSPREAD-25SEP18MIABUF
# KXNFLTOTAL-25SEP18MIABUF
# KXNFLGAME-25SEP18MIABUF
# KXWTAMATCH-25SEP16HADBAC
# KXMLSGAME-25SEP20DALCOL
# KXLALIGAGAME-25SEP23ESPVCF
PREFIX_RE = re.compile(r'^KX([A-Z]+)-')

def parse_league_from_ticker(ticker: str) -> Optional[str]:
    t = ticker.upper()
    m = PREFIX_RE.match(t)
    if not m:
        return None
    prefix = m.group(1)  # e.g., NFLSPREAD, WTAMATCH, LALIGAGAME, MLBTEAMTOTAL...
    # take the longest KNOWN league that is a prefix of this token
    for league in KNOWN_LEAGUES:
        if prefix.startswith(league):
            return league
    return None

def detect_sport(ticker: str) -> str:
    league = parse_league_from_ticker(ticker)
    if not league:
        return "unknown"
    return LEAGUE_TO_SPORT.get(league, "unknown")

# =============================
# ESPN site scoreboard (inline)
# =============================

# https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard?dates=YYYYMMDD
LEAGUE_SLUGS = {
    "NFL": ("football", "nfl"),
    "MLB": ("baseball", "mlb"),
    "NBA": ("basketball", "nba"),
    "NHL": ("hockey", "nhl"),
    "WNBA": ("basketball", "wnba"),
    "WTA": ("tennis", "wta"),
    "ATP": ("tennis", "atp"),
    "MLS": ("soccer", "usa.1"),
    "EPL": ("soccer", "eng.1"),
    "LALIGA": ("soccer", "esp.1"),
    "SERIEA": ("soccer", "ita.1"),
    "BUNDES": ("soccer", "ger.1"),
    "LIGUE1": ("soccer", "fra.1"),
    # generic cups default to EPL path so the endpoint is valid (best-effort)
    "UCL": ("soccer", "eng.1"),
    "UEL": ("soccer", "eng.1"),
    "USOC": ("soccer", "eng.1"),
    "FA": ("soccer", "eng.1"),
    "COPA": ("soccer", "eng.1"),
}

ABBREV_NORMALIZE = {
    # NFL
    "WAS": "WSH", "WSH": "WSH",
    "LVR": "LV", "LV": "LV",
    "NWE": "NE", "NE": "NE",
    "TBB": "TB", "TB": "TB",
    "LAK": "LAC", "LAC": "LAC",
    # MLB (add more as you see mismatches)
    "ANA": "LAA", "LAA": "LAA",
    "LAN": "LAD", "LAD": "LAD",
}

def _normalize_abbrev(abbr: Optional[str], league_code: str) -> str:
    if not abbr:
        return ""
    key = f"{abbr}-{league_code}"
    return ABBREV_NORMALIZE.get(key, ABBREV_NORMALIZE.get(abbr, abbr)).upper()

def _split_team_pair(pair: str) -> Tuple[Optional[str], Optional[str]]:
    # "MIABUF" -> ("MIA","BUF"), "NYYBOS" -> ("NYY","BOS")
    if not pair or len(pair) % 2 != 0:
        return None, None
    half = len(pair) // 2
    return pair[:half], pair[half:]

class RealSportsFeed:
    """ESPN scoreboard-based live state lookups with caching + robust ticker parsing."""
    def __init__(self):
        self._session = None

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            import aiohttp
            self._session = aiohttp.ClientSession(headers={"User-Agent": "KalshiAlphaBot/1.0"})

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    def _scoreboard_url(self, league_code: str, yyyymmdd: str) -> Optional[str]:
        sl = LEAGUE_SLUGS.get(league_code)
        if not sl:
            return None
        sport, league = sl
        return f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard?dates={yyyymmdd}"

    async def _fetch_json(self, url: str) -> dict | None:
        await self._ensure_session()
        try:
            async with self._session.get(url, timeout=6) as r:
                if r.status != 200:
                    return {"_error": f"status {r.status}"}
                return await r.json()
        except Exception as e:
            return {"_error": str(e)}

    @staticmethod
    def _state_from_event(ev: dict) -> str:
        try:
            comp = (ev.get("competitions") or [])[0]
            s = comp.get("status", {}).get("type", {}).get("state", "").lower()
            return {"pre":"pre", "in":"live", "post":"final"}.get(s, s or "")
        except Exception:
            return ""

    @staticmethod
    def _teams_from_event(ev: dict):
        try:
            comp = (ev.get("competitions") or [])[0]
            comps = comp.get("competitors") or []
            home = next((c for c in comps if c.get("homeAway") == "home"), None)
            away = next((c for c in comps if c.get("homeAway") == "away"), None)
            return home, away
        except Exception:
            return None, None

    def _event_matches(self, ev: dict, away_abbr: str, home_abbr: str) -> bool:
        home, away = self._teams_from_event(ev)
        if not home or not away:
            return False
        eh = (home.get("team", {}) or {}).get("abbreviation", "").upper()
        ea = (away.get("team", {}) or {}).get("abbreviation", "").upper()
        return (ea == away_abbr and eh == home_abbr)

    async def get_game_state(self, ticker: str) -> Optional[dict]:
        # Parse league + date + team pair from ticker
        t = ticker.upper()
        m = re.search(r'^KX([A-Z]+)-(\d{2}[A-Z]{3}\d{2})([A-Z]{4,8})?', t)
        if not m:
            return None
        league_token, date_token, teams_blk = m.group(1), m.group(2), m.group(3) or ""
        # Reduce league_token to a known code
        league_code = None
        for L in KNOWN_LEAGUES:
            if league_token.startswith(L):
                league_code = L
                break
        if not league_code:
            return None

        yyyymmdd = yyyymmdd_from_token(date_token)
        url = self._scoreboard_url(league_code, yyyymmdd)
        if not url:
            return None

        away_abbr, home_abbr = _split_team_pair(teams_blk)
        away_abbr = _normalize_abbrev(away_abbr, league_code)
        home_abbr = _normalize_abbrev(home_abbr, league_code)

        data = await self._fetch_json(url)
        if not data or data.get("_error"):
            return None

        events = data.get("events") or []
        match = None
        for ev in events:
            if away_abbr and home_abbr and self._event_matches(ev, away_abbr, home_abbr):
                match = ev
                break

        if not match and events and (not away_abbr or not home_abbr):
            # If we couldn't parse teams, fall back to first event for that date
            match = events[0]

        if not match:
            return None

        state = self._state_from_event(match)
        home, away = self._teams_from_event(match)

        def pack(side):
            if not side:
                return {}
            team = side.get("team", {}) or {}
            score = side.get("score")
            try:
                score = int(score)
            except Exception:
                pass
            return {
                "abbr": (team.get("abbreviation") or "").upper(),
                "name": team.get("displayName") or team.get("name") or "",
                "score": score,
            }

        return {
            "game_state": state or "",
            "event_id": match.get("id", ""),
            "home_team": pack(home),
            "away_team": pack(away),
            "source": "espn",
        }

# ==========================
# Kalshi auth (compat/fallback)
# ==========================

def get_auth_with_fallback(logger: logging.Logger):
    """
    Use your existing kalshi_auth_fixed.get_auth(). If required envs are missing,
    auto-populate from local defaults:
       ./kalshi_private_key.pem
       ./kalshi_key_id.txt
    """
    try:
        from kalshi_auth_fixed import get_auth
    except Exception as e:
        logger.error("kalshi_auth_fixed import failed: %s", e)
        raise

    def ensure_envs():
        changed = False
        if not os.getenv("KALSHI_PRIVATE_KEY_PATH"):
            if os.path.exists("./kalshi_private_key.pem"):
                os.environ["KALSHI_PRIVATE_KEY_PATH"] = "./kalshi_private_key.pem"
                changed = True
        if not os.getenv("KALSHI_KEY_ID"):
            key_path = "./kalshi_key_id.txt"
            if os.path.exists(key_path):
                try:
                    kid = open(key_path, "r", encoding="utf-8").read().strip()
                    if kid:
                        os.environ["KALSHI_KEY_ID"] = kid
                        changed = True
                except Exception:
                    pass
        return changed

    try:
        return get_auth()
    except Exception as e1:
        # Try to load defaults and retry
        if ensure_envs():
            try:
                return get_auth()
            except Exception as e2:
                logger.error("Kalshi auth failed after fallback: %s", e2)
                raise
        logger.error("Kalshi auth failed: %s", e1)
        raise

# ==========================
# Kalshi market pagination
# ==========================

def fetch_markets_paged(auth,
                        status: str = "open",
                        max_pages: int = 20,
                        min_sports: int = 40,
                        sports_allow: Optional[set] = None,
                        show_unknown: bool = False,
                        logger: Optional[logging.Logger] = None) -> List[Dict[str, Any]]:
    """
    Fetch Kalshi markets across pages until we either hit max_pages or
    we've seen at least min_sports *sports* markets as determined by ticker parsing.

    NOTE: Kalshi doesn't have a "category=sports" server-side filter in the Markets API,
          so we page and filter client-side.
    """
    if logger is None:
        logger = logging.getLogger("inspect")

    all_markets: List[Dict[str, Any]] = []
    cursor_keys = ("next", "next_cursor", "cursor", "nextPageToken", "next_page_token")
    params: Dict[str, Any] = {}
    if status and status.lower() != "any":
        params["status"] = status.lower()

    seen_sports = 0
    pages = 0
    next_val = None

    while pages < max_pages:
        if next_val:
            # try common param names for cursor
            for name in ("cursor", "page_token", "next", "next_cursor"):
                params[name] = next_val

        data = auth.get_markets(**params)
        markets = data.get("markets", [])
        all_markets.extend(markets)

        # count sports in this page
        for m in markets:
            t = m.get("ticker", "")
            sp = detect_sport(t)
            if sp == "unknown" and not show_unknown:
                continue
            if sports_allow and sp not in sports_allow:
                continue
            seen_sports += 1

        # find cursor
        next_val = None
        for k in cursor_keys:
            v = data.get(k)
            if v:
                next_val = v
                break

        pages += 1
        if seen_sports >= min_sports or not next_val:
            break

    logger.info(
        "Fetched %d markets from Kalshi (paged up to %d, min_sports=%d target, seen_sports=%d)",
        len(all_markets), pages, min_sports, seen_sports
    )
    return all_markets

# =====================
# ESPN enrichment block
# =====================

async def enrich_with_espn(feed: RealSportsFeed, rows: List[Dict[str, Any]], debug: bool=False) -> None:
    """
    Add 'espn_state', 'espn_note', and 'espn_id' (best effort).
    """
    for r in rows:
        t = r["ticker"]
        try:
            gs = await feed.get_game_state(t)
        except Exception as e:
            gs = None
            if debug:
                r["espn_debug"] = f"lookup error: {e}"

        r["espn_state"] = (gs or {}).get("game_state")
        r["espn_id"] = (gs or {}).get("event_id", "")

        if gs and "home_team" in gs and "away_team" in gs:
            ht, at = gs["home_team"], gs["away_team"]
            r["espn_note"] = f'{short(at.get("name"))} {at.get("score",0)}–{ht.get("score",0)} {short(ht.get("name"))}'
        else:
            r["espn_note"] = ""
            if debug and not r.get("espn_debug"):
                r["espn_debug"] = "no espn match"

# =====================
# Main
# =====================

async def main():
    import argparse

    parser = argparse.ArgumentParser()
    # Filtering intent
    parser.add_argument("--live-only", action="store_true", help="only items ESPN marks as live (or fallback heuristic)")
    parser.add_argument("--today-only", action="store_true", help="only events dated today (or today-ish by cutoff)")
    parser.add_argument("--today-ish-cutoff", type=int, default=6, help="hours around midnight to include prior/next day")
    parser.add_argument("--kalshi-live-fallback", action="store_true",
                        help="if ESPN not live, treat (today-ish + open + quoted) as likely live")
    # ESPN
    parser.add_argument("--no-espn", action="store_true", help="disable ESPN lookups")
    parser.add_argument("--espn-debug", action="store_true", help="print ESPN debug reasons + summary counts")
    # Sports filter
    parser.add_argument("--sports", type=str, default="",
                        help="comma list (e.g. nfl,mlb,tennis,soccer,mls,wnba). Empty = all.")
    parser.add_argument("--show-unknown", action="store_true", help="include rows where sport detection is unknown")
    # Market paging and status
    parser.add_argument("--status", type=str, default="open", choices=["open","closed","any"],
                        help="market status filter for Kalshi paging")
    parser.add_argument("--max-pages", type=int, default=20, help="max pages to fetch from Kalshi")
    parser.add_argument("--min-sports", type=int, default=40, help="stop paging when at least this many sports markets have been seen")
    # Row shaping
    parser.add_argument("--no-collapse", action="store_true", help="show each market leg (no outcome collapsing)")
    parser.add_argument("--limit", type=int, default=0, help="limit number of rows shown (after filtering)")
    parser.add_argument("--dump-unknown", type=int, default=0, help="print N sample unknown tickers w/ titles")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    log = logging.getLogger("inspect")

    # Sports allow-list
    sports_allow = None
    if args.sports.strip():
        sports_allow = {s.strip().lower() for s in args.sports.split(",") if s.strip()}

    # Auth (with compatibility fallback)
    try:
        auth = get_auth_with_fallback(log)
    except Exception:
        return

    # Fetch markets with early-exit once sports are surfaced
    markets = fetch_markets_paged(
        auth,
        status=args.status,
        max_pages=args.max_pages,
        min_sports=args.min_sports,
        sports_allow=sports_allow,
        show_unknown=args.show_unknown,
        logger=log
    )

    # Build rows (sports only unless show-unknown)
    today_local = datetime.now().astimezone()
    rows: List[Dict[str, Any]] = []
    unknowns: List[Dict[str, Any]] = []
    seen_base = set()

    for m in markets:
        ticker = m.get("ticker", "")
        if not ticker:
            continue
        sport = detect_sport(ticker)
        if sport == "unknown" and not args.show_unknown:
            unknowns.append({"ticker": ticker, "title": m.get("title","")})
            continue
        if sports_allow and (sport not in sports_allow):
            continue

        base_id = base_match_id(ticker)
        if (not args.no_collapse) and (base_id in seen_base):
            continue
        if not args.no_collapse:
            seen_base.add(base_id)

        row = {
            "sport": sport,
            "ticker": ticker,
            "base_id": base_id,
            "title": m.get("title", ""),
            "yes_bid": m.get("yes_bid", ""),
            "yes_ask": m.get("yes_ask", ""),
            "no_bid":  m.get("no_bid", ""),
            "no_ask":  m.get("no_ask", ""),
            "status":  m.get("status", ""),
            "volume":  m.get("volume", 0),
            "open_interest": m.get("open_interest", 0),
            "open_time": fmt_time(m.get("open_time")),
            "close_time": fmt_time(m.get("close_time")),
            "event_date": parse_kalshi_date(ticker),
        }
        rows.append(row)

    # ESPN enrichment
    if args.no_espn:
        log.info("ESPN lookups disabled (--no-espn)")
    else:
        log.info("Initializing Real Sports Feed (ESPN)...")
        feed = RealSportsFeed()
        await enrich_with_espn(feed, rows, debug=args.espn_debug)
        await feed.close()

    # Heuristic: likely live by Kalshi when ESPN is absent/broken
    def is_likely_live_by_kalshi(r: Dict[str, Any]) -> bool:
        try:
            todayish = is_today_ish(r.get("event_date"), today_local, args.today_ish_cutoff)
            open_status = (str(r.get("status","")).lower() == "open")
            yb, ya, nb, na = r.get("yes_bid"), r.get("yes_ask"), r.get("no_bid"), r.get("no_ask")
            has_quotes = (yb not in (None,"")) or (ya not in (None,"")) or (nb not in (None,"")) or (na not in (None,""))
            return bool(todayish and open_status and has_quotes)
        except Exception:
            return False

    def passes_filters(r: Dict[str, Any]) -> bool:
        es_live = live_flag(r.get("espn_state"))
        todayish = is_today_ish(r.get("event_date"), today_local, args.today_ish_cutoff)

        if args.live_only:
            if es_live:
                return True
            if args.kalshi_live_fallback and is_likely_live_by_kalshi(r):
                r.setdefault("espn_debug", "fallback: today-ish+open+quoted")
                return True
            return False

        if args.today_only:
            return todayish

        # default: live OR today-ish
        return es_live or todayish

    filtered = [r for r in rows if passes_filters(r)]

    # Sort live first, then today-ish, then by sport/base_id
    def sort_key(r):
        live = 0 if (live_flag(r.get("espn_state")) or (args.kalshi_live_fallback and is_likely_live_by_kalshi(r))) else 1
        todayish = 0 if is_today_ish(r.get("event_date"), today_local, args.today_ish_cutoff) else 1
        return (live, todayish, r.get("sport",""), r.get("base_id",""))

    filtered.sort(key=sort_key)

    if args.limit > 0:
        filtered = filtered[: args.limit]

    # Table header
    hdr = [
        ("SPORT", 7),
        ("LIVE", 4),
        ("TODAY", 5),
        ("TICKER", 34),
        ("TITLE", 34),
        ("YES_BID/ASK", 12),
        ("NO_BID/ASK", 12),
        ("STATE", 8),
        ("NOTE", 28),
    ]
    line = "  ".join(pad(h, w) for h, w in hdr)
    print(line)
    print("-" * len(line))

    for r in filtered:
        es_live_char = "Y" if (live_flag(r.get("espn_state")) or (args.kalshi_live_fallback and is_likely_live_by_kalshi(r))) else ""
        is_today_char = "Y" if is_today_ish(r.get("event_date"), today_local, args.today_ish_cutoff) else ""
        yes = f"{r.get('yes_bid','')}/{r.get('yes_ask','')}"
        no  = f"{r.get('no_bid','')}/{r.get('no_ask','')}"
        state_str = r.get("espn_state","") or ("fallback" if es_live_char and not live_flag(r.get("espn_state")) else "")
        row = [
            (r.get("sport",""), 7),
            (es_live_char, 4),
            (is_today_char, 5),
            (short(r.get("base_id") or r.get("ticker",""), 34), 34),
            (short(r.get("title",""), 34), 34),
            (yes, 12),
            (no, 12),
            (short(state_str, 8), 8),
            (short(r.get("espn_note",""), 28), 28),
        ]
        print("  ".join(pad(x, w) for x, w in row))

    print()
    print(f"Shown: {len(filtered)}  (from {len(rows)} sports markets, {len(markets)} total Kalshi markets)")

    # Unknowns summary + dump list if asked
    if unknowns and not args.show_unknown:
        print(f"(Filtered out {len(unknowns)} unknown-sport tickers — run with --show-unknown to list them)")
    if args.dump_unknown and unknowns:
        print("\nUnknown tickers (sample):")
        for u in unknowns[: args.dump_unknown]:
            print(f"  {short(u['ticker'], 30):30}  {short(u.get('title',''), 48)}")

    # ESPN diagnostics
    if args.espn_debug:
        total_live = sum(1 for r in rows if live_flag(r.get("espn_state")))
        total_todayish = sum(1 for r in rows if is_today_ish(r.get("event_date"), today_local, args.today_ish_cutoff))
        unmatched = [r for r in rows if not r.get("espn_state")]
        print(f"ESPN live matches: {total_live} | same-day events (today-ish): {total_todayish} | no ESPN match: {len(unmatched)}")
        # Per-sport counts
        counts: Dict[str,int] = {}
        for r in rows:
            counts[r["sport"]] = counts.get(r["sport"], 0) + 1
        if counts:
            keys = sorted(counts.keys())
            print("Per-sport counts: " + ", ".join(f"{k}:{counts[k]}" for k in keys))
        if unmatched[:20]:
            print("\nUnmatched (no ESPN state), sample:")
            for r in unmatched[:20]:
                why = r.get("espn_debug","")
                print(f"  {r['sport']:6}  {short(r['base_id'],34):34}  date={r.get('event_date')}  {why}")

    print("Tip: try without flags (LIVE or TODAY), or add --live-only, --today-only,")
    print("     --kalshi-live-fallback, --no-collapse, --show-unknown, --no-espn, --espn-debug,")
    print("     --today-ish-cutoff 6, --sports nfl,mlb,tennis,soccer,mls,wnba, --status any,")
    print("     --max-pages 30, --min-sports 60, --dump-unknown 30.")
    print()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
