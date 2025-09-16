import re, pathlib

path = pathlib.Path("clean_working_bot.py")
src = path.read_text(encoding="utf-8")
changed = False

def insert_after_team_mappings(s: str) -> str:
    """Add extra MLB/NFL/NBA mappings after self.team_mappings = {...} in FixedKalshiTickerParser.__init__"""
    global changed
    pat = re.compile(
        r"(class\s+FixedKalshiTickerParser:[\s\S]*?def\s+__init__\([\s\S]*?self\.team_mappings\s*=\s*\{[\s\S]*?\}\n)",
        re.MULTILINE,
    )
    m = pat.search(s)
    if not m:
        return s
    block = m.group(1)
    if "self.team_mappings.update({" in block:
        return s  # already patched
    inject = """\
        # Optional: expand mappings for popular MLB/NFL/NBA team codes
        self.team_mappings.update({
            # MLB
            "NYY": "Yankees", "BOS": "Red Sox", "LAD": "Dodgers", "ATL": "Braves",
            "CHC": "Cubs", "HOU": "Astros", "NYM": "Mets", "PHI": "Phillies",
            "SDP": "Padres", "SFG": "Giants", "SEA": "Mariners", "TBR": "Rays",
            "LAA": "Angels", "CLE": "Guardians", "DET": "Tigers",
            # NFL
            "KC": "Chiefs", "BUF": "Bills", "SF": "49ers", "DAL": "Cowboys",
            "PHI": "Eagles", "NYJ": "Jets", "NYG": "Giants", "MIA": "Dolphins",
            "NE": "Patriots", "BAL": "Ravens", "CIN": "Bengals", "PIT": "Steelers",
            "LAR": "Rams", "LAC": "Chargers", "JAX": "Jaguars",
            # NBA
            "LAL": "Lakers", "BOS": "Celtics", "GSW": "Warriors", "MIA": "Heat",
            "NYK": "Knicks", "DAL": "Mavericks", "MIL": "Bucks",
            "PHX": "Suns", "DEN": "Nuggets", "CHI": "Bulls",
        })
"""
    s2 = s.replace(block, block + inject)
    if s2 != s:
        changed = True
    return s2

def insert_helpers_before_init_auth(s: str) -> str:
    """Add _map_code_to_name and _format_label inside RealKalshiTradingBot before _initialize_kalshi_auth"""
    global changed
    if "_format_label(" in s and "_map_code_to_name(" in s:
        return s
    pat = re.compile(r"(class\s+RealKalshiTradingBot:[\s\S]*?)(\n\s+def\s+_initialize_kalshi_auth\()", re.MULTILINE)
    m = pat.search(s)
    if not m:
        return s
    head = m.group(1)
    rest_start = m.start(2)
    helpers = '''
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
'''
    s2 = s[:rest_start] + helpers + s[rest_start:]
    if s2 != s:
        changed = True
    return s2

def add_label_to_decision_data(s: str) -> str:
    """In _should_take_position, attach label and market snapshot to decision_data before return True"""
    global changed
    func_pat = re.compile(r"def\s+_should_take_position\([\s\S]*?\):([\s\S]*?)\n\s*return\s+True,\s*decision_data", re.MULTILINE)
    m = func_pat.search(s)
    if not m:
        return s
    segment = m.group(1)
    if 'decision_data["label"]' in segment:
        return s
    inject = '''
            # NEW: pretty label for logs + keep market snapshot for context
            decision_data["label"] = self._format_label(
                ticker, market_data, side=side, outcome=parsed.outcome
            )
            decision_data["market"] = market_data
'''
    s2 = s.replace(segment, segment + inject)
    if s2 != s:
        changed = True
    return s2

def replace_paper_trade_log(s: str) -> str:
    """Replace the simple PAPER TRADE log with labeled variant in _execute_real_trade"""
    global changed
    simple_line = re.compile(r'\n\s*self\.logger\.info\(f"PAPER TRADE: \{ticker\} \{side\.upper\(\)\} \{quantity\} @ \{entry_price:.*?\}"\)\n')
    if simple_line.search(s):
        labeled = '''
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
'''
        s2 = simple_line.sub("\n" + labeled, s, count=1)
        if s2 != s:
            changed = True
        return s2
    return s

def add_game_line_after_position_added(s: str) -> str:
    """After 'Position added:' log, add a 'Game: {label}' log"""
    global changed
    pat = re.compile(r'(\n\s*self\.logger\.info\(f"Position added: \{ticker\} \{side\} \{quantity\} @ \{entry_price:.2f\}"\)\n)')
    if not pat.search(s):
        return s
    if 'self.logger.info(f" Game: ' in s:
        return s
    inject = '''\
            # NEW: show game label on executed printout
            label = decision_data.get("label") or self._format_label(
                ticker, decision_data.get("market"), side=side, outcome=parse_ticker(ticker).outcome
            )
            self.logger.info(f" Game: {label}")
'''
    s2 = pat.sub(r"\1" + inject, s, count=1)
    if s2 != s:
        changed = True
    return s2

def label_close_position(s: str) -> str:
    """Swap 'POSITION CLOSED: {ticker}' to use friendly label"""
    global changed
    pat = re.compile(r'\n\s*# Log closure\s*\n\s*self\.logger\.info\(f"POSITION CLOSED: \{ticker\} - \{reason\.upper\(\)\}"\)\n')
    if pat.search(s):
        repl = '''
            # Log closure (NEW: friendly label)
            try:
                label = self._format_label(ticker)
            except Exception:
                label = ticker
            self.logger.info(f"POSITION CLOSED: {label} - {reason.upper()}")
'''
        s2 = pat.sub("\n" + repl, s, count=1)
        if s2 != s:
            changed = True
        return s2
    pat2 = re.compile(r'\n\s*self\.logger\.info\(f"POSITION CLOSED: \{ticker\} - \{reason\.upper\(\)\}"\)\n')
    if pat2.search(s):
        repl = '''
            try:
                label = self._format_label(ticker)
            except Exception:
                label = ticker
            self.logger.info(f"POSITION CLOSED: {label} - {reason.upper()}")
'''
        s2 = pat2.sub("\n" + repl, s, count=1)
        if s2 != s:
            changed = True
        return s2
    return s

# Apply all edits
src2 = src
src2 = insert_after_team_mappings(src2)
src2 = insert_helpers_before_init_auth(src2)
src2 = add_label_to_decision_data(src2)
src2 = replace_paper_trade_log(src2)
src2 = add_game_line_after_position_added(src2)
src2 = label_close_position(src2)

if not changed:
    print("No changes applied (already patched or patterns not found).")
else:
    path.write_text(src2, encoding="utf-8")
    print("Patch applied successfully.")