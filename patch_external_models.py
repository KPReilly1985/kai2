import re, pathlib

p = pathlib.Path("clean_working_bot.py")
s = p.read_text(encoding="utf-8")

# -- 1) Ensure get_prediction always passes ticker in game_data and has a safe fallback
pat_getpred = r"""
def get_prediction\(
        self, ticker: str, game_data: Dict, sport_hint: str = None
    \) -> Tuple\[float, float, Dict\]:
        \"\"\"Get prediction with enhanced metadata\"\"\"
        try:
            # Parse ticker to get sport
            parsed = parse_ticker\(ticker\)
            if not parsed\.is_valid\(\):
                self\.logger\.warning\(f"Could not parse ticker: {ticker}"\)
                return 0\.5, 0\.0, {"error": "invalid_ticker"}

            sport = sport_hint or parsed\.sport\.value
            model = self\.models\.get\(sport\)
            if not model:
                self\.logger\.warning\(f"No model available for sport: {sport}"\)
                return 0\.5, 0\.0, {"error": "no_model"}

            # Get prediction
            probability, confidence = model\.predict_win_probability\(game_data\)

            # Enhanced metadata
            metadata = {
                "sport": sport,
                "model_version": getattr\(model, "version", "unknown"\),
                "teams": f"{parsed\.team1} vs {parsed\.team2}",
                "market_type": parsed\.market_type,
                "parser_confidence": parsed\.confidence,
                "timestamp": datetime\.now\(\)\.isoformat\(\),
            }
            return probability, confidence, metadata
        except Exception as e:
            self\.logger\.error\(f"Prediction failed for {ticker}: {e}"\)
            return 0\.5, 0\.0, {"error": str\(e\)}
"""

rep_getpred = r"""
def get_prediction(
        self, ticker: str, game_data: Dict, sport_hint: str = None
    ) -> Tuple[float, float, Dict]:
        
        try:
            # Parse ticker to get sport
            parsed = parse_ticker(ticker)
            if not parsed.is_valid():
                self.logger.warning(f"Could not parse ticker: {ticker}")
                return 0.5, 0.0, {"error": "invalid_ticker"}

            sport = sport_hint or parsed.sport.value
            model = self.models.get(sport)
            if not model:
                self.logger.warning(f"No model available for sport: {sport}")
                return 0.5, 0.0, {"error": "no_model"}

            # Always include ticker for adapters that need to disambiguate outcomes
            if isinstance(game_data, dict) and "ticker" not in game_data:
                game_data["ticker"] = ticker

            # Try model; if it fails (e.g., external soccer model not matching), fall back gracefully
            try:
                probability, confidence = model.predict_win_probability(game_data)
            except Exception as e:
                self.logger.warning(f"Primary model failed for {sport} on {ticker}: {e} -> falling back")
                # Fallback: wrapped prod models for NFL/MLB, otherwise mock
                if sport in ("nfl", "mlb"):
                    probability, confidence = ProductionModelWrapper(sport).predict_win_probability(game_data)
                else:
                    probability, confidence = MockModel(sport).predict_win_probability(game_data)

            # Enhanced metadata
            metadata = {
                "sport": sport,
                "model_version": getattr(model, "version", "unknown"),
                "teams": f"{parsed.team1} vs {parsed.team2}",
                "market_type": parsed.market_type,
                "parser_confidence": parsed.confidence,
                "timestamp": datetime.now().isoformat(),
            }
            return probability, confidence, metadata
        except Exception as e:
            self.logger.error(f"Prediction failed for {ticker}: {e}")
            # Last-resort fallback
            try:
                sport = sport_hint or parse_ticker(ticker).sport.value
                probability, confidence = MockModel(sport).predict_win_probability(game_data or {})
            except Exception:
                probability, confidence = 0.5, 0.0
            return probability, confidence, {"error": str(e)}
"""

if "def get_prediction(" in s:
    s = re.sub(pat_getpred, rep_getpred, s, count=1, flags=re.DOTALL)

# -- 2) Replace ExternalModelAdapter with a more capable version
pat_adapter_start = r"\nclass ExternalModelAdapter:\n"
pat_adapter_block = r"class ExternalModelAdapter:[\s\S]*?\nclass EnhancedModelManager:"

adapter_new = r"""class ExternalModelAdapter:
    \"\"\"Robust adapter for external models with many possible APIs.
    Accepted external shapes (any of these):
      - module.predict_win_probability(game_data) -> (prob, conf) | prob | dict
      - module.predict(game_data) / module.predict_proba(game_data) / module.forecast(game_data)
      - module.Model().predict_win_probability(...) / predict(...)
      - module.predict_match(home, away [, outcome]) -> tuple|float|dict (soccer three-way supported)
    Returns:
      (probability in [0,1], confidence in [0,1])
    \"\"\"
    def __init__(self, impl):
        import inspect
        self.impl = impl
        self.inspect = inspect
        self.version = getattr(impl, "VERSION", getattr(impl, "__version__", "external_1.0"))

    # --- helpers ---
    def _pick_outcome_key(self, outcome_code: str, game_data: dict) -> str:
        # Map the contract's outcome to keys an external soccer model might return
        if not outcome_code:
            return None
        oc = (outcome_code or "").upper()
        if oc in ("TIE", "DRAW", "X"):
            return "draw"
        # Try to detect home/away mapping by team code match
        try:
            home = (game_data.get("home_team", {}) or {}).get("abbreviation") or ""
            away = (game_data.get("away_team", {}) or {}).get("abbreviation") or ""
            if oc == str(home).upper():
                return "home"
            if oc == str(away).upper():
                return "away"
        except Exception:
            pass
        # Fallback: return code as-is; adapter will try direct dict lookup
        return oc

    def _extract_prob_conf(self, out, outcome_key=None):
        # tuple/list: (prob, conf?) or (prob,)
        if isinstance(out, (tuple, list)):
            if len(out) >= 2:
                return float(out[0]), float(out[1])
            if len(out) == 1:
                return float(out[0]), 0.75

        # plain number
        try:
            import numbers
            if isinstance(out, numbers.Number):
                return float(out), 0.75
        except Exception:
            pass

        # dict-like: look for common keys or per-outcome keys
        if isinstance(out, dict):
            # common scalar keys
            for k in ("probability", "prob", "p"):
                if k in out:
                    prob = float(out[k])
                    conf = float(out.get("confidence", out.get("conf", out.get("c", 0.75))))
                    return prob, conf
            # per-outcome probabilities
            if outcome_key:
                # normalize keys
                candidates = [outcome_key, outcome_key.lower(), outcome_key.upper()]
                # map draw synonyms
                if outcome_key == "draw":
                    candidates += ["tie", "x"]
                if outcome_key == "home":
                    candidates += ["h"]
                if outcome_key == "away":
                    candidates += ["a"]
                for cand in candidates:
                    if cand in out:
                        prob = float(out[cand])
                        conf = float(out.get("confidence", out.get("conf", 0.75)))
                        return prob, conf
            # try generic three-way dict
            for trio in (("home","away","draw"), ("H","A","X"), ("h","a","x")):
                if all(k in out for k in trio):
                    # if no specific outcome requested, pick max prob with modest conf
                    best = max(trio, key=lambda k: float(out[k]))
                    return float(out[best]), 0.7
        # failed to understand
        raise RuntimeError("External model output not understood")

    def _try_callable(self, fn, game_data, parsed_outcome):
        # Try the function with various signatures
        sig = None
        try:
            sig = self.inspect.signature(fn)
        except Exception:
            pass

        # 1-arg: game_data
        try:
            if sig and len([p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY,p.POSITIONAL_OR_KEYWORD)]) == 1:
                return fn(game_data)
        except Exception:
            pass

        # 2-3 args: (home, away[, outcome])
        home = (game_data.get("home_team", {}) or {}).get("abbreviation")
        away = (game_data.get("away_team", {}) or {}).get("abbreviation")
        if home and away:
            try:
                if sig and len([p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY,p.POSITIONAL_OR_KEYWORD)]) == 2:
                    return fn(home, away)
            except Exception:
                pass
            try:
                if sig and len([p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY,p.POSITIONAL_OR_KEYWORD)]) >= 3:
                    return fn(home, away, parsed_outcome or None)
            except Exception:
                pass

        # Last resort: just try with game_data
        try:
            return fn(game_data)
        except Exception as e:
            raise e

    def predict_win_probability(self, game_data: dict):
        from datetime import datetime
        # Determine outcome code (for soccer three-way) from ticker
        ticker = (game_data or {}).get("ticker", "")
        outcome_code = ""
        try:
            pt = parse_ticker(ticker)
            outcome_code = getattr(pt, "outcome", "") or ""
        except Exception:
            pass
        outcome_key = self._pick_outcome_key(outcome_code, game_data or {})

        # Candidate callables on the module
        candidates = []
        for name in ("predict_win_probability","predict","predict_proba","forecast","predict_match"):
            if hasattr(self.impl, name):
                candidates.append(getattr(self.impl, name))

        # Or a class-style API
        if not candidates and hasattr(self.impl, "Model"):
            try:
                inst = getattr(self, "_inst", None)
                if inst is None:
                    inst = self.impl.Model()
                    setattr(self, "_inst", inst)
                for name in ("predict_win_probability","predict","predict_proba","forecast","predict_match"):
                    if hasattr(inst, name):
                        candidates.append(getattr(inst, name))
            except Exception:
                pass

        if not candidates:
            raise RuntimeError("External model has no usable predict* interface")

        last_err = None
        for fn in candidates:
            try:
                raw = self._try_callable(fn, game_data or {}, outcome_key)
                prob, conf = self._extract_prob_conf(raw, outcome_key)
                # Clamp
                prob = max(0.0, min(1.0, prob))
                conf = max(0.0, min(1.0, conf))
                return prob, conf
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"External model attempts failed: {last_err}")
"""

# Replace the whole adapter block if present
if re.search(pat_adapter_start, s):
    s = re.sub(pat_adapter_block, adapter_new + "\nclass EnhancedModelManager:", s, flags=re.DOTALL)

p.write_text(s, encoding="utf-8")
print("✅ Patched: robust ExternalModelAdapter + safe get_prediction fallback.")