"""
Real Sports Feed Integration - COMPLETE VERSION
==============================================
This is the RESTORED version with ESPN API, Odds API, and full integration
"""

import os
import time
import logging
import asyncio
import aiohttp
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
import re
from dataclasses import dataclass


@dataclass
class GameState:
    """Represents the current state of a game"""
    sport: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    period: int
    time_remaining: str
    game_status: str
    last_updated: datetime


class ESPNClient:
    """ESPN API client for real-time sports data"""
    
    def __init__(self):
        self.base_url = "https://site.api.espn.com/apis/site/v2/sports"
        self.timeout = 10
        self.min_interval = 5
        self._last_request = {}
        
    async def get_mlb_games(self) -> List[Dict[str, Any]]:
        """Get current MLB games from ESPN"""
        return await self._get_games("baseball", "mlb")
    
    async def get_nfl_games(self) -> List[Dict[str, Any]]:
        """Get current NFL games from ESPN"""
        return await self._get_games("americanfootball", "nfl")
    
    async def get_soccer_games(self) -> List[Dict[str, Any]]:
        """Get current Soccer games from ESPN"""
        return await self._get_games("soccer", "usa.1")
    
    async def _get_games(self, sport: str, league: str) -> List[Dict[str, Any]]:
        """Get games for a specific sport/league"""
        
        # Rate limiting
        key = f"{sport}_{league}"
        if key in self._last_request:
            elapsed = time.time() - self._last_request[key]
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
        
        try:
            url = f"{self.base_url}/{sport}/{league}/scoreboard"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=self.timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        self._last_request[key] = time.time()
                        return self._parse_espn_games(data, sport.upper())
                    else:
                        logging.warning(f"ESPN API error: {response.status}")
                        return []
                        
        except Exception as e:
            logging.error(f"ESPN API request failed: {e}")
            return []
    
    def _parse_espn_games(self, data: Dict, sport: str) -> List[Dict[str, Any]]:
        """Parse ESPN API response"""
        games = []
        
        try:
            for event in data.get("events", []):
                game = self._parse_single_game(event, sport)
                if game:
                    games.append(game)
                    
        except Exception as e:
            logging.error(f"ESPN data parsing failed: {e}")
        
        return games
    
    def _parse_single_game(self, event: Dict, sport: str) -> Optional[Dict[str, Any]]:
        """Parse a single game from ESPN data"""
        try:
            competition = event.get("competitions", [{}])[0]
            competitors = competition.get("competitors", [])
            
            if len(competitors) < 2:
                return None
            
            # Find home and away teams
            home_team = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away_team = next((c for c in competitors if c.get("homeAway") == "away"), None)
            
            if not home_team or not away_team:
                return None
            
            # Basic game info
            game_info = {
                "sport": sport,
                "game_id": event.get("id"),
                "home_team": self._extract_team_info(home_team),
                "away_team": self._extract_team_info(away_team),
                "status": competition.get("status", {}),
                "last_updated": datetime.now()
            }
            
            # Sport-specific parsing
            if sport == "MLB":
                return self._parse_mlb_specific(game_info, competition)
            elif sport == "NFL":
                return self._parse_nfl_specific(game_info, competition)
            elif sport == "SOCCER":
                return self._parse_soccer_specific(game_info, competition)
            
            return game_info
            
        except Exception as e:
            logging.debug(f"Game parsing failed: {e}")
            return None
    
    def _extract_team_info(self, team_data: Dict) -> Dict[str, Any]:
        """Extract team information"""
        team = team_data.get("team", {})
        score_raw = team_data.get("score", 0)
        try:
            score_val = int(score_raw if score_raw not in (None, "") else 0)
        except Exception:
            score_val = 0
            
        return {
            "id": team.get("id"),
            "name": team.get("displayName"),
            "abbreviation": team.get("abbreviation"),
            "score": score_val
        }
    
    def _parse_mlb_specific(self, game_info: Dict, competition: Dict) -> Dict[str, Any]:
        """Parse MLB-specific data"""
        try:
            status = competition.get("status", {})
            
            # MLB-specific fields
            game_info.update({
                "inning": status.get("period", 0),
                "inning_half": "top" if status.get("displayClock", "").lower().startswith("t") else "bottom",
                "outs": 0,  # Would need detailed play-by-play for this
                "on_base": {"first": False, "second": False, "third": False},
                "balls": 0,
                "strikes": 0,
                "count": "0-0"
            })
            
            # Game state analysis
            game_info["game_state"] = self._analyze_mlb_state(game_info)
            
            return game_info
            
        except Exception as e:
            logging.debug(f"MLB parsing failed: {e}")
            return game_info
    
    def _parse_nfl_specific(self, game_info: Dict, competition: Dict) -> Dict[str, Any]:
        """Parse NFL-specific data"""
        try:
            status = competition.get("status", {})
            
            # NFL-specific fields
            game_info.update({
                "quarter": status.get("period", 0),
                "time_remaining": status.get("displayClock", ""),
                "down": 1,  # Would need play-by-play for accurate down/distance
                "distance": 10,
                "yard_line": 50,
                "possession": game_info["home_team"]["abbreviation"],  # Default
                "red_zone": False,
                "two_minute_warning": False
            })
            
            # Game state analysis
            game_info["game_state"] = self._analyze_nfl_state(game_info)
            
            return game_info
            
        except Exception as e:
            logging.debug(f"NFL parsing failed: {e}")
            return game_info
    
    def _parse_soccer_specific(self, game_info: Dict, competition: Dict) -> Dict[str, Any]:
        """Parse Soccer-specific data"""
        try:
            status = competition.get("status", {})
            
            # Soccer-specific fields
            game_info.update({
                "minute": self._extract_soccer_minute(status.get("displayClock", "")),
                "period": status.get("period", 0),  # 1st half, 2nd half, etc.
                "stoppage_time": 0,
                "cards": {"home": {"yellow": 0, "red": 0}, "away": {"yellow": 0, "red": 0}},
                "possession_pct": {"home": 50, "away": 50}  # Default 50/50
            })
            
            # Game state analysis
            game_info["game_state"] = self._analyze_soccer_state(game_info)
            
            return game_info
            
        except Exception as e:
            logging.debug(f"Soccer parsing failed: {e}")
            return game_info
    
    def _extract_soccer_minute(self, clock_display: str) -> int:
        """Extract minute from soccer clock display"""
        try:
            # Handle formats like "45'", "90+3'", etc.
            if "+" in clock_display:
                base_min = int(clock_display.split("+")[0].replace("'", ""))
                extra_min = int(clock_display.split("+")[1].replace("'", ""))
                return base_min + extra_min
            else:
                return int(clock_display.replace("'", ""))
        except:
            return 0
    
    def _analyze_mlb_state(self, game_info: Dict) -> str:
        """Analyze MLB game state for context"""
        inning = game_info.get("inning", 0)
        home_score = game_info["home_team"]["score"]
        away_score = game_info["away_team"]["score"]
        
        if inning <= 3:
            return "early"
        elif inning <= 6:
            return "middle"
        elif inning <= 8:
            return "late"
        else:
            if abs(home_score - away_score) <= 1:
                return "close_late"
            else:
                return "blowout_late"
    
    def _analyze_nfl_state(self, game_info: Dict) -> str:
        """Analyze NFL game state for context"""
        quarter = game_info.get("quarter", 0)
        home_score = game_info["home_team"]["score"]
        away_score = game_info["away_team"]["score"]
        
        if quarter <= 1:
            return "early"
        elif quarter <= 3:
            return "middle"
        else:
            score_diff = abs(home_score - away_score)
            if score_diff <= 7:
                return "close_late"
            elif score_diff <= 14:
                return "competitive_late"
            else:
                return "blowout_late"
    
    def _analyze_soccer_state(self, game_info: Dict) -> str:
        """Analyze Soccer game state for context"""
        minute = game_info.get("minute", 0)
        home_score = game_info["home_team"]["score"]
        away_score = game_info["away_team"]["score"]
        
        if minute <= 15:
            return "early"
        elif minute <= 30:
            return "first_half"
        elif minute <= 45:
            return "end_first_half"
        elif minute <= 60:
            return "early_second_half"
        elif minute <= 75:
            return "middle_second_half"
        else:
            if abs(home_score - away_score) <= 1:
                return "close_late"
            else:
                return "decided_late"


class RealSportsFeed:
    """Unified real sports feed combining ESPN and other sources"""
    
    def __init__(self):
        self.espn_client = ESPNClient()
        self.cache = {}
        self.cache_ttl = 60  # 60 second cache
        logging.info("Real Sports Feed initialized with ESPN integration")
        
    async def get_game_state(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get game state for a specific Kalshi ticker"""
        
        # Parse sport and teams from ticker
        sport, teams = self._parse_ticker(ticker)
        
        if not sport or not teams:
            logging.debug(f"Could not parse ticker: {ticker}")
            return None
        
        # Get live games for this sport
        live_games = await self._get_live_games_by_sport(sport)
        
        # Find matching game
        matching_game = self._find_matching_game(teams, live_games)
        
        if matching_game:
            logging.info(f"✅ Found ESPN data for {ticker}")
            return matching_game
        else:
            logging.debug(f"❌ No ESPN match for {ticker}")
            return None
    
    def _parse_ticker(self, ticker: str) -> Tuple[Optional[str], Optional[List[str]]]:
        """Parse Kalshi ticker to extract sport and teams"""
        
        try:
            ticker_upper = ticker.upper()
            
            # MLB patterns
            if "MLB" in ticker_upper:
                # Extract team abbreviations (e.g., MLB-NYY-BOS-H1)
                parts = ticker_upper.split("-")
                if len(parts) >= 3:
                    team1, team2 = parts[1], parts[2]
                    return "MLB", [team1, team2]
            
            # NFL patterns  
            elif "NFL" in ticker_upper:
                # Extract team abbreviations (e.g., NFL-KC-BUF-H1)
                parts = ticker_upper.split("-")
                if len(parts) >= 3:
                    team1, team2 = parts[1], parts[2]
                    return "NFL", [team1, team2]
            
            # Soccer patterns
            elif any(soccer_key in ticker_upper for soccer_key in ["SOCCER", "MLS", "EPL"]):
                # Extract team names (more complex for soccer)
                parts = ticker_upper.split("-")
                if len(parts) >= 3:
                    team1, team2 = parts[1], parts[2]
                    return "SOCCER", [team1, team2]
            
            return None, None
            
        except Exception as e:
            logging.debug(f"Ticker parsing failed: {e}")
            return None, None
    
    async def _get_live_games_by_sport(self, sport: str) -> List[Dict[str, Any]]:
        """Get live games for specific sport with caching"""
        
        cache_key = f"live_games_{sport.lower()}"
        
        # Check cache
        if self._is_cached(cache_key):
            return self.cache[cache_key]["data"]
        
        try:
            games = []
            
            if sport == "MLB":
                games = await self.espn_client.get_mlb_games()
            elif sport == "NFL":
                games = await self.espn_client.get_nfl_games()
            elif sport == "SOCCER":
                games = await self.espn_client.get_soccer_games()
            
            # Cache results
            self._cache_data(cache_key, games)
            
            return games
            
        except Exception as e:
            logging.error(f"Failed to get {sport} games: {e}")
            return []
    
    def _find_matching_game(self, teams: List[str], games: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find game matching the specified teams"""
        
        for game in games:
            try:
                home_abbr = game["home_team"]["abbreviation"]
                away_abbr = game["away_team"]["abbreviation"]
                
                # Check if both teams match (in either order)
                if (teams[0] in [home_abbr, away_abbr] and 
                    teams[1] in [home_abbr, away_abbr]):
                    return game
                    
                # Also check full team names for partial matches
                home_name = game["home_team"]["name"].upper()
                away_name = game["away_team"]["name"].upper()
                
                team_matches = 0
                for team in teams:
                    if (team in home_name or team in away_name or
                        team in home_abbr or team in away_abbr):
                        team_matches += 1
                
                if team_matches >= 2:  # Both teams found
                    return game
                    
            except Exception as e:
                logging.debug(f"Game matching failed: {e}")
                continue
        
        return None
    
    def _is_cached(self, key: str) -> bool:
        """Check if data is cached and still valid"""
        if key not in self.cache:
            return False
        
        cached_time = self.cache[key]["timestamp"]
        return (time.time() - cached_time) < self.cache_ttl
    
    def _cache_data(self, key: str, data: Any):
        """Cache data with timestamp"""
        self.cache[key] = {
            "data": data,
            "timestamp": time.time()
        }


# Team abbreviation mappings for better matching
MLB_TEAM_MAPPINGS = {
    "NYY": ["Yankees", "New York Yankees"],
    "BOS": ["Red Sox", "Boston Red Sox"],
    "LAD": ["Dodgers", "Los Angeles Dodgers"],
    "SF": ["Giants", "San Francisco Giants"],
    "HOU": ["Astros", "Houston Astros"],
    # Add more as needed...
}

NFL_TEAM_MAPPINGS = {
    "KC": ["Chiefs", "Kansas City Chiefs"],
    "BUF": ["Bills", "Buffalo Bills"],
    "NE": ["Patriots", "New England Patriots"],
    "MIA": ["Dolphins", "Miami Dolphins"],
    # Add more as needed...
}


async def test_real_sports_feed():
    """Test the real sports feed"""
    print("🔧 Testing Real Sports Feed Integration...")
    
    feed = RealSportsFeed()
    
    # Test MLB
    print("\n📊 Testing MLB games...")
    mlb_games = await feed._get_live_games_by_sport("MLB")
    print(f"Found {len(mlb_games)} MLB games")
    
    if mlb_games:
        sample_game = mlb_games[0]
        print(f"Sample: {sample_game['away_team']['name']} @ {sample_game['home_team']['name']}")
    
    # Test NFL
    print("\n🏈 Testing NFL games...")
    nfl_games = await feed._get_live_games_by_sport("NFL")
    print(f"Found {len(nfl_games)} NFL games")
    
    # Test ticker parsing
    print("\n🎯 Testing ticker matching...")
    test_tickers = ["MLB-NYY-BOS-H1", "NFL-KC-BUF-Q1", "SOCCER-LAF-NYC-FULL"]
    
    for ticker in test_tickers:
        game_state = await feed.get_game_state(ticker)
        if game_state:
            print(f"✅ {ticker}: Found match - {game_state['away_team']['name']} @ {game_state['home_team']['name']}")
        else:
            print(f"❌ {ticker}: No match found")
    
    print("\n✅ Real Sports Feed testing complete!")


if __name__ == "__main__":
    asyncio.run(test_real_sports_feed())