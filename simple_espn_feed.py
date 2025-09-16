#!/usr/bin/env python3
"""
Simple ESPN Feed Integration
============================
Works with your existing bot - no external dependencies
"""

import aiohttp
import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional

class SimpleESPNFeed:
    """Simple ESPN integration for your existing bot"""
    
    def __init__(self):
        self.base_url = "https://site.api.espn.com/apis/site/v2/sports"
        self.cache = {}
        self.cache_ttl = 60  # Cache for 60 seconds
        self.logger = logging.getLogger(__name__)
        
    async def get_game_state(self, ticker: str) -> Optional[Dict]:
        """Get live game state from ESPN for a Kalshi ticker"""
        
        # Parse sport from ticker
        sport_info = self._parse_ticker_for_espn(ticker)
        if not sport_info:
            return None
            
        try:
            # Check cache first
            cache_key = f"{sport_info['sport']}_{sport_info['league']}"
            if cache_key in self.cache:
                cached_time, cached_data = self.cache[cache_key]
                if (datetime.now().timestamp() - cached_time) < self.cache_ttl:
                    return self._find_game_in_data(cached_data, sport_info['teams'])
            
            # Fetch fresh data from ESPN
            url = f"{self.base_url}/{sport_info['espn_sport']}/{sport_info['league']}/scoreboard"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Cache the data
                        self.cache[cache_key] = (datetime.now().timestamp(), data)
                        
                        # Find specific game
                        return self._find_game_in_data(data, sport_info['teams'])
                        
        except Exception as e:
            self.logger.debug(f"ESPN API error for {ticker}: {e}")
            return None
    
    def _parse_ticker_for_espn(self, ticker: str) -> Optional[Dict]:
        """Parse Kalshi ticker to get ESPN-compatible info"""
        
        # MLB parsing
        if 'MLB' in ticker:
            # Example: MLB_NYY_BOS_H1 -> Yankees vs Red Sox
            parts = ticker.split('_')
            if len(parts) >= 3:
                return {
                    'sport': 'mlb',
                    'league': 'mlb', 
                    'espn_sport': 'baseball',
                    'teams': [parts[1], parts[2]]  # Team codes
                }
        
        # NFL parsing  
        elif 'NFL' in ticker:
            # Example: NFL_KC_BUF_H1 -> Chiefs vs Bills
            parts = ticker.split('_')
            if len(parts) >= 3:
                return {
                    'sport': 'nfl',
                    'league': 'nfl',
                    'espn_sport': 'americanfootball', 
                    'teams': [parts[1], parts[2]]
                }
        
        return None
    
    def _find_game_in_data(self, espn_data: Dict, team_codes: list) -> Optional[Dict]:
        """Find specific game in ESPN data"""
        
        events = espn_data.get('events', [])
        
        for event in events:
            competitors = event.get('competitions', [{}])[0].get('competitors', [])
            
            if len(competitors) >= 2:
                # Get team abbreviations
                team_abbrevs = []
                for comp in competitors:
                    team = comp.get('team', {})
                    abbrev = team.get('abbreviation', '').upper()
                    team_abbrevs.append(abbrev)
                
                # Check if this matches our target teams
                if any(code.upper() in team_abbrevs for code in team_codes):
                    return self._parse_espn_game(event)
        
        return None
    
    def _parse_espn_game(self, event: Dict) -> Dict:
        """Parse ESPN game data into standard format"""
        
        competition = event.get('competitions', [{}])[0]
        competitors = competition.get('competitors', [])
        status = event.get('status', {})
        
        # Get scores and teams
        home_team = away_team = None
        home_score = away_score = 0
        
        for comp in competitors:
            team = comp.get('team', {})
            score = int(comp.get('score', 0))
            
            if comp.get('homeAway') == 'home':
                home_team = team.get('abbreviation', 'HOME')
                home_score = score
            else:
                away_team = team.get('abbreviation', 'AWAY') 
                away_score = score
        
        # Get game state
        period = status.get('period', 1)
        clock = status.get('displayClock', '0:00')
        game_status = status.get('type', {}).get('description', 'Unknown')
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'period': period,
            'clock': clock,
            'status': game_status,
            'espn_data': True  # Flag to indicate this came from ESPN
        }

# Test function
async def test_espn_feed():
    """Test the ESPN feed"""
    feed = SimpleESPNFeed()
    
    # Test with mock ticker
    result = await feed.get_game_state("NFL_KC_BUF_H1")
    print(f"ESPN test result: {result}")

if __name__ == "__main__":
    asyncio.run(test_espn_feed())