#!/usr/bin/env python3
"""
Enhanced Configuration Validation
=================================
Replaces your enhanced_config_loader.py with strict validation using Pydantic
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# Install pydantic if not available
try:
    from pydantic import BaseModel, validator, Field, ValidationError
except ImportError:
    print("Installing pydantic...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pydantic"])
    from pydantic import BaseModel, validator, Field, ValidationError

from enum import Enum


class LogLevel(str, Enum):
    """Valid logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO" 
    WARNING = "WARNING"
    ERROR = "ERROR"


class TradingMode(str, Enum):
    """Trading modes"""
    PAPER = "paper"
    LIVE = "live"


class KalshiConfig(BaseModel):
    """Kalshi API configuration with validation"""
    base_url: str = "https://trading-api.kalshi.com/trading-api/v2"
    key_id: Optional[str] = None
    private_key_path: Optional[str] = None
    timeout_seconds: int = Field(default=30, ge=5, le=120)
    max_retries: int = Field(default=3, ge=1, le=10)
    
    @validator('key_id')
    def validate_key_id(cls, v):
        if v and len(v) < 10:
            raise ValueError('Key ID appears invalid (too short)')
        return v
    
    @validator('private_key_path')
    def validate_private_key_path(cls, v):
        if v and not os.path.exists(v):
            raise ValueError(f'Private key file not found: {v}')
        return v


class RiskConfig(BaseModel):
    """Risk management with strict bounds"""
    max_daily_loss_pct: float = Field(default=15.0, gt=0, le=50)
    max_portfolio_risk: float = Field(default=25.0, gt=0, le=100)
    max_per_event_risk: float = Field(default=8.0, gt=0, le=50)
    max_positions: int = Field(default=8, ge=1, le=50)
    
    @validator('max_daily_loss_pct')
    def validate_daily_loss(cls, v):
        if v > 30:
            logging.warning(f'High daily loss limit: {v}%')
        return v
    
    @validator('max_portfolio_risk')
    def validate_portfolio_risk(cls, v, values):
        daily_loss = values.get('max_daily_loss_pct', 15)
        if v < daily_loss:
            raise ValueError('Portfolio risk must be >= daily loss limit')
        return v


class SizingConfig(BaseModel):
    """Position sizing with Kelly validation"""
    kelly_fraction: float = Field(default=0.25, gt=0, le=1.0)
    max_position_size: float = Field(default=100.0, gt=0)
    min_position_size: float = Field(default=5.0, gt=0)
    
    @validator('kelly_fraction')
    def validate_kelly(cls, v):
        if v > 0.5:
            logging.warning(f'High Kelly fraction: {v:.1%} - consider reducing')
        return v
    
    @validator('max_position_size')
    def validate_max_size(cls, v, values):
        min_size = values.get('min_position_size', 5)
        if v <= min_size:
            raise ValueError('Max position size must be > min position size')
        return v


class EdgeConfig(BaseModel):
    """Edge detection parameters"""
    threshold: float = Field(default=0.05, gt=0, le=0.5)
    min_confidence: float = Field(default=0.4, ge=0, le=1.0)
    fee_buffer: float = Field(default=0.01, ge=0, le=0.1)
    
    @validator('threshold')
    def validate_threshold(cls, v):
        if v < 0.02:
            logging.warning(f'Very low edge threshold: {v:.1%}')
        elif v > 0.15:
            logging.warning(f'Very high edge threshold: {v:.1%}')
        return v


class ExitConfig(BaseModel):
    """Exit strategy configuration"""
    take_profit_pct: float = Field(default=25.0, gt=0, le=200)
    stop_loss_pct: float = Field(default=20.0, gt=0, le=100)
    trailing_stop_pct: float = Field(default=10.0, gt=0, le=50)
    max_hold_hours: int = Field(default=72, ge=1, le=168)  # Max 1 week
    
    @validator('stop_loss_pct')
    def validate_stop_loss(cls, v):
        if v > 50:
            logging.warning(f'High stop loss: {v}%')
        return v


class TradingConfig(BaseModel):
    """Trading loop configuration"""
    poll_interval_seconds: int = Field(default=30, ge=5, le=300)
    paper_trading: bool = True
    bankroll: float = Field(default=1000.0, gt=0)
    
    @validator('bankroll')
    def validate_bankroll(cls, v):
        if v < 100:
            logging.warning(f'Low bankroll: ${v}')
        elif v > 1000000:
            logging.warning(f'High bankroll: ${v} - ensure proper risk management')
        return v


class LoggingConfig(BaseModel):
    """Logging configuration"""
    log_level: LogLevel = LogLevel.INFO
    log_file: str = "./logs/institutional_bot.log"
    max_log_size_mb: int = Field(default=100, ge=1, le=1000)
    backup_count: int = Field(default=5, ge=1, le=20)
    
    @validator('log_file')
    def validate_log_file(cls, v):
        # Ensure log directory exists
        log_path = Path(v)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        return v


class InstitutionalConfig(BaseModel):
    """Complete institutional trading configuration with validation"""
    
    # Configuration sections
    kalshi: KalshiConfig = KalshiConfig()
    risk: RiskConfig = RiskConfig()
    sizing: SizingConfig = SizingConfig()
    edge: EdgeConfig = EdgeConfig()
    exits: ExitConfig = ExitConfig()
    trading: TradingConfig = TradingConfig()
    logging: LoggingConfig = LoggingConfig()
    
    # Additional settings
    created_at: datetime = Field(default_factory=datetime.now)
    config_version: str = "1.0"
    
    class Config:
        """Pydantic configuration"""
        validate_assignment = True  # Validate on assignment
        extra = "forbid"  # Reject unknown fields
        
    def model_post_init(self, __context: Any) -> None:
        """Post-initialization validation"""
        # Cross-section validation
        if self.trading.paper_trading:
            # Paper trading - relax some requirements
            if not self.kalshi.key_id:
                logging.info("Paper trading mode - API credentials optional")
        else:
            # Live trading - strict requirements
            if not self.kalshi.key_id or not self.kalshi.private_key_path:
                raise ValueError("Live trading requires API credentials")
            
            # Validate risk for live trading
            if self.risk.max_daily_loss_pct > 20:
                raise ValueError("Daily loss limit too high for live trading")
    
    def get_redacted_dict(self) -> Dict[str, Any]:
        """Get configuration as dict with sensitive fields redacted"""
        config_dict = self.model_dump()
        
        # Redact sensitive fields
        if 'kalshi' in config_dict:
            if config_dict['kalshi'].get('key_id'):
                config_dict['kalshi']['key_id'] = '***REDACTED***'
            if config_dict['kalshi'].get('private_key_path'):
                config_dict['kalshi']['private_key_path'] = '***REDACTED***'
        
        return config_dict
    
    def validate_for_production(self) -> List[str]:
        """Additional validation for production deployment"""
        warnings = []
        
        if self.trading.paper_trading:
            warnings.append("Still in paper trading mode")
        
        if self.risk.max_daily_loss_pct > 10:
            warnings.append(f"High daily loss limit for production: {self.risk.max_daily_loss_pct}%")
        
        if self.sizing.kelly_fraction > 0.3:
            warnings.append(f"Aggressive Kelly fraction: {self.sizing.kelly_fraction:.1%}")
        
        if self.edge.threshold < 0.03:
            warnings.append(f"Low edge threshold may increase trading frequency")
        
        return warnings


class ValidatedConfigLoader:
    """Configuration loader with Pydantic validation"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self.logger = logging.getLogger(__name__)
    
    def _find_config_file(self) -> str:
        """Find configuration file in standard locations"""
        search_paths = [
            "./institutional_config.yaml",
            "./config/institutional_config.yaml",
            "./config.yaml",
            os.path.expanduser("~/.kalshi/config.yaml")
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                return path
        
        return "./institutional_config.yaml"
    
    def load_config(self) -> InstitutionalConfig:
        """Load and validate configuration"""
        try:
            # Load YAML
            config_data = self._load_yaml()
            
            # Merge environment variables
            config_data = self._merge_environment_vars(config_data)
            
            # Create and validate config object
            config = InstitutionalConfig(**config_data)
            
            # Log success with redacted data
            self.logger.info(f"Configuration loaded from {self.config_path}")
            self.logger.debug(f"Config: {config.get_redacted_dict()}")
            
            # Check production readiness
            warnings = config.validate_for_production()
            for warning in warnings:
                self.logger.warning(f"Production check: {warning}")
            
            return config
            
        except ValidationError as e:
            self.logger.error(f"Configuration validation failed:")
            for error in e.errors():
                field = " -> ".join(str(x) for x in error['loc'])
                self.logger.error(f"  {field}: {error['msg']}")
            
            self.logger.info("Using default configuration")
            return InstitutionalConfig()
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return InstitutionalConfig()
    
    def _load_yaml(self) -> Dict[str, Any]:
        """Load YAML file"""
        if not os.path.exists(self.config_path):
            self.logger.warning(f"Config file not found: {self.config_path}")
            return {}
        
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            self.logger.error(f"Invalid YAML in {self.config_path}: {e}")
            return {}
    
    def _merge_environment_vars(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Merge environment variables with priority"""
        
        # Environment variable mappings
        env_mappings = {
            'KALSHI_KEY_ID': ['kalshi', 'key_id'],
            'KALSHI_PRIVATE_KEY_PATH': ['kalshi', 'private_key_path'],
            'BANKROLL': ['trading', 'bankroll'],
            'PAPER_TRADING': ['trading', 'paper_trading'],
            'LOG_LEVEL': ['logging', 'log_level'],
            'MAX_POSITIONS': ['risk', 'max_positions'],
            'DAILY_LOSS_LIMIT': ['risk', 'max_daily_loss_pct']
        }
        
        for env_var, path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Navigate to the correct nested location
                current = config_data
                for key in path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                
                # Convert value to appropriate type
                if env_var in ['BANKROLL', 'DAILY_LOSS_LIMIT']:
                    value = float(value)
                elif env_var in ['MAX_POSITIONS']:
                    value = int(value)
                elif env_var == 'PAPER_TRADING':
                    value = value.lower() in ['true', '1', 'yes']
                
                current[path[-1]] = value
                self.logger.info(f"Using environment variable {env_var}")
        
        return config_data


# Convenience function
def load_validated_config(config_path: Optional[str] = None) -> InstitutionalConfig:
    """Load and validate configuration - main entry point"""
    loader = ValidatedConfigLoader(config_path)
    return loader.load_config()


if __name__ == "__main__":
    # Test the validation
    print("🔍 Testing configuration validation...")
    
    try:
        config = load_validated_config()
        print("✅ Configuration loaded successfully")
        print(f"   Bankroll: ${config.trading.bankroll:,.0f}")
        print(f"   Risk Limit: {config.risk.max_daily_loss_pct}%")
        print(f"   Paper Trading: {config.trading.paper_trading}")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")