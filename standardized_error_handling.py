#!/usr/bin/env python3
"""
Standardized API Error Handling - FIXED VERSION
===============================================
Unified error handling across all API calls with proper retry logic
"""

import asyncio
import aiohttp
import logging
import time
from typing import Dict, Any, Optional, Callable, Type, List  # Added List import
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
import json


class ErrorType(Enum):
    """Standardized error types"""
    AUTH_ERROR = "authentication_failed"
    RATE_LIMIT = "rate_limited" 
    NETWORK_ERROR = "network_error"
    SERVER_ERROR = "server_error"
    VALIDATION_ERROR = "validation_error"
    MARKET_CLOSED = "market_closed"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class APIError(Exception):
    """Standardized API error with context"""
    error_type: ErrorType
    message: str
    status_code: Optional[int] = None
    endpoint: Optional[str] = None
    request_id: Optional[str] = None
    retry_after: Optional[int] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def __str__(self):
        return f"{self.error_type.value}: {self.message} (status={self.status_code})"
    
    def is_retryable(self) -> bool:
        """Check if this error type should trigger a retry"""
        retryable_types = {
            ErrorType.RATE_LIMIT,
            ErrorType.NETWORK_ERROR,
            ErrorType.SERVER_ERROR,
            ErrorType.TIMEOUT_ERROR
        }
        return self.error_type in retryable_types


class RetryConfig:
    """Retry configuration"""
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        backoff_factor: float = 2.0,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt"""
        delay = self.base_delay * (self.backoff_factor ** (attempt - 1))
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add ±25% jitter
            import random
            jitter_factor = 0.75 + (random.random() * 0.5)
            delay *= jitter_factor
        
        return delay


class APIErrorHandler:
    """Centralized API error handling and mapping"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._error_mappings = self._build_error_mappings()
    
    def _build_error_mappings(self) -> Dict[int, ErrorType]:
        """Map HTTP status codes to error types"""
        return {
            400: ErrorType.VALIDATION_ERROR,
            401: ErrorType.AUTH_ERROR,
            403: ErrorType.AUTH_ERROR,
            404: ErrorType.VALIDATION_ERROR,
            408: ErrorType.TIMEOUT_ERROR,
            429: ErrorType.RATE_LIMIT,
            500: ErrorType.SERVER_ERROR,
            502: ErrorType.SERVER_ERROR,
            503: ErrorType.SERVER_ERROR,
            504: ErrorType.TIMEOUT_ERROR
        }
    
    def parse_response_error(
        self,
        response: aiohttp.ClientResponse,
        endpoint: str,
        response_text: str = ""
    ) -> APIError:
        """Parse HTTP response into standardized error"""
        
        status_code = response.status
        error_type = self._error_mappings.get(status_code, ErrorType.UNKNOWN_ERROR)
        
        # Extract error message
        message = self._extract_error_message(response_text, status_code)
        
        # Extract retry-after header for rate limits
        retry_after = None
        if status_code == 429:
            retry_after_header = response.headers.get('Retry-After')
            if retry_after_header:
                try:
                    retry_after = int(retry_after_header)
                except ValueError:
                    pass
        
        # Extract request ID if available
        request_id = response.headers.get('X-Request-ID') or response.headers.get('Request-Id')
        
        return APIError(
            error_type=error_type,
            message=message,
            status_code=status_code,
            endpoint=endpoint,
            request_id=request_id,
            retry_after=retry_after
        )
    
    def _extract_error_message(self, response_text: str, status_code: int) -> str:
        """Extract human-readable error message"""
        
        # Try to parse JSON error response
        try:
            error_data = json.loads(response_text)
            
            # Common error message fields
            for field in ['error', 'message', 'detail', 'error_description']:
                if field in error_data:
                    return str(error_data[field])
                    
            # Nested error structures
            if 'error' in error_data and isinstance(error_data['error'], dict):
                return error_data['error'].get('message', 'Unknown error')
                
        except (json.JSONDecodeError, TypeError):
            pass
        
        # Fallback to HTTP status messages
        status_messages = {
            400: "Bad request - check parameters",
            401: "Authentication failed - check credentials",
            403: "Access forbidden - insufficient permissions",
            404: "Resource not found",
            408: "Request timeout",
            429: "Rate limit exceeded",
            500: "Internal server error",
            502: "Bad gateway",
            503: "Service unavailable",
            504: "Gateway timeout"
        }
        
        return status_messages.get(status_code, f"HTTP {status_code} error")
    
    def handle_network_error(self, error: Exception, endpoint: str) -> APIError:
        """Handle network-level errors"""
        
        if isinstance(error, asyncio.TimeoutError):
            return APIError(
                error_type=ErrorType.TIMEOUT_ERROR,
                message=f"Request timeout for {endpoint}",
                endpoint=endpoint
            )
        elif isinstance(error, aiohttp.ClientConnectorError):
            return APIError(
                error_type=ErrorType.NETWORK_ERROR,
                message=f"Connection failed for {endpoint}: {error}",
                endpoint=endpoint
            )
        else:
            return APIError(
                error_type=ErrorType.NETWORK_ERROR,
                message=f"Network error for {endpoint}: {error}",
                endpoint=endpoint
            )


class ResilientAPIClient:
    """Base class for resilient API clients with standardized error handling"""
    
    def __init__(
        self,
        base_url: str,
        timeout: int = 30,
        retry_config: Optional[RetryConfig] = None
    ):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.retry_config = retry_config or RetryConfig()
        self.error_handler = APIErrorHandler()
        self.logger = logging.getLogger(__name__)
        
        # Session will be created when needed
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def close(self):
        """Close HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def make_request(
        self,
        method: str,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        retry_config: Optional[RetryConfig] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with standardized error handling and retries"""
        
        retry_config = retry_config or self.retry_config
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        last_error = None
        
        for attempt in range(1, retry_config.max_attempts + 1):
            try:
                # Log attempt
                self.logger.debug(f"API request attempt {attempt}/{retry_config.max_attempts}: {method} {endpoint}")
                
                # Make request
                session = await self._get_session()
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    json=json_data
                ) as response:
                    
                    # Read response text
                    response_text = await response.text()
                    
                    # Handle successful responses
                    if 200 <= response.status < 300:
                        try:
                            return json.loads(response_text) if response_text else {}
                        except json.JSONDecodeError:
                            return {"response": response_text}
                    
                    # Handle error responses
                    error = self.error_handler.parse_response_error(
                        response, endpoint, response_text
                    )
                    
                    # Log error details
                    self.logger.warning(
                        f"API error on attempt {attempt}: {error} "
                        f"(endpoint={endpoint}, request_id={error.request_id})"
                    )
                    
                    # Check if we should retry
                    if attempt < retry_config.max_attempts and error.is_retryable():
                        # Use retry-after header if available
                        delay = error.retry_after or retry_config.get_delay(attempt)
                        self.logger.info(f"Retrying in {delay:.1f} seconds...")
                        await asyncio.sleep(delay)
                        last_error = error
                        continue
                    
                    # No more retries, raise the error
                    raise error
                    
            except APIError:
                # Re-raise API errors directly
                raise
            except Exception as e:
                # Handle network and other errors
                error = self.error_handler.handle_network_error(e, endpoint)
                
                self.logger.warning(f"Network error on attempt {attempt}: {error}")
                
                if attempt < retry_config.max_attempts and error.is_retryable():
                    delay = retry_config.get_delay(attempt)
                    self.logger.info(f"Retrying in {delay:.1f} seconds...")
                    await asyncio.sleep(delay)
                    last_error = error
                    continue
                
                raise error
        
        # This shouldn't be reached, but just in case
        raise last_error or APIError(
            error_type=ErrorType.UNKNOWN_ERROR,
            message="Max retries exceeded",
            endpoint=endpoint
        )


class EnhancedKalshiClient(ResilientAPIClient):
    """Enhanced Kalshi client with standardized error handling"""
    
    def __init__(self, key_id: str, private_key_path: str, base_url: str = None):
        base_url = base_url or "https://trading-api.kalshi.com/trading-api/v2"
        super().__init__(base_url)
        
        self.key_id = key_id
        self.private_key_path = private_key_path
        self._auth_token: Optional[str] = None
        self._token_expires: Optional[datetime] = None
    
    async def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers with token refresh"""
        
        # Check if token needs refresh
        if (not self._auth_token or 
            not self._token_expires or 
            datetime.now() >= self._token_expires - timedelta(minutes=5)):
            
            await self._refresh_token()
        
        return {
            "Authorization": f"Bearer {self._auth_token}",
            "Content-Type": "application/json"
        }
    
    async def _refresh_token(self):
        """Refresh authentication token"""
        try:
            # Import your existing auth logic
            from kalshi_auth_fixed import KalshiAuth
            
            auth = KalshiAuth(self.key_id, self.private_key_path)
            self._auth_token = auth.get_access_token()
            self._token_expires = datetime.now() + timedelta(hours=1)  # Assume 1-hour expiry
            
            self.logger.info("Authentication token refreshed")
            
        except Exception as e:
            raise APIError(
                error_type=ErrorType.AUTH_ERROR,
                message=f"Token refresh failed: {e}",
                endpoint="auth/login"
            )
    
    async def get_markets(self) -> List[Dict[str, Any]]:
        """Get active markets with error handling"""
        headers = await self._get_auth_headers()
        
        try:
            response = await self.make_request("GET", "markets", headers=headers)
            return response.get("markets", [])
            
        except APIError as e:
            # Add context for markets endpoint
            if e.error_type == ErrorType.AUTH_ERROR:
                self.logger.error("Authentication failed - check API credentials")
            elif e.error_type == ErrorType.RATE_LIMIT:
                self.logger.warning(f"Rate limited - retry after {e.retry_after}s")
            
            raise e
    
    async def place_order(
        self,
        ticker: str,
        side: str,
        quantity: int,
        price: float
    ) -> Dict[str, Any]:
        """Place order with enhanced error handling"""
        headers = await self._get_auth_headers()
        
        order_data = {
            "ticker": ticker,
            "client_order_id": f"order_{int(time.time())}",
            "side": side,
            "action": "buy",
            "count": quantity,
            "type": "limit",
            "yes_price": int(price * 100) if side == "yes" else None,
            "no_price": int(price * 100) if side == "no" else None
        }
        
        try:
            response = await self.make_request("POST", "orders", headers=headers, json_data=order_data)
            self.logger.info(f"Order placed successfully: {ticker} {side} {quantity}@{price}")
            return response
            
        except APIError as e:
            # Enhanced error context for orders
            if e.error_type == ErrorType.VALIDATION_ERROR:
                self.logger.error(f"Order validation failed: {e.message}")
            elif e.error_type == ErrorType.INSUFFICIENT_FUNDS:
                self.logger.error("Insufficient funds for order")
            elif e.error_type == ErrorType.MARKET_CLOSED:
                self.logger.error(f"Market closed: {ticker}")
            
            # Add order context to error
            e.message = f"Order failed ({ticker} {side} {quantity}@{price}): {e.message}"
            raise e


# Usage example and testing
async def test_error_handling():
    """Test the error handling system"""
    print("🔧 Testing standardized error handling...")
    
    # Test with invalid credentials
    try:
        client = EnhancedKalshiClient("invalid_key", "./nonexistent.pem")
        await client.get_markets()
    except APIError as e:
        print(f"✅ Caught expected error: {e.error_type.value}")
        print(f"   Message: {e.message}")
        print(f"   Retryable: {e.is_retryable()}")
    except Exception as e:
        print(f"✅ Caught error (expected): {e}")
    
    print("✅ Error handling system ready!")


if __name__ == "__main__":
    asyncio.run(test_error_handling())