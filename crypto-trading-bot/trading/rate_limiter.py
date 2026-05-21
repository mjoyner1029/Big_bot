"""
Rate Limiter for Exchange API Calls
Prevents API bans by enforcing rate limits with exponential backoff.
"""
import time
import logging
from collections import deque
from threading import Lock
from typing import Optional


class RateLimiter:
    """Thread-safe rate limiter for exchange APIs"""
    
    def __init__(self, max_calls: int, period_seconds: float, name: str = "API"):
        """
        Initialize rate limiter
        
        Args:
            max_calls: Maximum number of calls allowed in period
            period_seconds: Time window in seconds
            name: Name for logging purposes
        """
        self.max_calls = max_calls
        self.period = period_seconds
        self.name = name
        self.calls = deque()
        self.lock = Lock()
        self.total_waits = 0
        self.total_wait_time = 0.0
        
    def wait_if_needed(self) -> float:
        """
        Check rate limit and wait if necessary
        
        Returns:
            float: Time waited in seconds (0 if no wait needed)
        """
        with self.lock:
            now = time.time()
            
            # Remove old calls outside the time window
            while self.calls and self.calls[0] < now - self.period:
                self.calls.popleft()
            
            # Check if we've hit the limit
            if len(self.calls) >= self.max_calls:
                # Calculate how long to wait
                oldest_call = self.calls[0]
                sleep_time = oldest_call + self.period - now
                
                if sleep_time > 0:
                    self.total_waits += 1
                    self.total_wait_time += sleep_time
                    
                    logging.debug(
                        f"[RateLimit] {self.name} limit reached "
                        f"({len(self.calls)}/{self.max_calls}). "
                        f"Waiting {sleep_time:.2f}s"
                    )
                    time.sleep(sleep_time)
                    now = time.time()
                    
                    # Clean up old calls after waiting
                    while self.calls and self.calls[0] < now - self.period:
                        self.calls.popleft()
                    
                    # Record this call
                    self.calls.append(now)
                    return sleep_time
            
            # Record this call
            self.calls.append(now)
            return 0.0
    
    def get_stats(self) -> dict:
        """Get rate limiter statistics"""
        with self.lock:
            return {
                "name": self.name,
                "max_calls": self.max_calls,
                "period_seconds": self.period,
                "current_calls": len(self.calls),
                "total_waits": self.total_waits,
                "total_wait_time": self.total_wait_time,
            }
    
    def reset_stats(self):
        """Reset statistics counters"""
        with self.lock:
            self.total_waits = 0
            self.total_wait_time = 0.0


class RetryWithBackoff:
    """Exponential backoff retry logic for API calls"""
    
    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
        backoff_factor: float = 2.0
    ):
        """
        Initialize retry handler
        
        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds
            max_delay: Maximum delay cap in seconds
            backoff_factor: Multiplier for each retry
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
    
    def execute(self, func, *args, **kwargs):
        """
        Execute function with exponential backoff retry
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Result of function call
            
        Raises:
            Exception: If all retries exhausted
        """
        last_exception = None
        delay = self.initial_delay
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                # Don't retry on certain errors
                error_str = str(e).lower()
                if any(x in error_str for x in ["invalid", "unauthorized", "forbidden", "not found"]):
                    logging.error(f"[Retry] Non-retryable error: {e}")
                    raise
                
                if attempt < self.max_retries:
                    logging.warning(
                        f"[Retry] Attempt {attempt + 1}/{self.max_retries} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay = min(delay * self.backoff_factor, self.max_delay)
                else:
                    logging.error(
                        f"[Retry] All {self.max_retries} retry attempts exhausted"
                    )
        
        raise last_exception


# Global rate limiters for different exchanges
_coinbase_limiter = RateLimiter(max_calls=10, period_seconds=1.0, name="Coinbase")
_alpaca_limiter = RateLimiter(max_calls=200, period_seconds=60.0, name="Alpaca")
_yfinance_limiter = RateLimiter(max_calls=2000, period_seconds=3600.0, name="yFinance")
_anthropic_limiter = RateLimiter(max_calls=50, period_seconds=60.0, name="Anthropic")

# Global retry handler
_default_retry = RetryWithBackoff(max_retries=3, initial_delay=1.0, max_delay=30.0)


def get_coinbase_limiter() -> RateLimiter:
    """Get the Coinbase API rate limiter"""
    return _coinbase_limiter


def get_alpaca_limiter() -> RateLimiter:
    """Get the Alpaca API rate limiter"""
    return _alpaca_limiter


def get_yfinance_limiter() -> RateLimiter:
    """Get the yFinance API rate limiter"""
    return _yfinance_limiter


def get_anthropic_limiter() -> RateLimiter:
    """Get the Anthropic API rate limiter"""
    return _anthropic_limiter


def get_default_retry() -> RetryWithBackoff:
    """Get the default retry handler"""
    return _default_retry


def log_rate_limiter_stats():
    """Log statistics for all rate limiters"""
    logging.info("="*60)
    logging.info("RATE LIMITER STATISTICS")
    logging.info("="*60)
    
    for limiter in [_coinbase_limiter, _alpaca_limiter, _yfinance_limiter, _anthropic_limiter]:
        stats = limiter.get_stats()
        logging.info(
            f"{stats['name']:12s}: {stats['current_calls']:3d}/{stats['max_calls']:3d} calls | "
            f"Waits: {stats['total_waits']:4d} | "
            f"Total wait time: {stats['total_wait_time']:.1f}s"
        )
    
    logging.info("="*60)
