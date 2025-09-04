from fastapi import Request, HTTPException, status
from datetime import datetime, timedelta
from typing import Dict
import asyncio

class RateLimiter:
    def __init__(self):
        # In-memory store for rate limiting (use Redis in production)
        self.requests: Dict[str, List[datetime]] = {}
        self.cleanup_interval = 300  # Clean up every 5 minutes
        asyncio.create_task(self._cleanup_old_requests())
    
    async def check_rate_limit(
        self, 
        identifier: str, 
        max_requests: int, 
        window_seconds: int = 3600
    ) -> bool:
        """Check if identifier has exceeded rate limit"""
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=window_seconds)
        
        # Get existing requests for this identifier
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        # Filter out old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier] 
            if req_time > window_start
        ]
        
        # Check if within limit
        if len(self.requests[identifier]) >= max_requests:
            return False
        
        # Add current request
        self.requests[identifier].append(now)
        return True
    
    async def _cleanup_old_requests(self):
        """Periodically clean up old requests to prevent memory leak"""
        while True:
            await asyncio.sleep(self.cleanup_interval)
            now = datetime.utcnow()
            one_hour_ago = now - timedelta(hours=1)
            
            # Clean up requests older than 1 hour
            for identifier in list(self.requests.keys()):
                self.requests[identifier] = [
                    req_time for req_time in self.requests[identifier] 
                    if req_time > one_hour_ago
                ]
                
                # Remove identifier if no recent requests
                if not self.requests[identifier]:
                    del self.requests[identifier]

# Global rate limiter instance
rate_limiter = RateLimiter()