"""
Example usage of the retry decorator with exponential backoff.

This example demonstrates how to use the @retry decorator to handle
flaky API calls and network operations gracefully.
"""

import random
from agentunit.core import retry


# Example 1: Basic retry with default settings
@retry()
def fetch_user_data(user_id: int) -> dict:
    """
    Fetch user data from an API with automatic retries.
    
    Uses default settings:
    - max_retries=3
    - base_delay=1.0s
    - exponential_base=2.0
    """
    # Simulated API call that might fail
    if random.random() < 0.3:  # 30% chance of failure
        raise ConnectionError("Network timeout")
    
    return {"id": user_id, "name": "John Doe", "email": "john@example.com"}


# Example 2: Custom retry configuration for critical operations
@retry(
    max_retries=5,
    base_delay=2.0,
    max_delay=30.0,
    exceptions=(ConnectionError, TimeoutError)
)
def critical_database_operation(query: str) -> list:
    """
    Execute a critical database query with aggressive retry policy.
    
    Configuration:
    - Retries up to 5 times
    - Starts with 2s delay
    - Caps delay at 30s
    - Only retries on connection/timeout errors
    """
    # Simulated database operation
    if random.random() < 0.4:  # 40% chance of failure
        raise ConnectionError("Database connection lost")
    
    return [{"result": "data"}]


# Example 3: Quick retries for fast operations
@retry(
    max_retries=10,
    base_delay=0.1,
    max_delay=5.0,
    exponential_base=1.5
)
def quick_cache_lookup(key: str) -> str:
    """
    Look up value in cache with quick retries.
    
    Configuration:
    - Many retries (10) with short delays
    - Starts with 0.1s delay
    - Slower exponential growth (1.5x)
    - Caps at 5s
    """
    # Simulated cache lookup
    if random.random() < 0.2:  # 20% chance of failure
        raise ConnectionError("Cache server unavailable")
    
    return f"value_for_{key}"


# Example 4: Specific exception handling
@retry(
    max_retries=3,
    base_delay=1.0,
    exceptions=(ValueError, KeyError)
)
def parse_api_response(response: dict) -> dict:
    """
    Parse API response with retries only for specific errors.
    
    Only retries on ValueError and KeyError.
    Other exceptions (like TypeError) will raise immediately.
    """
    # Simulated parsing that might fail
    if random.random() < 0.3:
        raise ValueError("Invalid response format")
    
    return {"parsed": True, "data": response}


# Example 5: Using retry in adapter classes
class APIAdapter:
    """Example adapter class using retry decorator."""
    
    @retry(max_retries=3, base_delay=1.0, exceptions=(ConnectionError,))
    def call_llm(self, prompt: str) -> str:
        """
        Call LLM API with automatic retries on connection errors.
        """
        # Simulated LLM API call
        if random.random() < 0.25:
            raise ConnectionError("LLM API unavailable")
        
        return f"Response to: {prompt}"
    
    @retry(max_retries=5, base_delay=2.0, max_delay=60.0)
    def fetch_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Fetch embeddings with more aggressive retry policy.
        """
        # Simulated embedding API call
        if random.random() < 0.3:
            raise ConnectionError("Embedding service timeout")
        
        return [[0.1, 0.2, 0.3] for _ in texts]


def main():
    """Demonstrate retry decorator usage."""
    
    print("Example 1: Basic retry")
    try:
        user = fetch_user_data(123)
        print(f"✓ Fetched user: {user}")
    except ConnectionError as e:
        print(f"✗ Failed after retries: {e}")
    
    print("\nExample 2: Critical operation with custom config")
    try:
        results = critical_database_operation("SELECT * FROM users")
        print(f"✓ Query succeeded: {results}")
    except ConnectionError as e:
        print(f"✗ Failed after retries: {e}")
    
    print("\nExample 3: Quick cache lookup")
    try:
        value = quick_cache_lookup("user:123")
        print(f"✓ Cache hit: {value}")
    except ConnectionError as e:
        print(f"✗ Cache unavailable: {e}")
    
    print("\nExample 4: Specific exception handling")
    try:
        parsed = parse_api_response({"status": "ok"})
        print(f"✓ Parsed response: {parsed}")
    except ValueError as e:
        print(f"✗ Parse failed: {e}")
    
    print("\nExample 5: Adapter class usage")
    adapter = APIAdapter()
    try:
        response = adapter.call_llm("What is AI?")
        print(f"✓ LLM response: {response}")
    except ConnectionError as e:
        print(f"✗ LLM call failed: {e}")
    
    try:
        embeddings = adapter.fetch_embeddings(["hello", "world"])
        print(f"✓ Got embeddings: {len(embeddings)} vectors")
    except ConnectionError as e:
        print(f"✗ Embedding fetch failed: {e}")


if __name__ == "__main__":
    main()
