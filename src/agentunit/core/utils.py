"""
Utility functions for AgentUnit core functionality.
"""

import functools
import logging
import time
from typing import Any, Callable, TypeVar, cast

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator that retries a function with exponential backoff on failure.

    This decorator is useful for handling flaky API calls or network operations
    that may fail temporarily. It implements exponential backoff to avoid
    overwhelming the service with rapid retry attempts.

    Args:
        max_retries: Maximum number of retry attempts (default: 3).
        base_delay: Initial delay between retries in seconds (default: 1.0).
        max_delay: Maximum delay between retries in seconds (default: 60.0).
        exponential_base: Base for exponential backoff calculation (default: 2.0).
        exceptions: Tuple of exception types to catch and retry (default: (Exception,)).

    Returns:
        Decorated function that implements retry logic with exponential backoff.

    Example:
        >>> @retry(max_retries=3, base_delay=1.0, exceptions=(ConnectionError,))
        ... def fetch_data():
        ...     # API call that might fail
        ...     return api.get_data()

        >>> @retry(max_retries=5, base_delay=2.0, max_delay=30.0)
        ... def process_request():
        ...     # Processing that might need retries
        ...     return process()
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries. "
                            f"Last error: {e}"
                        )
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base**attempt), max_delay)

                    logger.warning(
                        f"Function {func.__name__} failed on attempt {attempt + 1}/{max_retries + 1}. "
                        f"Retrying in {delay:.2f}s. Error: {e}"
                    )

                    time.sleep(delay)

            # This should never be reached, but satisfies type checker
            if last_exception:
                raise last_exception

        return cast(F, wrapper)

    return decorator
