"""
Tests for the retry decorator with exponential backoff.
"""

import time
from unittest.mock import Mock, patch

import pytest

from agentunit.core.utils import retry


class TestRetryDecorator:
    """Test suite for the retry decorator."""

    def test_successful_call_no_retry(self):
        """Test that successful calls don't trigger retries."""
        mock_func = Mock(return_value="success")
        decorated = retry()(mock_func)

        result = decorated()

        assert result == "success"
        assert mock_func.call_count == 1

    def test_retry_on_exception(self):
        """Test that function retries on exception."""
        mock_func = Mock(side_effect=[ValueError("error"), "success"])
        decorated = retry(max_retries=2, base_delay=0.01)(mock_func)

        result = decorated()

        assert result == "success"
        assert mock_func.call_count == 2

    def test_max_retries_exceeded(self):
        """Test that exception is raised after max retries."""
        mock_func = Mock(side_effect=ValueError("persistent error"))
        decorated = retry(max_retries=2, base_delay=0.01)(mock_func)

        with pytest.raises(ValueError, match="persistent error"):
            decorated()

        assert mock_func.call_count == 3  # Initial + 2 retries

    def test_exponential_backoff(self):
        """Test that delays follow exponential backoff pattern."""
        mock_func = Mock(side_effect=[ValueError(), ValueError(), "success"])
        
        with patch("time.sleep") as mock_sleep:
            decorated = retry(
                max_retries=3,
                base_delay=1.0,
                exponential_base=2.0
            )(mock_func)
            
            result = decorated()

        assert result == "success"
        assert mock_func.call_count == 3
        
        # Check exponential backoff: 1.0, 2.0
        calls = mock_sleep.call_args_list
        assert len(calls) == 2
        assert calls[0][0][0] == 1.0  # First retry: base_delay * 2^0
        assert calls[1][0][0] == 2.0  # Second retry: base_delay * 2^1

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        mock_func = Mock(side_effect=[ValueError(), ValueError(), "success"])
        
        with patch("time.sleep") as mock_sleep:
            decorated = retry(
                max_retries=3,
                base_delay=10.0,
                max_delay=15.0,
                exponential_base=2.0
            )(mock_func)
            
            result = decorated()

        assert result == "success"
        
        # Check that delays are capped
        calls = mock_sleep.call_args_list
        assert len(calls) == 2
        assert calls[0][0][0] == 10.0  # First retry: 10.0
        assert calls[1][0][0] == 15.0  # Second retry: capped at max_delay

    def test_specific_exceptions_only(self):
        """Test that only specified exceptions trigger retries."""
        mock_func = Mock(side_effect=TypeError("wrong type"))
        decorated = retry(
            max_retries=2,
            base_delay=0.01,
            exceptions=(ValueError,)
        )(mock_func)

        # TypeError should not be caught, so it raises immediately
        with pytest.raises(TypeError, match="wrong type"):
            decorated()

        assert mock_func.call_count == 1  # No retries

    def test_multiple_exception_types(self):
        """Test retry with multiple exception types."""
        mock_func = Mock(side_effect=[ValueError(), ConnectionError(), "success"])
        decorated = retry(
            max_retries=3,
            base_delay=0.01,
            exceptions=(ValueError, ConnectionError)
        )(mock_func)

        result = decorated()

        assert result == "success"
        assert mock_func.call_count == 3

    def test_function_with_arguments(self):
        """Test that decorated function preserves arguments."""
        mock_func = Mock(return_value="result")
        decorated = retry()(mock_func)

        result = decorated("arg1", "arg2", kwarg1="value1")

        assert result == "result"
        mock_func.assert_called_once_with("arg1", "arg2", kwarg1="value1")

    def test_function_metadata_preserved(self):
        """Test that function metadata is preserved by decorator."""
        def sample_function():
            """Sample docstring."""
            pass

        decorated = retry()(sample_function)

        assert decorated.__name__ == "sample_function"
        assert decorated.__doc__ == "Sample docstring."

    def test_zero_retries(self):
        """Test behavior with max_retries=0."""
        mock_func = Mock(side_effect=ValueError("error"))
        decorated = retry(max_retries=0, base_delay=0.01)(mock_func)

        with pytest.raises(ValueError, match="error"):
            decorated()

        assert mock_func.call_count == 1  # Only initial call, no retries

    @patch("agentunit.core.utils.logger")
    def test_logging_on_retry(self, mock_logger):
        """Test that retries are logged appropriately."""
        mock_func = Mock(side_effect=[ValueError("error"), "success"])
        decorated = retry(max_retries=2, base_delay=0.01)(mock_func)

        result = decorated()

        assert result == "success"
        # Check that warning was logged
        assert mock_logger.warning.called

    @patch("agentunit.core.utils.logger")
    def test_logging_on_final_failure(self, mock_logger):
        """Test that final failure is logged as error."""
        mock_func = Mock(side_effect=ValueError("persistent error"))
        decorated = retry(max_retries=1, base_delay=0.01)(mock_func)

        with pytest.raises(ValueError):
            decorated()

        # Check that error was logged
        assert mock_logger.error.called

    def test_real_world_api_scenario(self):
        """Test a realistic API call scenario."""
        call_count = 0
        
        def flaky_api_call():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network timeout")
            return {"status": "success", "data": [1, 2, 3]}

        decorated = retry(
            max_retries=5,
            base_delay=0.01,
            exceptions=(ConnectionError,)
        )(flaky_api_call)

        result = decorated()

        assert result == {"status": "success", "data": [1, 2, 3]}
        assert call_count == 3
