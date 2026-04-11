"""Test that limit order type is properly passed to the Bitget API body."""
from server.execution.live_adapter import LiveExecutionAdapter


def test_normalize_price_rounds_to_tick():
    adapter = LiveExecutionAdapter.__new__(LiveExecutionAdapter)
    contract = {"tickSz": "0.01"}
    result = adapter._normalize_price(123.456, contract)
    assert result == "123.45"  # rounded down to tick size


def test_normalize_price_with_large_tick():
    adapter = LiveExecutionAdapter.__new__(LiveExecutionAdapter)
    contract = {"tickSz": "0.1"}
    result = adapter._normalize_price(99.87, contract)
    assert result == "99.8"


def test_normalize_price_invalid():
    adapter = LiveExecutionAdapter.__new__(LiveExecutionAdapter)
    contract = {"tickSz": "0.01"}
    result = adapter._normalize_price(0.0, contract)
    assert result is None


def test_normalize_price_uses_pricePlace_fallback():
    adapter = LiveExecutionAdapter.__new__(LiveExecutionAdapter)
    contract = {"pricePlace": "0.001"}
    result = adapter._normalize_price(50.1234, contract)
    assert result == "50.123"
