from __future__ import annotations

from server.okx_trader import OKXTrader


def test_okx_trader_accepts_okx_api_secret_fallback(monkeypatch) -> None:
    monkeypatch.setenv("OKX_API_KEY", "key")
    monkeypatch.setenv("OKX_API_SECRET", "secret-from-api-secret")
    monkeypatch.setenv("OKX_PASSPHRASE", "pass")
    monkeypatch.delenv("OKX_SECRET", raising=False)

    trader = OKXTrader(api_key="", api_secret="", passphrase="")

    assert trader.api_key == "key"
    assert trader.api_secret == "secret-from-api-secret"
    assert trader.passphrase == "pass"
    assert trader.has_api_keys() is True
