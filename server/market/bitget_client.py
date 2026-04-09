import httpx


class BitgetPublicClient:
    """Minimal public Bitget futures market client for normalized data ingestion."""

    def __init__(
        self,
        http_client: httpx.AsyncClient | None = None,
        *,
        base_url: str = "https://api.bitget.com",
        product_type: str = "usdt-futures",
    ) -> None:
        self._http_client = http_client
        self._base_url = base_url.rstrip("/")
        self._product_type = product_type.lower()

    async def _get(self, path: str, params: dict[str, str] | None = None) -> list | dict:
        owns_client = self._http_client is None
        client = self._http_client or httpx.AsyncClient(timeout=30.0)
        try:
            response = await client.get(f"{self._base_url}{path}", params=params)
            response.raise_for_status()
            payload = response.json()
        finally:
            if owns_client:
                await client.aclose()

        if not isinstance(payload, dict) or payload.get("code") != "00000":
            raise ValueError(f"Bitget API error at {path}: {payload}")
        return payload.get("data") or []

    async def get_contracts(self, symbol: str | None = None) -> list[dict]:
        params = {"productType": self._product_type}
        if symbol:
            params["symbol"] = symbol.upper()
        data = await self._get("/api/v2/mix/market/contracts", params)
        return list(data) if isinstance(data, list) else []

    async def get_tickers(self) -> list[dict]:
        data = await self._get(
            "/api/v2/mix/market/tickers",
            {"productType": self._product_type},
        )
        return list(data) if isinstance(data, list) else []

    async def get_candles(
        self,
        symbol: str,
        granularity: str,
        *,
        start_time: int | None = None,
        end_time: int | None = None,
        limit: int = 200,
        history: bool = True,
    ) -> list[list[str]]:
        params: dict[str, str] = {
            "symbol": symbol.upper(),
            "productType": self._product_type,
            "granularity": granularity,
            "limit": str(limit),
        }
        if start_time is not None:
            params["startTime"] = str(start_time)
        if end_time is not None:
            params["endTime"] = str(end_time)

        path = "/api/v2/mix/market/history-candles" if history else "/api/v2/mix/market/candles"
        data = await self._get(path, params)
        return list(data) if isinstance(data, list) else []
