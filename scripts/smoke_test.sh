#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8001}"
BASE_URL="http://127.0.0.1:${PORT}"
LOG_FILE="/tmp/crypto_ta_smoke.log"
PID_FILE="/tmp/crypto_ta_smoke.pid"

cleanup() {
  if [[ -f "${PID_FILE}" ]]; then
    kill "$(cat "${PID_FILE}")" >/dev/null 2>&1 || true
    rm -f "${PID_FILE}"
  fi
}
trap cleanup EXIT

python run.py >"${LOG_FILE}" 2>&1 &
echo $! > "${PID_FILE}"

for _ in {1..20}; do
  if curl -fsS "${BASE_URL}/api/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "[smoke] health"
curl -fsS "${BASE_URL}/api/health" | python -c 'import sys,json;d=json.load(sys.stdin); assert d.get("status")=="ok"; print(d)'

echo "[smoke] symbols"
SYMBOLS_JSON="$(curl -fsS "${BASE_URL}/api/symbols")"
python - <<'PY' "${SYMBOLS_JSON}"
import json,sys
arr=json.loads(sys.argv[1])
assert isinstance(arr,list) and arr
print("symbols ok, count=", len(arr), "first=", arr[0])
PY

echo "[smoke] ohlcv (best-effort local/API)"
python - <<'PY' "${BASE_URL}" "${SYMBOLS_JSON}"
import json,sys,urllib.request,urllib.error
base=sys.argv[1]
symbols=json.loads(sys.argv[2])
preferred=["ENSOUSDT","RIVERUSDT","HYPEUSDT","ACHUSDT","IPUSDT"]
pool=preferred + [s for s in symbols if s not in preferred]
last_err=None
for s in pool[:80]:
    url=f"{base}/api/ohlcv?symbol={s}&interval=1h&days=30"
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            data=json.loads(r.read().decode("utf-8"))
        print("ohlcv symbol", s, "candles", len(data.get("candles",[])), "volume", len(data.get("volume",[])))
        break
    except Exception as e:
        last_err=e
else:
    raise SystemExit(f"no symbol returned ohlcv successfully; last_error={last_err}")
PY

echo "[smoke] done"
