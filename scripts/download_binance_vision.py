"""Binance Vision OHLCV downloader.

Pulls public 1m candles for arbitrary USDT-perp / spot symbols from
https://data.binance.vision/ — NO API KEY REQUIRED, free, unlimited.

Each (symbol, year, month) is a single .zip ~5MB. Top 200 USDT-perp
symbols × 60 months ≈ ~60 GB total (over years).

Usage:
    python -m scripts.download_binance_vision \\
        --symbols BTCUSDT ETHUSDT SOLUSDT \\
        --start 2023-01 --end 2026-04 \\
        --market futures --timeframe 1m \\
        --out-dir data/binance_vision/

Resumable: skips already-downloaded zips. CSV is extracted into a
sibling .csv file ready to be loaded by pandas / polars.
"""
from __future__ import annotations
import argparse
import io
import time
import urllib.request
import urllib.error
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _months(start: str, end: str) -> list[tuple[int, int]]:
    """Inclusive range of (year, month) tuples."""
    sy, sm = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        out.append((y, m))
        m += 1
        if m > 12:
            y += 1
            m = 1
    return out


def _url(market: str, symbol: str, tf: str, year: int, month: int) -> str:
    """
    market = 'spot' | 'futures' (futures = USDT-M perpetual)
    Files at:
      data.binance.vision/data/spot/monthly/klines/<SYMBOL>/<TF>/<SYMBOL>-<TF>-YYYY-MM.zip
      data.binance.vision/data/futures/um/monthly/klines/<SYMBOL>/<TF>/<SYMBOL>-<TF>-YYYY-MM.zip
    """
    seg = "futures/um" if market == "futures" else "spot"
    return (f"https://data.binance.vision/data/{seg}/monthly/klines/"
            f"{symbol}/{tf}/{symbol}-{tf}-{year:04d}-{month:02d}.zip")


def _download_one(url: str, target_zip: Path, target_csv: Path,
                  *, retries: int = 3, sleep: float = 2.0) -> tuple[str, int]:
    if target_csv.exists() and target_csv.stat().st_size > 0:
        return ("skip", target_csv.stat().st_size)
    target_zip.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "trendline-tokenizer/0.1"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            with target_zip.open("wb") as fh:
                fh.write(data)
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                names = zf.namelist()
                if not names:
                    return ("empty", 0)
                with zf.open(names[0]) as src, target_csv.open("wb") as dst:
                    dst.write(src.read())
            target_zip.unlink()  # keep CSV, drop zip to save disk
            return ("ok", target_csv.stat().st_size)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return ("404", 0)   # listing started later; not an error
            last_err = e
        except Exception as e:
            last_err = e
        time.sleep(sleep * (2 ** attempt))
    return (f"fail:{last_err}", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--start", required=True, help="YYYY-MM")
    ap.add_argument("--end", required=True, help="YYYY-MM")
    ap.add_argument("--market", choices=("spot", "futures"), default="futures")
    ap.add_argument("--timeframe", default="1m",
                    help="1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "binance_vision")
    args = ap.parse_args()

    months = _months(args.start, args.end)
    n_total = len(args.symbols) * len(months)
    print(f"[binance-vision] {len(args.symbols)} symbols × {len(months)} months "
          f"= {n_total} files; market={args.market} tf={args.timeframe}")
    print(f"[binance-vision] out_dir={args.out_dir}")

    counts = {"ok": 0, "skip": 0, "404": 0, "empty": 0, "fail": 0}
    bytes_total = 0
    t0 = time.time()
    n_done = 0

    for sym in args.symbols:
        for (y, m) in months:
            url = _url(args.market, sym, args.timeframe, y, m)
            csv_path = args.out_dir / args.market / sym / args.timeframe / f"{sym}-{args.timeframe}-{y:04d}-{m:02d}.csv"
            zip_path = csv_path.with_suffix(".zip")
            status, size = _download_one(url, zip_path, csv_path)
            n_done += 1
            if status.startswith("fail"):
                counts["fail"] += 1
            else:
                counts[status] = counts.get(status, 0) + 1
                bytes_total += size
            if n_done % 20 == 0 or status.startswith("fail"):
                elapsed = time.time() - t0
                rate = n_done / max(1e-6, elapsed)
                eta = int((n_total - n_done) / max(1e-6, rate))
                print(f"[binance-vision] {n_done}/{n_total}  "
                      f"counts={counts}  size={bytes_total/1e6:.1f}MB  "
                      f"rate={rate:.1f}/s eta={eta}s "
                      f"({sym} {y}-{m:02d}: {status})")

    elapsed = time.time() - t0
    print(f"[binance-vision] done in {elapsed:.0f}s. counts={counts}, "
          f"size={bytes_total/1e6:.1f} MB")


if __name__ == "__main__":
    main()
