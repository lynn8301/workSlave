#!/usr/bin/env python3
"""
Stock Screener - Filter stocks by technical and fundamental criteria.
Supports both Taiwan (TWS) and US markets via yfinance.
"""

import argparse
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------------
# Indicator calculations
# ---------------------------------------------------------------------------

def _rsi(prices: pd.Series, period: int = 14) -> float:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return float(100 - 100 / (1 + rs.iloc[-1]))


def _macd(prices: pd.Series) -> tuple[float, float]:
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return float(macd.iloc[-1]), float(signal.iloc[-1])


def _pct_change(series: pd.Series, n: int) -> float | None:
    if len(series) <= n:
        return None
    return float((series.iloc[-1] - series.iloc[-n]) / series.iloc[-n] * 100)


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_stock_info(ticker: str) -> dict | None:
    """Return a dict of price/indicator/fundamental data for one ticker."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")
        if hist.empty or len(hist) < 30:
            return None

        info = stock.info
        close = hist["Close"]
        volume = hist["Volume"]
        current = float(close.iloc[-1])

        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma60 = float(close.rolling(60).mean().iloc[-1]) if len(close) >= 60 else None

        rsi = _rsi(close)
        macd_val, macd_sig = _macd(close)

        return {
            "ticker": ticker,
            "name": (info.get("shortName") or info.get("longName") or ticker)[:22],
            "price": round(current, 2),
            "change_1d": _pct_change(close, 1),
            "change_1w": _pct_change(close, 5),
            "change_1m": _pct_change(close, 20),
            "rsi": round(rsi, 1),
            "macd_bullish": macd_val > macd_sig,
            "above_ma20": current > ma20,
            "above_ma60": (current > ma60) if ma60 is not None else None,
            "avg_volume_20d": int(volume.tail(20).mean()),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "market_cap": info.get("marketCap"),
            "eps": info.get("trailingEps"),
            "dividend_yield": info.get("dividendYield"),
            "sector": info.get("sector") or "N/A",
        }
    except Exception as exc:
        print(f"  Warning: {ticker} — {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------

def apply_filters(stocks: list[dict], criteria: dict) -> list[dict]:
    result = []
    for s in stocks:
        if s is None:
            continue

        rsi = s["rsi"]
        if "rsi_min" in criteria and rsi < criteria["rsi_min"]:
            continue
        if "rsi_max" in criteria and rsi > criteria["rsi_max"]:
            continue

        pe = s["pe_ratio"]
        if "pe_min" in criteria and (pe is None or pe < criteria["pe_min"]):
            continue
        if "pe_max" in criteria and (pe is None or pe <= 0 or pe > criteria["pe_max"]):
            continue

        c1d = s["change_1d"]
        if "change_1d_min" in criteria and (c1d is None or c1d < criteria["change_1d_min"]):
            continue
        if "change_1d_max" in criteria and (c1d is None or c1d > criteria["change_1d_max"]):
            continue

        c1m = s["change_1m"]
        if "change_1m_min" in criteria and (c1m is None or c1m < criteria["change_1m_min"]):
            continue

        if criteria.get("above_ma20") and not s["above_ma20"]:
            continue
        if criteria.get("above_ma60") and s["above_ma60"] is not True:
            continue
        if criteria.get("macd_bullish") and not s["macd_bullish"]:
            continue

        if "min_volume" in criteria and s["avg_volume_20d"] < criteria["min_volume"]:
            continue

        mc = s["market_cap"]
        if "min_market_cap" in criteria and (mc is None or mc < criteria["min_market_cap"]):
            continue

        result.append(s)
    return result


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _fmt_pct(v) -> str:
    return f"{v:+.2f}%" if v is not None else "N/A"


def _fmt_cap(v) -> str:
    if v is None:
        return "N/A"
    if v >= 1e12:
        return f"{v/1e12:.1f}T"
    if v >= 1e9:
        return f"{v/1e9:.1f}B"
    if v >= 1e6:
        return f"{v/1e6:.1f}M"
    return str(v)


def display_results(stocks: list[dict], show_fundamentals: bool = False):
    if not stocks:
        print("\nNo stocks matched the criteria.")
        return

    sep = "=" * 90
    print(f"\n{sep}")
    print(f"  {len(stocks)} stock(s) matched")
    print(sep)

    tech_rows = [
        [
            s["ticker"],
            s["name"],
            s["price"],
            _fmt_pct(s["change_1d"]),
            _fmt_pct(s["change_1w"]),
            _fmt_pct(s["change_1m"]),
            s["rsi"],
            "Bull" if s["macd_bullish"] else "Bear",
            "Y" if s["above_ma20"] else "N",
            ("Y" if s["above_ma60"] else "N") if s["above_ma60"] is not None else "N/A",
        ]
        for s in stocks
    ]
    df = pd.DataFrame(
        tech_rows,
        columns=["Ticker", "Name", "Price", "1D%", "1W%", "1M%", "RSI", "MACD", ">MA20", ">MA60"],
    )
    print(df.to_string(index=False))

    if show_fundamentals:
        print(f"\n{sep}")
        print("  Fundamental Data")
        print(sep)
        fun_rows = [
            [
                s["ticker"],
                f"{s['pe_ratio']:.1f}" if s["pe_ratio"] else "N/A",
                f"{s['forward_pe']:.1f}" if s["forward_pe"] else "N/A",
                f"{s['eps']:.2f}" if s["eps"] else "N/A",
                f"{s['dividend_yield']*100:.2f}%" if s["dividend_yield"] else "N/A",
                _fmt_cap(s["market_cap"]),
                s["sector"],
            ]
            for s in stocks
        ]
        df2 = pd.DataFrame(
            fun_rows,
            columns=["Ticker", "P/E", "Fwd P/E", "EPS", "Div Yield", "Mkt Cap", "Sector"],
        )
        print(df2.to_string(index=False))


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS: dict[str, dict] = {
    "oversold": {
        "_description": "Oversold (RSI < 35) — potential reversal candidates",
        "rsi_max": 35,
    },
    "momentum": {
        "_description": "Momentum — RSI 55-75, above MA20 & MA60, MACD bullish, +5% in a month",
        "rsi_min": 55,
        "rsi_max": 75,
        "above_ma20": True,
        "above_ma60": True,
        "macd_bullish": True,
        "change_1m_min": 5,
    },
    "value": {
        "_description": "Value — P/E between 1 and 15",
        "pe_min": 1,
        "pe_max": 15,
    },
    "breakout": {
        "_description": "Breakout — above both MAs, MACD bullish, up ≥2% today",
        "above_ma20": True,
        "above_ma60": True,
        "macd_bullish": True,
        "change_1d_min": 2,
    },
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stock Screener — supports Taiwan (.TW) and US tickers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # US momentum stocks
  python stock_screener.py AAPL MSFT GOOGL TSLA NVDA --preset momentum

  # Taiwan oversold stocks
  python stock_screener.py 2330.TW 2317.TW 2454.TW 2382.TW --preset oversold

  # Custom filters: RSI 40-60, P/E < 20, must be above MA20
  python stock_screener.py AAPL MSFT AMZN --rsi-min 40 --rsi-max 60 --pe-max 20 --above-ma20

  # Mix Taiwan + US, show fundamentals
  python stock_screener.py AAPL TSLA 2330.TW 2317.TW --preset value --fundamentals

  # Show all data without filtering
  python stock_screener.py AAPL MSFT --no-filter --fundamentals

Presets:  oversold | momentum | value | breakout
""",
    )
    parser.add_argument("tickers", nargs="+", help="Tickers, e.g. AAPL TSLA 2330.TW")
    parser.add_argument("--preset", choices=list(PRESETS), help="Predefined screening strategy")

    g = parser.add_argument_group("technical filters")
    g.add_argument("--rsi-min", type=float, metavar="N")
    g.add_argument("--rsi-max", type=float, metavar="N")
    g.add_argument("--above-ma20", action="store_true")
    g.add_argument("--above-ma60", action="store_true")
    g.add_argument("--macd-bullish", action="store_true")
    g.add_argument("--change-1d-min", type=float, metavar="PCT")
    g.add_argument("--change-1d-max", type=float, metavar="PCT")
    g.add_argument("--change-1m-min", type=float, metavar="PCT")
    g.add_argument("--min-volume", type=int, metavar="N", help="Min 20-day avg volume")

    g2 = parser.add_argument_group("fundamental filters")
    g2.add_argument("--pe-min", type=float, metavar="N")
    g2.add_argument("--pe-max", type=float, metavar="N")
    g2.add_argument("--min-market-cap", type=float, metavar="N", help="e.g. 1e9 for 1B")

    parser.add_argument("--fundamentals", action="store_true", help="Show fundamental table")
    parser.add_argument("--no-filter", action="store_true", help="Skip all filters, show raw data")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    criteria: dict = {}
    if args.preset:
        preset = PRESETS[args.preset]
        print(f"\nPreset: {preset['_description']}")
        criteria.update({k: v for k, v in preset.items() if not k.startswith("_")})

    # Individual overrides
    overrides = {
        "rsi_min": args.rsi_min,
        "rsi_max": args.rsi_max,
        "pe_min": args.pe_min,
        "pe_max": args.pe_max,
        "change_1d_min": args.change_1d_min,
        "change_1d_max": args.change_1d_max,
        "change_1m_min": args.change_1m_min,
        "min_volume": args.min_volume,
        "min_market_cap": args.min_market_cap,
    }
    criteria.update({k: v for k, v in overrides.items() if v is not None})
    if args.above_ma20:
        criteria["above_ma20"] = True
    if args.above_ma60:
        criteria["above_ma60"] = True
    if args.macd_bullish:
        criteria["macd_bullish"] = True

    print(f"\nFetching data for {len(args.tickers)} ticker(s)...")
    stocks: list[dict] = []
    for i, ticker in enumerate(args.tickers, 1):
        print(f"  [{i}/{len(args.tickers)}] {ticker} ...", end=" ", flush=True)
        data = fetch_stock_info(ticker)
        if data:
            stocks.append(data)
            print("OK")
        else:
            print("SKIP")

    if args.no_filter or not criteria:
        display_results(stocks, show_fundamentals=args.fundamentals)
    else:
        print(f"\nApplying {len(criteria)} filter(s): {', '.join(criteria)}")
        filtered = apply_filters(stocks, criteria)
        display_results(filtered, show_fundamentals=args.fundamentals)


if __name__ == "__main__":
    main()
