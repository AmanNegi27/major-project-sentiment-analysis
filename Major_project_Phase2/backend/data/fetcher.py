"""
Financial data fetcher using yfinance.
Handles multi-ticker downloads with robust error handling.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List, Union, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetch OHLCV data from Yahoo Finance for one or more tickers."""

    @staticmethod
    def _normalize_tickers(tickers: List[str]) -> List[str]:
        out = []
        for t in tickers:
            if isinstance(t, str):
                t = t.strip().upper()
                if t:
                    out.append(t)
        return sorted(set(out))

    @classmethod
    def fetch_prices(
        cls,
        tickers: List[str],
        start: Union[str, datetime] = "2015-01-01",
        end: Union[str, datetime, None] = None,
    ) -> pd.DataFrame:
        tickers = cls._normalize_tickers(tickers)
        if not tickers:
            return pd.DataFrame()

        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end) if end else pd.Timestamp.now()

        logger.info("Fetching prices for %s from %s to %s", tickers, start_dt.date(), end_dt.date())

        try:
            data = yf.download(
                tickers=tickers,
                start=start_dt,
                end=end_dt + timedelta(days=2),
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )
        except Exception as e:
            logger.error("yfinance download failed: %s", e)
            return pd.DataFrame()

        if data is None or len(data) == 0:
            return pd.DataFrame()

        frames = []
        if len(tickers) == 1:
            t = tickers[0]
            df = data.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [field.lower().replace(" ", "_") for _, field in df.columns]
            else:
                df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            if "adj_close" not in df.columns and "close" in df.columns:
                df["adj_close"] = df["close"]
            df["ticker"] = t
            df["date"] = pd.to_datetime(df.index).tz_localize(None)
            frames.append(df.reset_index(drop=True))
        else:
            for t in tickers:
                try:
                    df = data[t].copy()
                except (KeyError, TypeError):
                    continue
                if df is None or df.empty:
                    continue
                df.columns = [c.lower().replace(" ", "_") for c in df.columns]
                if "adj_close" not in df.columns and "close" in df.columns:
                    df["adj_close"] = df["close"]
                df["ticker"] = t
                df["date"] = pd.to_datetime(df.index).tz_localize(None)
                frames.append(df.reset_index(drop=True))

        if not frames:
            return pd.DataFrame()

        prices = pd.concat(frames, ignore_index=True)
        for c in ["open", "high", "low", "close", "adj_close", "volume"]:
            if c not in prices.columns:
                prices[c] = np.nan
        prices = prices.dropna(subset=["adj_close"])
        prices = prices[["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]]
        prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

        logger.info("Fetched %d rows for %d tickers", len(prices), len(tickers))
        return prices

    @classmethod
    def fetch_single(
        cls,
        ticker: str,
        start: Union[str, datetime] = "2015-01-01",
        end: Union[str, datetime, None] = None,
    ) -> pd.DataFrame:
        df = cls.fetch_prices([ticker], start=start, end=end)
        if df.empty:
            # Fallback: generate sample data for demo purposes
            logger.warning(f"No data from yfinance for {ticker}, using sample data")
            df = cls._generate_sample_data(ticker, start, end)
        return df

    @classmethod
    def _generate_sample_data(
        cls,
        ticker: str,
        start: Union[str, datetime],
        end: Union[str, datetime, None]
    ) -> pd.DataFrame:
        """Generate sample price data for demonstration when yfinance fails."""
        import numpy as np

        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end) if end else pd.Timestamp.now()

        # Generate daily dates
        dates = pd.date_range(start=start_dt, end=end_dt, freq='D')

        # Generate realistic sample prices (starting around $100-200)
        np.random.seed(42)  # For reproducible results
        base_price = np.random.uniform(100, 200)

        prices = []
        current_price = base_price

        for date in dates:
            # Random walk with slight upward trend
            change = np.random.normal(0.001, 0.02)  # Small daily changes
            current_price *= (1 + change)

            # Generate OHLCV data
            volatility = 0.02
            high = current_price * (1 + abs(np.random.normal(0, volatility)))
            low = current_price * (1 - abs(np.random.normal(0, volatility)))
            open_price = current_price * (1 + np.random.normal(0, volatility/2))
            close = current_price
            volume = np.random.randint(1000000, 10000000)  # 1M to 10M shares

            prices.append({
                'ticker': ticker,
                'date': date,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'adj_close': round(close, 2),
                'volume': int(volume)
            })

        return pd.DataFrame(prices)
