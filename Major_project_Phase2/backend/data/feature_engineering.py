"""
Feature engineering for financial time series.
Computes 28+ technical indicators, calendar features, and target labels.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Compute technical indicators, calendar features, and labels."""

    # -------------------------------------------------------
    # Technical Indicator Helpers
    # -------------------------------------------------------
    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        up = np.where(delta > 0, delta, 0.0)
        down = np.where(delta < 0, -delta, 0.0)
        roll_up = pd.Series(up, index=series.index).ewm(alpha=1 / period, adjust=False).mean()
        roll_down = pd.Series(down, index=series.index).ewm(alpha=1 / period, adjust=False).mean()
        rs = roll_up / roll_down.replace(0, np.nan)
        return (100 - (100 / (1 + rs))).fillna(50.0)

    @staticmethod
    def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - signal_line
        return macd_line, signal_line, hist

    @staticmethod
    def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    @staticmethod
    def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        tr_raw = pd.concat([
            (high - low),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        tr_n = tr_raw.ewm(alpha=1 / period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean() / tr_n.replace(0, np.nan)
        minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean() / tr_n.replace(0, np.nan)
        dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
        return dx.ewm(alpha=1 / period, adjust=False).mean()

    @staticmethod
    def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3):
        lowest_low = low.rolling(k_period, min_periods=1).min()
        highest_high = high.rolling(k_period, min_periods=1).max()
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
        stoch_d = stoch_k.rolling(d_period, min_periods=1).mean()
        return stoch_k.fillna(50.0), stoch_d.fillna(50.0)

    # -------------------------------------------------------
    # Main Feature Computation
    # -------------------------------------------------------
    @classmethod
    def add_technicals(cls, prices: pd.DataFrame) -> pd.DataFrame:
        if prices is None or prices.empty:
            return prices
        df = prices.sort_values(["ticker", "date"]).copy()

        def _calc(g: pd.DataFrame) -> pd.DataFrame:
            g = g.copy()
            adj = g["adj_close"].astype(float)
            high = g["high"].astype(float)
            low = g["low"].astype(float)
            close = g["close"].astype(float)
            vol = g["volume"].astype(float)

            # Returns
            g["ret_1d"] = adj.pct_change(1)
            g["ret_3d"] = adj.pct_change(3)
            g["ret_5d"] = adj.pct_change(5)
            g["ret_10d"] = adj.pct_change(10)

            # Volatility
            g["vol_10d"] = g["ret_1d"].rolling(10, min_periods=3).std()
            g["vol_21d"] = g["ret_1d"].rolling(21, min_periods=5).std()

            # Moving averages and ratios
            sma_5 = adj.rolling(5, min_periods=1).mean()
            sma_10 = adj.rolling(10, min_periods=1).mean()
            sma_20 = adj.rolling(20, min_periods=5).mean()
            g["sma_ratio_5_10"] = sma_5 / sma_10.replace(0, np.nan)
            g["sma_ratio_5_20"] = sma_5 / sma_20.replace(0, np.nan)

            # RSI
            g["rsi_14"] = cls._rsi(adj, 14)

            # MACD
            macd_line, signal_line, hist = cls._macd(adj)
            g["macd"] = macd_line
            g["macd_signal"] = signal_line
            g["macd_hist"] = hist

            # Bollinger Bands
            mean20 = adj.rolling(20, min_periods=5).mean()
            std20 = adj.rolling(20, min_periods=5).std()
            upper = mean20 + 2 * std20
            lower = mean20 - 2 * std20
            g["bb_pos"] = ((adj - lower) / (upper - lower)).clip(0, 1)
            g["bb_bw"] = ((upper - lower) / mean20).replace([np.inf, -np.inf], np.nan)

            # ATR & ADX
            g["atr_14"] = cls._atr(high, low, close, 14)
            g["adx_14"] = cls._adx(high, low, close, 14)

            # Momentum
            g["mom_10"] = adj.pct_change(10)
            g["mom_20"] = adj.pct_change(20)

            # Z-scores
            g["zscore_20"] = ((adj - mean20) / std20).replace([np.inf, -np.inf], np.nan)
            g["vol_z_20"] = (vol - vol.rolling(20, min_periods=5).mean()) / vol.rolling(20, min_periods=5).std()

            # OBV normalized
            sign = np.sign(adj.diff()).fillna(0)
            obv = (sign * vol).cumsum()
            obv_mean = obv.rolling(20, min_periods=5).mean()
            obv_std = obv.rolling(20, min_periods=5).std().replace(0, np.nan)
            g["obv_norm"] = ((obv - obv_mean) / obv_std).fillna(0)

            # VWAP ratio
            typical_price = (high + low + close) / 3
            vwap = (typical_price * vol).cumsum() / vol.cumsum().replace(0, np.nan)
            g["vwap_ratio"] = (close / vwap).fillna(1.0)

            # Stochastic Oscillator
            stoch_k, stoch_d = cls._stochastic(high, low, close)
            g["stoch_k"] = stoch_k
            g["stoch_d"] = stoch_d

            return g

        df = df.groupby("ticker", group_keys=False).apply(_calc)
        return df

    @classmethod
    def add_calendar_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"])
        d["dow"] = d["date"].dt.weekday
        d["month"] = d["date"].dt.month
        d["dow_sin"] = np.sin(2 * np.pi * d["dow"] / 7)
        d["dow_cos"] = np.cos(2 * np.pi * d["dow"] / 7)
        d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
        d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)
        return d

    @classmethod
    def add_labels(cls, df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
        d = df.sort_values(["ticker", "date"]).copy()

        def _label(g: pd.DataFrame) -> pd.DataFrame:
            adj = g["adj_close"].astype(float)
            future_ret = adj.shift(-horizon) / adj - 1.0
            g["ret_target"] = future_ret
            g["y_class"] = (future_ret > 0).astype(int)
            return g

        d = d.groupby("ticker", group_keys=False).apply(_label)
        return d

    @classmethod
    def build_dataset(
        cls,
        prices: pd.DataFrame,
        horizon: int = 1,
        feature_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        df = cls.add_technicals(prices)
        df = cls.add_calendar_features(df)
        df = cls.add_labels(df, horizon=horizon)

        if feature_cols is None:
            from config import TECHNICAL_FEATURES
            feature_cols = TECHNICAL_FEATURES

        meta_cols = ["ticker", "date", "adj_close"]
        label_cols = ["ret_target", "y_class"]
        keep = meta_cols + feature_cols + label_cols
        keep = [c for c in keep if c in df.columns]

        ds = df[keep].replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
        logger.info("Built dataset: %d rows, %d features", len(ds), len(feature_cols))
        return ds
