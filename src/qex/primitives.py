"""Primitive signal operations: returns, volatility, volume.

Also the one rule for turning prices into returns, used by every part of qex that does it
(``returns``, ``volatility``, the risk model, and the forward returns of backtests and
quantile/IC analysis): ``price_returns`` and ``forward_returns``.
"""

import numpy as np

from .signal import Signal
from .data import panel

# Default minimum periods for rolling operations (~1 month)
DEFAULT_MIN_PERIODS = 20


# A price more than this many rows old is treated as missing (about two trading weeks).
MAX_PRICE_GAP = 10


def price_returns(prices, period: int = 1, max_gap: int = None):
    """Simple returns over ``period`` rows, bridging missing prices without inventing any.

    The return on day t is ``p[t] / p[last valid at or before t - period] - 1``, and NaN when
    ``p[t]`` is missing. So:

    - during a halt, and after a delisting, the return is NaN (no fake 0% days);
    - the day a halted name reopens carries the whole move since its last price;
    - daily returns compound exactly to the ``period``-day return.

    A price older than ``max_gap`` rows (default ``MAX_PRICE_GAP``) no longer counts, so a
    name that comes back after a long suspension gets NaN, not one enormous return.

    This replaces ``pct_change``, whose handling of missing prices depends on the pandas
    version (pandas 2 carried prices forward and gave 0% returns; pandas 3 does not bridge
    gaps at all). This function gives the same answer on both.

    Args:
        prices: DataFrame (dates x names) or Series (dates) of prices.
        period: Number of rows the return spans.
        max_gap: Most rows a last valid price may be carried forward.

    Returns:
        Returns, with the same shape as ``prices``.
    """
    if max_gap is None:
        max_gap = MAX_PRICE_GAP
    if period < 1:
        raise ValueError(f"period must be >= 1 (got {period})")
    if max_gap < 0:
        raise ValueError(f"max_gap must be >= 0 (got {max_gap})")
    prices = prices.astype(float)
    bridged = prices.ffill(limit=max_gap) if max_gap > 0 else prices
    out = prices / bridged.shift(period) - 1
    return out.where(prices.notna())


def forward_returns(prices, period: int = 1, max_gap: int = None):
    """Return from day t to day t + ``period``: ``price_returns(prices, period).shift(-period)``.

    Day t's value is ``p[t + period] / p[last valid at or before t] - 1``, NaN when
    ``p[t + period]`` is missing. A position held through a halt therefore gets the whole
    gap move on the row before the reopen, and nothing on the halt days.
    """
    return price_returns(prices, period, max_gap).shift(-period)


def _get_min_periods(period: int, min_periods: int = None) -> int:
    """Get effective min_periods, capped at window size."""
    if min_periods is not None:
        return min(min_periods, period)
    return min(DEFAULT_MIN_PERIODS, period)


class Returns(Signal):
    """Price returns over a period, of 'close' or of another price field (such as 'market').

    Missing prices are handled by ``price_returns``: NaN on days with no price, and a missing
    start price is replaced by the last valid one before it (up to ``MAX_PRICE_GAP`` rows old).
    """

    def __init__(self, period: int, price_field: str = 'close'):
        self.period = period
        self.price_field = price_field

    def _compute(self, data):
        prices = panel(data, self.price_field)
        return price_returns(prices, self.period)

    def _cache_key(self):
        return ('Returns', self.period, self.price_field)


class Volatility(Signal):
    """Rolling volatility (annualized standard deviation of returns)."""

    def __init__(self, period: int, price_field: str = 'close', min_periods: int = None):
        self.period = period
        self.price_field = price_field
        self.min_periods = _get_min_periods(period, min_periods)

    def _compute(self, data):
        prices = panel(data, self.price_field)
        rets = price_returns(prices)
        return rets.rolling(self.period, min_periods=self.min_periods).std() * np.sqrt(252)

    def _cache_key(self):
        return ('Volatility', self.period, self.price_field, self.min_periods)


class Volume(Signal):
    """Rolling average volume."""

    def __init__(self, period: int, min_periods: int = None):
        self.period = period
        self.min_periods = _get_min_periods(period, min_periods)

    def _compute(self, data):
        vol = data['volume']
        return vol.rolling(self.period, min_periods=self.min_periods).mean()

    def _cache_key(self):
        return ('Volume', self.period, self.min_periods)


def returns(period: int, price_field: str = 'close') -> Returns:
    """Create a Returns signal. returns(1, 'market') is the market's daily return."""
    return Returns(period, price_field)


def volatility(period: int, price_field: str = 'close') -> Volatility:
    """Create a Volatility signal. volatility(63, 'market') is the market's."""
    return Volatility(period, price_field)


def volume(period: int) -> Volume:
    """Create a Volume signal."""
    return Volume(period)


class Adv(Signal):
    """Average dollar volume (price * volume) over a period."""

    def __init__(self, period: int, price_field: str = 'close', min_periods: int = None):
        self.period = period
        self.price_field = price_field
        self.min_periods = _get_min_periods(period, min_periods)

    def _compute(self, data):
        prices = data[self.price_field]
        vol = data['volume']
        dollar_volume = prices * vol
        return dollar_volume.rolling(self.period, min_periods=self.min_periods).mean()

    def _cache_key(self):
        return ('Adv', self.period, self.price_field, self.min_periods)


def adv(period: int) -> Adv:
    """Create an Adv signal (average dollar volume)."""
    return Adv(period)
