"""Primitive signal operations: returns, volatility, volume."""

import numpy as np

from .signal import Signal
from .data import panel

# Default minimum periods for rolling operations (~1 month)
DEFAULT_MIN_PERIODS = 20


def _get_min_periods(period: int, min_periods: int = None) -> int:
    """Get effective min_periods, capped at window size."""
    if min_periods is not None:
        return min(min_periods, period)
    return min(DEFAULT_MIN_PERIODS, period)


class Returns(Signal):
    """Price returns over a period, of 'close' or of another price field (such as 'market')."""

    def __init__(self, period: int, price_field: str = 'close'):
        self.period = period
        self.price_field = price_field

    def _compute(self, data):
        prices = panel(data, self.price_field)
        return prices.pct_change(self.period)

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
        rets = prices.pct_change()
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
