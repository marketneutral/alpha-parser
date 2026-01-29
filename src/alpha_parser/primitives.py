"""Primitive signal operations: returns, volatility, volume."""

import numpy as np

from .signal import Signal


class Returns(Signal):
    """Price returns over a period."""

    def __init__(self, period: int, price_field: str = 'close'):
        self.period = period
        self.price_field = price_field

    def _compute(self, data):
        prices = data[self.price_field]
        return prices.pct_change(self.period)

    def _cache_key(self):
        return ('Returns', self.period, self.price_field)


class Volatility(Signal):
    """Rolling volatility (annualized standard deviation of returns)."""

    def __init__(self, period: int, price_field: str = 'close'):
        self.period = period
        self.price_field = price_field

    def _compute(self, data):
        prices = data[self.price_field]
        rets = prices.pct_change()
        return rets.rolling(self.period).std() * np.sqrt(252)

    def _cache_key(self):
        return ('Volatility', self.period, self.price_field)


class Volume(Signal):
    """Rolling average volume."""

    def __init__(self, period: int):
        self.period = period

    def _compute(self, data):
        vol = data['volume']
        return vol.rolling(self.period).mean()

    def _cache_key(self):
        return ('Volume', self.period)


def returns(period: int) -> Returns:
    """Create a Returns signal."""
    return Returns(period)


def volatility(period: int) -> Volatility:
    """Create a Volatility signal."""
    return Volatility(period)


def volume(period: int) -> Volume:
    """Create a Volume signal."""
    return Volume(period)
