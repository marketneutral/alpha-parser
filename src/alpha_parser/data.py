"""Data access signals for accessing raw data fields."""

from typing import Callable, Dict, Union
import pandas as pd

from .signal import Signal


class LazyData:
    """Wrapper for data dict that supports lazy loading via callables.

    Values can be DataFrames or callables that return DataFrames.
    Callables are invoked on first access and cached.

    Example:
        data = LazyData({
            'close': lambda: pd.read_parquet('close.parquet'),
            'volume': volume_df,  # Already loaded
        })

        # 'close' is loaded only when first accessed
        close_data = data['close']
    """

    def __init__(self, data: Dict[str, Union[pd.DataFrame, Callable[[], pd.DataFrame]]]):
        self._data = data
        self._cache: Dict[str, pd.DataFrame] = {}

    def __getitem__(self, key: str) -> pd.DataFrame:
        # Return from cache if already resolved
        if key in self._cache:
            return self._cache[key]

        if key not in self._data:
            raise KeyError(key)

        value = self._data[key]

        # Resolve callable
        if callable(value):
            value = value()

        # Cache and return
        self._cache[key] = value
        return value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def keys(self):
        return self._data.keys()

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default


def resolve_data(data) -> LazyData:
    """Wrap data dict in LazyData if not already wrapped."""
    if isinstance(data, LazyData):
        return data
    return LazyData(data)


class Field(Signal):
    """Access a raw data field by name."""

    def __init__(self, name: str):
        self.name = name

    def _compute(self, data):
        data = resolve_data(data)
        if self.name not in data:
            raise ValueError(f"Field '{self.name}' not found in data. "
                           f"Available fields: {list(data.keys())}")
        return data[self.name]

    def _cache_key(self):
        return ('Field', self.name)


def close() -> Field:
    """Access the 'close' price field."""
    return Field('close')


def open() -> Field:
    """Access the 'open' price field."""
    return Field('open')


def high() -> Field:
    """Access the 'high' price field."""
    return Field('high')


def low() -> Field:
    """Access the 'low' price field."""
    return Field('low')


def field(name: str) -> Field:
    """Access any field by name."""
    return Field(name)
