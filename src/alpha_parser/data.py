"""Data access signals for accessing raw data fields."""

from typing import Callable, Dict, Optional, Union
import pandas as pd

from .signal import Signal


class LazyData:
    """Wrapper for data dict that supports lazy loading via callables.

    Values can be DataFrames or callables that return DataFrames.
    Callables are invoked on first access and cached.

    Supports optional field descriptions for self-documenting data,
    enabling AI agents to understand available fields without external docs.

    Example:
        data = LazyData({
            'close': lambda: pd.read_parquet('close.parquet'),
            'volume': volume_df,  # Already loaded
        })

        # 'close' is loaded only when first accessed
        close_data = data['close']

    Example with descriptions:
        data = LazyData(
            data={
                'close': close_df,
                'volume': volume_df,
                'sector': sector_df,
            },
            descriptions={
                'close': 'Daily closing price, adjusted for splits and dividends',
                'volume': 'Daily trading volume in shares',
                'sector': 'GICS sector classification string',
            }
        )

        data.describe()  # Pretty-print available fields
    """

    def __init__(
        self,
        data: Dict[str, Union[pd.DataFrame, Callable[[], pd.DataFrame]]],
        descriptions: Optional[Dict[str, str]] = None,
    ):
        self._data = data
        self._cache: Dict[str, pd.DataFrame] = {}
        self._descriptions: Dict[str, str] = descriptions or {}

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

    @property
    def descriptions(self) -> Dict[str, str]:
        """Return the field descriptions dictionary."""
        return self._descriptions

    def describe(self, field: Optional[str] = None) -> str:
        """Pretty-print available fields with descriptions.

        Args:
            field: If provided, describe only this field. Otherwise describe all.

        Returns:
            Formatted string describing the field(s).

        Example output:
            Available fields:
              close:  Daily closing price, adjusted for splits and dividends
              volume: Daily trading volume in shares
              sector: GICS sector classification string
        """
        if field is not None:
            # Describe single field
            if field not in self._data:
                return f"Field '{field}' not found. Available: {list(self.keys())}"
            desc = self._descriptions.get(field, "No description available")
            dtype_info = self._get_dtype_info(field)
            return f"{field}: {desc}{dtype_info}"

        # Describe all fields
        lines = ["Available fields:"]
        max_name_len = max(len(k) for k in self._data.keys()) if self._data else 0

        for key in sorted(self._data.keys()):
            desc = self._descriptions.get(key, "No description available")
            dtype_info = self._get_dtype_info(key)
            lines.append(f"  {key:<{max_name_len}}  {desc}{dtype_info}")

        return "\n".join(lines)

    def _get_dtype_info(self, key: str) -> str:
        """Get dtype information for a field if already loaded."""
        if key in self._cache:
            df = self._cache[key]
            # Get the predominant dtype
            if len(df.dtypes.unique()) == 1:
                dtype = df.dtypes.iloc[0]
                return f" (dtype: {dtype})"
            else:
                return f" (mixed dtypes)"
        return ""

    def __repr__(self) -> str:
        """Return a concise representation."""
        n_fields = len(self._data)
        n_described = len(self._descriptions)
        return f"LazyData({n_fields} fields, {n_described} with descriptions)"


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
