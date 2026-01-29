"""Data access signals for accessing raw data fields."""

from .signal import Signal


class Field(Signal):
    """Access a raw data field by name."""

    def __init__(self, name: str):
        self.name = name

    def _compute(self, data):
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
