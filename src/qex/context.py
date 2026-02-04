"""Context management for alpha computations with shared cache."""

import hashlib
from contextlib import contextmanager
from typing import Dict, Optional

import pandas as pd


class ComputeContext:
    """Global computation context with shared cache."""

    def __init__(self):
        self.cache = {}
        self.data_hash = None

    def get_data_hash(self, data: Dict[str, pd.DataFrame]) -> str:
        """Create stable hash of data for cache validation."""
        parts = []
        for key in sorted(data.keys()):
            df = data[key]
            if isinstance(df, pd.DataFrame):
                parts.append(f"{key}:{df.shape}:{df.index[0]}:{df.index[-1]}")
            else:
                parts.append(f"{key}:{id(df)}")
        return hashlib.md5("".join(parts).encode()).hexdigest()

    def clear(self):
        """Clear the cache."""
        self.cache = {}
        self.data_hash = None


_context: Optional[ComputeContext] = None


@contextmanager
def compute_context():
    """Context manager for alpha computations with shared cache."""
    global _context
    old_context = _context
    _context = ComputeContext()
    try:
        yield _context
    finally:
        _context = old_context


def get_context() -> Optional[ComputeContext]:
    """Get current compute context."""
    return _context
