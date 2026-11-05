"""Context management for alpha computations with shared cache.

A context caches each subexpression's result for one set of data. It knows the data by the identity of its objects:
a dict is the frames it holds (a new dict around the same frames is the same data), any other mapping (a LazyData) is
itself. Evaluating on different data, even with the same shape and dates, clears the cache. The context keeps a
reference to the data it caches for, so an object's id is not reused while the cache depends on it.

A value changed in place inside a frame the context has already seen is not detected: checking the values would mean
reading the whole panel at every node (about a second on a 3,500 x 5,000 panel of eight fields). After changing
data in place, call ``ctx.clear()`` or open a new context.
"""

from contextlib import contextmanager
from typing import Optional


def _identity(data):
    """What the cache is valid for, and the objects to keep alive so their ids stay unique."""
    if isinstance(data, dict):
        keep = tuple(data.values())
        return tuple(sorted((str(k), id(v)) for k, v in data.items())), keep
    return (("", id(data)),), (data,)


class ComputeContext:
    """Global computation context with shared cache."""

    def __init__(self):
        self.cache = {}
        self.data_hash = None
        self._data = ()                      # the objects data_hash names, held so their ids are not reused

    def get_data_hash(self, data) -> tuple:
        """The identity of the data the cache is valid for (see the module docstring)."""
        return _identity(data)[0]

    def check(self, data):
        """Clear the cache if data is not the data it was computed on."""
        key, keep = _identity(data)
        if key != self.data_hash:
            self.cache = {}
            self.data_hash, self._data = key, keep

    def clear(self):
        """Clear the cache."""
        self.cache = {}
        self.data_hash = None
        self._data = ()


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
