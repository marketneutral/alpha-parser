"""Book construction with the standard simulation settings of Finding Alphas.

A signal becomes a book the same way every time: restricted to a universe, delayed so
that a day's weights use only data known before that day's trade, neutralized, scaled
to a constant book size, and truncated so no name dominates. The settings are the ones
WorldQuant BRAIN exposes (delay, universe, neutralization, truncation).

Example:
    >>> from qex import qex
    >>> from evaluation import BookSettings, build_book
    >>> settings = BookSettings(delay=1, universe='member', neutralize='sector', truncation=0.05)
    >>> weights = build_book(qex("-returns(5)"), data, settings)
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from qex.signal import Signal
from qex.context import compute_context
from qex.data import close, field, resolve_data
from qex.groups import _get_group_data, _group_codes


@dataclass(frozen=True)
class BookSettings:
    """Simulation settings for turning a signal into daily weights.

    Attributes:
        delay: Weights for day t use data up to day t - delay. Delay 1 trades at day t's
            close on what was known at day t-1's close; delay 0 trades at the same close
            the signal saw.
        universe: Name of a data field that is 1 (or any finite value) for names in the
            universe that day and NaN otherwise. None means every name with a price.
        neutralize: "market" demeans across all names each day, a group field name
            (e.g. "sector") demeans within each group, None leaves the book unneutralized.
        truncation: Largest absolute weight of any one name, as a fraction of the book.
            None means no cap.
        rounds: Neutralize, scale and clip are repeated, day by day, until all three hold,
            at most this many times (clipping alone breaks neutrality and book size). Most
            days need a few rounds; a heavy-tailed signal on a small cross-section can need
            a few hundred. A day still short of the cap after the last round is neutralized
            and scaled once more, so neutrality and book size always hold exactly and the
            cap is the one that may be exceeded, slightly.
    """

    delay: int = 1
    universe: Optional[str] = None
    neutralize: Optional[str] = "market"
    truncation: Optional[float] = 0.05
    rounds: int = 1000

    def __post_init__(self):
        if self.delay < 0:
            raise ValueError(f"delay must be >= 0 (got {self.delay})")
        if self.truncation is not None and not 0 < self.truncation <= 1:
            raise ValueError(f"truncation must be in (0, 1] or None (got {self.truncation})")
        if self.rounds < 1:
            raise ValueError(f"rounds must be >= 1 (got {self.rounds})")


def build_book(
    signal: Signal,
    data,
    settings: BookSettings = BookSettings(),
    start=None,
    end=None,
) -> pd.DataFrame:
    """Daily weights of the book a signal implies under the given settings.

    Each day the weights sum to zero (within each group, if neutralized by group), their
    absolute values sum to exactly one, and none exceeds the truncation. A day with no
    tradable names, or where the signal is flat, has all-zero weights. A name gets no
    weight on a day it has no close (it cannot be traded), is outside the universe as of
    the signal's date, or has no group under group neutralization.

    Args:
        signal: The alpha signal.
        data: Dict or LazyData with at least 'close', plus the universe and group fields
            the settings name.
        settings: Simulation settings.
        start, end: Optional date bounds, applied after the delay.

    Returns:
        DataFrame of weights, dates by names.
    """
    masked = signal + 0 * close()
    if settings.universe is not None:
        masked = masked + 0 * field(settings.universe)
    with compute_context():
        raw = masked.evaluate(data)
    if settings.delay:
        raw = raw.shift(settings.delay)
    raw = raw.where(resolve_data(data)["close"].reindex_like(raw).notna())
    raw = raw.loc[start:end]

    w = raw.to_numpy(dtype=float).copy()
    ok = np.isfinite(w)
    codes, n_groups = None, 0
    if settings.neutralize not in (None, "market"):
        codes, n_groups = _group_codes(raw, _get_group_data(data, settings.neutralize))
        ok &= codes >= 0
    w[~ok] = 0.0

    def neutral_and_scaled(rows):
        sub = _neutralize(w[rows], ok[rows], None if codes is None else codes[rows],
                          n_groups if codes is not None else 0, settings.neutralize)
        g = np.abs(sub).sum(axis=1, keepdims=True)
        return np.divide(sub, g, out=np.zeros_like(sub), where=g > 0)

    cap = settings.truncation
    rows = np.arange(w.shape[0])                      # days not yet within the cap
    for _ in range(settings.rounds):
        sub = neutral_and_scaled(rows)
        if cap is None:
            w[rows] = sub
            rows = rows[:0]
            break
        done = np.abs(sub).max(axis=1, initial=0.0) <= cap + 1e-9
        w[rows] = np.where(done[:, None], sub, np.clip(sub, -cap, cap))
        rows = rows[~done]
        if rows.size == 0:
            break
    if rows.size:                                     # out of rounds: keep neutrality and book size exact
        w[rows] = neutral_and_scaled(rows)
    return pd.DataFrame(w, index=raw.index, columns=raw.columns)


def _neutralize(w, ok, codes, n_groups, how):
    """Demean each row across its names (market) or within its groups; None leaves it alone."""
    if how is None:
        return w
    if how == "market":
        n = np.maximum(ok.sum(axis=1, keepdims=True), 1)
        return np.where(ok, w - w.sum(axis=1, keepdims=True) / n, 0.0)
    key = np.arange(w.shape[0])[:, None] * n_groups + np.where(ok, codes, 0)
    size = w.shape[0] * n_groups
    sums = np.bincount(key[ok], weights=w[ok], minlength=size)
    counts = np.bincount(key[ok], minlength=size)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / counts
    return np.where(ok, w - np.nan_to_num(means[key]), 0.0)
