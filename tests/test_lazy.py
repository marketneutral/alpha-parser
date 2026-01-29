"""Tests for lazy data loading."""

import numpy as np
import pandas as pd
import pytest

from alpha_parser import alpha, LazyData, compute_context


class TestLazyData:
    """Test LazyData wrapper."""

    def test_lazy_data_with_dataframe(self, sample_data):
        """LazyData works with regular DataFrames."""
        lazy = LazyData(sample_data)

        # Access should work
        close = lazy['close']
        assert isinstance(close, pd.DataFrame)
        assert close.equals(sample_data['close'])

    def test_lazy_data_with_callable(self, sample_data):
        """LazyData resolves callables on first access."""
        call_count = {'count': 0}

        def load_close():
            call_count['count'] += 1
            return sample_data['close']

        lazy = LazyData({
            'close': load_close,
            'volume': sample_data['volume'],
        })

        # Callable not invoked yet
        assert call_count['count'] == 0

        # First access invokes callable
        close1 = lazy['close']
        assert call_count['count'] == 1
        assert isinstance(close1, pd.DataFrame)

        # Second access uses cache
        close2 = lazy['close']
        assert call_count['count'] == 1  # Not called again
        assert close1 is close2  # Same object

    def test_lazy_data_contains(self, sample_data):
        """LazyData supports 'in' operator."""
        lazy = LazyData({'close': sample_data['close']})

        assert 'close' in lazy
        assert 'nonexistent' not in lazy

    def test_lazy_data_keys(self, sample_data):
        """LazyData supports keys()."""
        lazy = LazyData({
            'close': sample_data['close'],
            'volume': sample_data['volume'],
        })

        assert set(lazy.keys()) == {'close', 'volume'}

    def test_lazy_data_get(self, sample_data):
        """LazyData supports get() with default."""
        lazy = LazyData({'close': sample_data['close']})

        assert lazy.get('close') is not None
        assert lazy.get('nonexistent') is None
        assert lazy.get('nonexistent', 'default') == 'default'

    def test_lazy_data_missing_key(self, sample_data):
        """LazyData raises KeyError for missing keys."""
        lazy = LazyData({'close': sample_data['close']})

        with pytest.raises(KeyError):
            _ = lazy['nonexistent']


class TestLazyDataIntegration:
    """Test lazy loading with signal evaluation."""

    def test_signal_with_lazy_data(self, sample_data):
        """Signals work with LazyData."""
        call_count = {'close': 0, 'volume': 0}

        def load_close():
            call_count['close'] += 1
            return sample_data['close']

        def load_volume():
            call_count['volume'] += 1
            return sample_data['volume']

        lazy = LazyData({
            'close': load_close,
            'volume': load_volume,
        })

        # Signal that only uses close
        signal = alpha("returns(20)")
        result = signal.evaluate(lazy)

        # Only close should be loaded
        assert call_count['close'] == 1
        assert call_count['volume'] == 0
        assert isinstance(result, pd.DataFrame)

    def test_unused_field_not_loaded(self, sample_data):
        """Fields not used by signal are not loaded."""
        loaded = {'earnings': False}

        def load_earnings():
            loaded['earnings'] = True
            return pd.DataFrame(np.nan, index=sample_data['close'].index,
                              columns=sample_data['close'].columns)

        lazy = LazyData({
            'close': sample_data['close'],
            'volume': sample_data['volume'],
            'earnings': load_earnings,
        })

        # Signal that doesn't use earnings
        signal = alpha("rank(returns(20))")
        signal.evaluate(lazy)

        # Earnings should not be loaded
        assert not loaded['earnings']

    def test_lazy_data_with_compute_context(self, sample_data):
        """LazyData works with compute_context caching."""
        call_count = {'count': 0}

        def load_close():
            call_count['count'] += 1
            return sample_data['close']

        lazy = LazyData({'close': load_close})

        with compute_context():
            signal1 = alpha("returns(20)")
            signal2 = alpha("returns(60)")

            result1 = signal1.evaluate(lazy)
            result2 = signal2.evaluate(lazy)

        # Close should only be loaded once
        assert call_count['count'] == 1
