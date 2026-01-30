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


class TestSelfDescribingData:
    """Test self-describing data with field descriptions."""

    def test_descriptions_property(self, sample_data):
        """LazyData exposes descriptions via property."""
        descriptions = {
            'close': 'Daily closing price',
            'volume': 'Daily trading volume',
        }
        lazy = LazyData(sample_data, descriptions=descriptions)

        assert lazy.descriptions == descriptions

    def test_descriptions_default_empty(self, sample_data):
        """Descriptions default to empty dict if not provided."""
        lazy = LazyData(sample_data)

        assert lazy.descriptions == {}

    def test_describe_all_fields(self, sample_data):
        """describe() returns formatted string of all fields."""
        descriptions = {
            'close': 'Daily closing price, adjusted for splits',
            'volume': 'Daily trading volume in shares',
        }
        lazy = LazyData(
            {'close': sample_data['close'], 'volume': sample_data['volume']},
            descriptions=descriptions,
        )

        output = lazy.describe()

        assert 'Available fields:' in output
        assert 'close' in output
        assert 'Daily closing price' in output
        assert 'volume' in output
        assert 'Daily trading volume' in output

    def test_describe_single_field(self, sample_data):
        """describe(field) returns description for single field."""
        descriptions = {'close': 'Daily closing price'}
        lazy = LazyData({'close': sample_data['close']}, descriptions=descriptions)

        output = lazy.describe('close')

        assert 'close' in output
        assert 'Daily closing price' in output

    def test_describe_missing_description(self, sample_data):
        """Fields without descriptions show default message."""
        lazy = LazyData({'close': sample_data['close']}, descriptions={})

        output = lazy.describe()

        assert 'No description available' in output

    def test_describe_missing_field(self, sample_data):
        """describe(field) handles missing field gracefully."""
        lazy = LazyData({'close': sample_data['close']})

        output = lazy.describe('nonexistent')

        assert 'not found' in output
        assert 'close' in output  # Shows available fields

    def test_describe_shows_dtype_when_loaded(self, sample_data):
        """describe() shows dtype info for loaded fields."""
        lazy = LazyData(
            {'close': sample_data['close']},
            descriptions={'close': 'Closing price'},
        )

        # Load the field
        _ = lazy['close']

        output = lazy.describe()

        assert 'dtype' in output

    def test_repr(self, sample_data):
        """LazyData has informative repr."""
        descriptions = {'close': 'Closing price'}
        lazy = LazyData(
            {'close': sample_data['close'], 'volume': sample_data['volume']},
            descriptions=descriptions,
        )

        repr_str = repr(lazy)

        assert 'LazyData' in repr_str
        assert '2 fields' in repr_str
        assert '1 with descriptions' in repr_str

    def test_describe_with_callable_not_loaded(self, sample_data):
        """describe() works even when data is lazy-loaded callables."""
        call_count = {'count': 0}

        def load_close():
            call_count['count'] += 1
            return sample_data['close']

        lazy = LazyData(
            {'close': load_close},
            descriptions={'close': 'Closing price (lazy loaded)'},
        )

        # describe() should NOT trigger loading
        output = lazy.describe()

        assert call_count['count'] == 0
        assert 'close' in output
        assert 'Closing price' in output
        # No dtype since not loaded yet
        assert 'dtype' not in output

    def test_full_workflow_with_descriptions(self, sample_data):
        """End-to-end test: create data, describe, evaluate signal."""
        lazy = LazyData(
            data={
                'close': sample_data['close'],
                'volume': sample_data['volume'],
            },
            descriptions={
                'close': 'Daily closing price, adjusted for splits and dividends',
                'volume': 'Daily trading volume in shares',
            },
        )

        # Agent can inspect available data
        desc = lazy.describe()
        assert 'close' in desc
        assert 'volume' in desc

        # Agent builds and evaluates signal
        signal = alpha("rank(returns(20))")
        result = signal.evaluate(lazy)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_data['close'].shape
