"""Tests for distributed signal evaluation and graph visualization."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import shutil

from qex import qex, SignalGraph, DistributedSignalEngine, compute_context


def has_graphviz_executable():
    """Check if graphviz dot executable is available."""
    return shutil.which('dot') is not None


requires_graphviz_exe = pytest.mark.skipif(
    not has_graphviz_executable(),
    reason="graphviz executable (dot) not installed"
)


class TestSignalGraph:
    """Tests for SignalGraph DAG construction."""

    def test_single_signal(self, sample_data):
        """Single signal creates correct graph."""
        graph = SignalGraph()
        graph.add_signal(qex("returns(20)"), name="mom")

        assert len(graph.outputs) == 1
        assert "mom" in graph.outputs
        assert len(graph.nodes) >= 1

    def test_shared_subexpressions(self, sample_data):
        """Shared sub-expressions are deduplicated."""
        graph = SignalGraph()

        # Both signals use returns(60)
        graph.add_signal(qex("rank(returns(60))"), name="sig1")
        graph.add_signal(qex("returns(60) / volatility(60)"), name="sig2")

        # Count Returns(60) nodes - should be exactly 1
        returns_nodes = [k for k in graph.nodes.keys()
                        if k[0] == 'Returns' and k[1] == 60]
        assert len(returns_nodes) == 1, "returns(60) should appear only once"

    def test_complex_shared_graph(self, sample_data):
        """Complex signals with many shared parts."""
        graph = SignalGraph()

        expressions = [
            "rank(returns(60))",
            "rank(returns(60) / volatility(60))",
            "rank(-returns(5))",
            "returns(60) * volatility(60)",
            "rank(delta(returns(60), 20))",
        ]

        for i, expr in enumerate(expressions):
            graph.add_signal(qex(expr), name=f"signal_{i}")

        stats = graph.stats()
        assert stats['output_signals'] == 5
        assert stats['shared_nodes'] > 0, "Should have shared nodes"

        # returns(60) should be shared
        returns_60 = [k for k in graph.nodes.keys()
                     if k[0] == 'Returns' and k[1] == 60]
        assert len(returns_60) == 1

    def test_dependencies_tracked(self, sample_data):
        """Dependencies are correctly tracked."""
        graph = SignalGraph()
        graph.add_signal(qex("rank(returns(20))"), name="ranked_mom")

        # Rank depends on Returns
        rank_key = [k for k in graph.nodes.keys() if k[0] == 'Rank'][0]
        returns_key = [k for k in graph.nodes.keys() if k[0] == 'Returns'][0]

        assert returns_key in graph.dependencies[rank_key]

    def test_auto_naming(self, sample_data):
        """Signals get auto-named if no name provided."""
        graph = SignalGraph()
        graph.add_signal(qex("returns(20)"))
        graph.add_signal(qex("returns(60)"))

        assert "signal_0" in graph.outputs
        assert "signal_1" in graph.outputs

    def test_stats(self, sample_data):
        """Stats are computed correctly."""
        graph = SignalGraph()
        graph.add_signal(qex("rank(returns(20))"), name="s1")
        graph.add_signal(qex("rank(returns(20) / volatility(20))"), name="s2")

        stats = graph.stats()
        assert 'total_nodes' in stats
        assert 'output_signals' in stats
        assert 'leaf_nodes' in stats
        assert 'shared_nodes' in stats
        assert stats['output_signals'] == 2

    def test_format_key_simple(self, sample_data):
        """Cache keys are formatted correctly."""
        graph = SignalGraph()

        # Simple function
        assert graph._format_key(('Returns', 20, 'close')) == 'Returns(20)'

        # Nested binary op
        key = ('Add', ('Returns', 20, 'close'), ('Constant', 0.5))
        formatted = graph._format_key(key)
        assert '+' in formatted or 'Add' in formatted

    def test_compute_single_machine(self, sample_data):
        """Compute works on single machine."""
        graph = SignalGraph()
        graph.add_signal(qex("rank(returns(20))"), name="mom")
        graph.add_signal(qex("rank(-returns(5))"), name="rev")

        results = graph.compute(sample_data)

        assert "mom" in results
        assert "rev" in results
        assert isinstance(results["mom"], pd.DataFrame)
        assert results["mom"].shape == sample_data["close"].shape

    def test_compute_subset(self, sample_data):
        """Can compute subset of outputs."""
        graph = SignalGraph()
        graph.add_signal(qex("returns(20)"), name="s1")
        graph.add_signal(qex("returns(60)"), name="s2")
        graph.add_signal(qex("returns(5)"), name="s3")

        results = graph.compute(sample_data, outputs=["s1", "s3"])

        assert "s1" in results
        assert "s3" in results
        assert "s2" not in results


class TestSignalGraphVisualization:
    """Tests for graphviz visualization."""

    @requires_graphviz_exe
    def test_visualize_creates_file(self, sample_data):
        """Visualization creates output file."""
        pytest.importorskip("graphviz")

        graph = SignalGraph()
        graph.add_signal(qex("rank(returns(20))"), name="mom")

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_graph.png")
            result = graph.visualize(filepath)

            assert result is not None
            assert os.path.exists(result)

    @requires_graphviz_exe
    def test_visualize_svg(self, sample_data):
        """Can output SVG format."""
        pytest.importorskip("graphviz")

        graph = SignalGraph()
        graph.add_signal(qex("rank(returns(20))"), name="mom")

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_graph.svg")
            result = graph.visualize(filepath, format='svg')

            assert result is not None
            assert result.endswith('.svg')

    @requires_graphviz_exe
    def test_visualize_complex_graph(self, sample_data):
        """Complex graph can be visualized."""
        pytest.importorskip("graphviz")

        graph = SignalGraph()
        graph.add_signal(qex("rank(returns(60))"), name="mom")
        graph.add_signal(qex("rank(returns(60) / volatility(60))"), name="sharpe")
        graph.add_signal(qex("rank(-returns(5))"), name="rev")

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "complex")
            result = graph.visualize(filepath, format='png')

            assert result is not None
            assert os.path.exists(result)

    def test_visualize_no_graphviz(self, sample_data, monkeypatch):
        """Raises ImportError when graphviz not available."""
        import sys

        # Hide graphviz module
        monkeypatch.setitem(sys.modules, 'graphviz', None)

        graph = SignalGraph()
        graph.add_signal(qex("returns(20)"))

        with pytest.raises(ImportError, match="graphviz"):
            graph.visualize("test.png")

    @requires_graphviz_exe
    def test_visualize_options(self, sample_data):
        """Visualization options work."""
        pytest.importorskip("graphviz")

        graph = SignalGraph()
        graph.add_signal(qex("rank(returns(20))"), name="mom")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test rankdir
            filepath = os.path.join(tmpdir, "lr")
            result = graph.visualize(filepath, format='png', rankdir='LR')
            assert result is not None

            # Test without stats
            filepath2 = os.path.join(tmpdir, "no_stats")
            result2 = graph.visualize(filepath2, format='png', show_stats=False)
            assert result2 is not None


class TestDistributedSignalEngine:
    """Tests for DistributedSignalEngine."""

    def test_local_evaluation(self, sample_data):
        """Engine works in local mode (no dask)."""
        engine = DistributedSignalEngine(client=None)
        engine.load_data(sample_data)

        expressions = [
            "rank(returns(60))",
            "rank(-returns(5))",
        ]

        results = engine.evaluate(expressions)

        assert len(results) == 2
        assert "signal_0" in results
        assert "signal_1" in results

    def test_custom_names(self, sample_data):
        """Can provide custom names for signals."""
        engine = DistributedSignalEngine(client=None)
        engine.load_data(sample_data)

        expressions = ["rank(returns(60))", "rank(-returns(5))"]
        names = ["momentum", "reversal"]

        results = engine.evaluate(expressions, names=names)

        assert "momentum" in results
        assert "reversal" in results

    def test_context_manager(self, sample_data):
        """Engine works as context manager."""
        with DistributedSignalEngine(client=None) as engine:
            engine.load_data(sample_data)
            results = engine.evaluate(["returns(20)"])
            assert len(results) == 1

    def test_no_data_error(self):
        """Raises error if data not loaded."""
        engine = DistributedSignalEngine(client=None)

        with pytest.raises(ValueError, match="No data loaded"):
            engine.evaluate(["returns(20)"])

    @requires_graphviz_exe
    def test_visualize_method(self, sample_data):
        """Engine can visualize expressions."""
        pytest.importorskip("graphviz")

        engine = DistributedSignalEngine(client=None)

        expressions = ["rank(returns(60))", "rank(-returns(5))"]

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "engine_graph.png")
            result = engine.visualize(expressions, filename=filepath)

            assert result is not None
            assert os.path.exists(result)


class TestDaskIntegration:
    """Tests for Dask distributed integration."""

    def test_to_dask_graph(self, sample_data):
        """Graph converts to Dask format."""
        graph = SignalGraph()
        graph.add_signal(qex("rank(returns(20))"), name="mom")

        dask_graph = graph.to_dask_graph()

        assert isinstance(dask_graph, dict)
        assert len(dask_graph) > 0

        # Each value should be a tuple (function, *deps)
        for key, value in dask_graph.items():
            assert isinstance(value, tuple)
            assert callable(value[0])

    def test_compute_dask_local(self, sample_data):
        """Dask compute works with local cluster."""
        dask_distributed = pytest.importorskip("dask.distributed")

        graph = SignalGraph()
        graph.add_signal(qex("rank(returns(20))"), name="mom")
        graph.add_signal(qex("rank(-returns(5))"), name="rev")

        # Use thread-based client for testing
        with dask_distributed.Client(processes=False) as client:
            results = graph.compute_dask(sample_data, client=client)

        assert "mom" in results
        assert "rev" in results
        assert isinstance(results["mom"], pd.DataFrame)

    def test_compute_dask_creates_client(self, sample_data):
        """Dask compute creates client if not provided."""
        pytest.importorskip("dask.distributed")

        graph = SignalGraph()
        graph.add_signal(qex("returns(20)"), name="ret")

        # Should create and close its own client
        results = graph.compute_dask(sample_data)

        assert "ret" in results


class TestGraphDeduplication:
    """Verify deduplication of shared computations."""

    def test_identical_signals_deduplicated(self, sample_data):
        """Identical signals added multiple times are deduplicated."""
        graph = SignalGraph()

        # Add same signal twice
        graph.add_signal(qex("returns(60)"), name="s1")
        graph.add_signal(qex("returns(60)"), name="s2")

        # Should have only one Returns node
        returns_nodes = [k for k in graph.nodes.keys() if k[0] == 'Returns']
        assert len(returns_nodes) == 1

    def test_let_bindings_deduplicated(self, sample_data):
        """Let bindings that expand to same expression are deduplicated."""
        graph = SignalGraph()

        # These should share returns(60) and volatility(60)
        expr1 = """
            let mom = returns(60), vol = volatility(60)
            in rank(mom / vol)
        """
        expr2 = "rank(returns(60) / volatility(60))"

        graph.add_signal(qex(expr1), name="s1")
        graph.add_signal(qex(expr2), name="s2")

        # Both should share the same underlying nodes
        returns_nodes = [k for k in graph.nodes.keys()
                        if k[0] == 'Returns' and k[1] == 60]
        vol_nodes = [k for k in graph.nodes.keys()
                    if k[0] == 'Volatility' and k[1] == 60]

        assert len(returns_nodes) == 1
        assert len(vol_nodes) == 1

    def test_shared_computation_count(self, sample_data):
        """Many signals sharing returns(60) only compute it once."""
        graph = SignalGraph()

        expressions = [
            "rank(returns(60))",
            "zscore(returns(60))",
            "demean(returns(60))",
            "returns(60) / volatility(60)",
            "delay(returns(60), 1)",
            "delta(returns(60), 5)",
            "ts_mean(returns(60), 20)",
        ]

        for i, expr in enumerate(expressions):
            graph.add_signal(qex(expr), name=f"s{i}")

        # All 7 signals share returns(60)
        returns_nodes = [k for k in graph.nodes.keys()
                        if k[0] == 'Returns' and k[1] == 60]
        assert len(returns_nodes) == 1

        stats = graph.stats()
        # returns(60) should be counted as shared
        assert stats['shared_nodes'] >= 1
