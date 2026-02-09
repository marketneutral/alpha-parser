"""Distributed signal evaluation with Dask and graph visualization."""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Any

import pandas as pd

from .signal import Signal


class SignalGraph:
    """
    Build a unified DAG from multiple signals for efficient computation.

    Deduplicates shared sub-expressions across signals using cache keys.
    Supports visualization with graphviz and execution with Dask.

    Example:
        >>> from qex import qex
        >>> graph = SignalGraph()
        >>> graph.add_signal(qex("rank(returns(60))"), name="mom")
        >>> graph.add_signal(qex("rank(returns(60) / volatility(60))"), name="sharpe")
        >>> graph.visualize("signals.png")  # returns(60) appears once
        >>> results = graph.compute(data)
    """

    def __init__(self):
        # Map cache_key -> Signal instance
        self.nodes: Dict[tuple, Signal] = {}
        # Map cache_key -> list of dependency cache_keys
        self.dependencies: Dict[tuple, List[tuple]] = defaultdict(list)
        # Map user-friendly name -> cache_key for output signals
        self.outputs: Dict[str, tuple] = {}
        # Reverse map for display
        self._output_names: Dict[tuple, str] = {}

    def add_signal(self, signal: Signal, name: Optional[str] = None) -> str:
        """
        Add a signal to the graph.

        Args:
            signal: Signal to add
            name: Optional name for this output signal

        Returns:
            The cache key for this signal (as string for display)
        """
        cache_key = self._add_node(signal)

        # Register as output
        if name is None:
            name = f"signal_{len(self.outputs)}"
        self.outputs[name] = cache_key
        self._output_names[cache_key] = name

        return self._format_key(cache_key)

    def _add_node(self, signal: Signal) -> tuple:
        """Recursively add signal and its dependencies to the graph."""
        cache_key = signal._cache_key()

        if cache_key in self.nodes:
            return cache_key

        # Get child signals (dependencies)
        children = self._get_children(signal)
        child_keys = []

        for child in children:
            child_key = self._add_node(child)
            child_keys.append(child_key)

        self.nodes[cache_key] = signal
        self.dependencies[cache_key] = child_keys

        return cache_key

    def _get_children(self, signal: Signal) -> List[Signal]:
        """Extract child signals from a signal node."""
        children = []

        # Check common attribute patterns
        if hasattr(signal, 'left') and isinstance(signal.left, Signal):
            children.append(signal.left)
        if hasattr(signal, 'right') and isinstance(signal.right, Signal):
            children.append(signal.right)
        if hasattr(signal, 'signal') and isinstance(signal.signal, Signal):
            children.append(signal.signal)
        if hasattr(signal, 'x') and isinstance(signal.x, Signal):
            children.append(signal.x)
        if hasattr(signal, 'y') and isinstance(signal.y, Signal):
            children.append(signal.y)
        if hasattr(signal, 'condition') and isinstance(signal.condition, Signal):
            children.append(signal.condition)
        if hasattr(signal, 'if_true') and isinstance(signal.if_true, Signal):
            children.append(signal.if_true)
        if hasattr(signal, 'if_false') and isinstance(signal.if_false, Signal):
            children.append(signal.if_false)

        return children

    def _format_key(self, key: tuple) -> str:
        """Convert cache key tuple to readable string."""
        if not isinstance(key, tuple) or len(key) == 0:
            return str(key)

        name = key[0]

        # Handle nested keys (binary ops)
        if len(key) >= 2 and isinstance(key[1], tuple):
            # Binary operation like ('Add', left_key, right_key)
            if len(key) == 3:
                left = self._format_key(key[1])
                right = self._format_key(key[2])

                op_symbols = {
                    'Add': '+', 'Sub': '-', 'Mul': '*', 'Div': '/',
                    'Greater': '>', 'Less': '<', 'GreaterEqual': '>=',
                    'LessEqual': '<=', 'Equal': '==', 'NotEqual': '!=',
                    'And': '&', 'Or': '|', 'Max': 'max', 'Min': 'min',
                }

                if name in op_symbols:
                    symbol = op_symbols[name]
                    if symbol in ['+', '-', '*', '/', '>', '<', '>=', '<=', '==', '!=', '&', '|']:
                        return f"({left} {symbol} {right})"
                    else:
                        return f"{symbol}({left}, {right})"

                return f"{name}({left}, {right})"

            # Unary operation like ('Neg', child_key)
            if len(key) == 2:
                child = self._format_key(key[1])
                unary_symbols = {'Neg': '-', 'Not': '~'}
                if name in unary_symbols:
                    return f"{unary_symbols[name]}({child})"
                return f"{name}({child})"

        # Simple function call like ('Returns', 60, 'close')
        args = key[1:]
        if args:
            # Filter out default values for cleaner display
            arg_strs = []
            for arg in args:
                if isinstance(arg, str) and arg in ('close',):
                    continue  # Skip default field names
                arg_strs.append(str(arg))

            if arg_strs:
                return f"{name}({', '.join(arg_strs)})"

        return name

    def stats(self) -> Dict[str, int]:
        """Return statistics about the graph."""
        return {
            'total_nodes': len(self.nodes),
            'output_signals': len(self.outputs),
            'leaf_nodes': sum(1 for deps in self.dependencies.values() if not deps),
            'shared_nodes': sum(1 for key in self.nodes
                               if sum(key in deps for deps in self.dependencies.values()) > 1),
        }

    def to_dask_graph(self) -> Dict[str, Tuple]:
        """
        Convert to Dask-compatible task graph.

        Returns dict suitable for dask.threaded.get() or distributed execution.
        """
        dask_graph = {}

        for cache_key in self.nodes:
            deps = self.dependencies[cache_key]
            label = self._format_key(cache_key)
            dep_labels = [self._format_key(d) for d in deps]

            # Create task tuple: (function, *args)
            # For visualization, we use a placeholder function
            dask_graph[label] = (lambda *args: None,) + tuple(dep_labels)

        return dask_graph

    def visualize(
        self,
        filename: str = 'signal_graph.png',
        format: str = None,
        rankdir: str = 'TB',
        highlight_outputs: bool = True,
        show_stats: bool = True,
    ) -> Optional[str]:
        """
        Render the signal DAG using graphviz.

        Args:
            filename: Output filename (extension determines format if not specified)
            format: Output format ('png', 'svg', 'pdf'). Inferred from filename if None.
            rankdir: Graph direction ('TB'=top-bottom, 'LR'=left-right)
            highlight_outputs: Highlight output signal nodes
            show_stats: Include stats in graph title

        Returns:
            Path to rendered file, or None if graphviz not available

        Requires: pip install graphviz (and graphviz system package)
        """
        try:
            import graphviz
        except ImportError:
            raise ImportError(
                "graphviz package required for visualization. "
                "Install with: pip install graphviz\n"
                "Also install system package: apt install graphviz (Linux) "
                "or brew install graphviz (macOS)"
            )

        # Infer format from filename
        if format is None:
            if '.' in filename:
                format = filename.rsplit('.', 1)[1]
                filename = filename.rsplit('.', 1)[0]
            else:
                format = 'png'

        # Create directed graph
        dot = graphviz.Digraph(comment='Signal Graph')
        dot.attr(rankdir=rankdir)

        # Add title with stats
        if show_stats:
            stats = self.stats()
            title = (
                f"Signal Graph\\n"
                f"{stats['total_nodes']} nodes, "
                f"{stats['output_signals']} outputs, "
                f"{stats['shared_nodes']} shared"
            )
            dot.attr(label=title, labelloc='t', fontsize='14')

        # Node styles
        default_style = {'shape': 'box', 'style': 'rounded'}
        output_style = {'shape': 'box', 'style': 'rounded,filled', 'fillcolor': 'lightblue'}
        leaf_style = {'shape': 'ellipse', 'style': 'filled', 'fillcolor': 'lightgreen'}

        # Add nodes
        for cache_key in self.nodes:
            label = self._format_key(cache_key)
            deps = self.dependencies[cache_key]

            # Choose style
            if highlight_outputs and cache_key in self._output_names:
                style = output_style.copy()
                # Add output name to label
                output_name = self._output_names[cache_key]
                label = f"{output_name}\\n{label}"
            elif not deps:
                style = leaf_style
            else:
                style = default_style

            node_id = str(hash(cache_key))
            dot.node(node_id, label, **style)

        # Add edges
        for cache_key, deps in self.dependencies.items():
            parent_id = str(hash(cache_key))
            for dep_key in deps:
                child_id = str(hash(dep_key))
                dot.edge(child_id, parent_id)

        # Render
        output_path = dot.render(filename, format=format, cleanup=True)
        return output_path

    def compute(
        self,
        data: Dict[str, pd.DataFrame],
        outputs: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute signals using the graph (single-machine, cached).

        Args:
            data: Price/volume data
            outputs: List of output names to compute (default: all)

        Returns:
            Dict mapping output names to result DataFrames
        """
        from .context import compute_context

        if outputs is None:
            outputs = list(self.outputs.keys())

        results = {}
        with compute_context():
            for name in outputs:
                cache_key = self.outputs[name]
                signal = self.nodes[cache_key]
                results[name] = signal.evaluate(data)

        return results

    def compute_dask(
        self,
        data: Dict[str, pd.DataFrame],
        client: Optional[Any] = None,
        outputs: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute signals using Dask distributed.

        Args:
            data: Price/volume data
            client: Dask distributed Client (creates local cluster if None)
            outputs: List of output names to compute (default: all)

        Returns:
            Dict mapping output names to result DataFrames

        Requires: pip install dask distributed
        """
        try:
            from dask.distributed import Client, as_completed
            import dask
        except ImportError:
            raise ImportError(
                "Dask distributed required. Install with: pip install dask distributed"
            )

        # Create local client if needed
        own_client = False
        if client is None:
            client = Client(processes=False)  # Thread-based for shared memory
            own_client = True

        try:
            if outputs is None:
                outputs = list(self.outputs.keys())

            # Scatter data to workers once
            data_future = client.scatter(data, broadcast=True)

            # Build computation graph
            # We need to compute in dependency order
            computed = {}  # cache_key -> future

            def get_result(cache_key):
                """Recursively compute a node and its dependencies."""
                if cache_key in computed:
                    return computed[cache_key]

                signal = self.nodes[cache_key]
                deps = self.dependencies[cache_key]

                # Compute dependencies first
                for dep_key in deps:
                    get_result(dep_key)

                # Submit this computation
                future = client.submit(
                    _evaluate_signal,
                    signal,
                    data_future,
                    key=str(cache_key),
                )
                computed[cache_key] = future
                return future

            # Compute all requested outputs
            output_futures = {}
            for name in outputs:
                cache_key = self.outputs[name]
                output_futures[name] = get_result(cache_key)

            # Gather results
            results = client.gather(output_futures)
            return results

        finally:
            if own_client:
                client.close()


def _evaluate_signal(signal: Signal, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Worker function to evaluate a signal."""
    return signal.evaluate(data)


class DistributedSignalEngine:
    """
    High-level interface for distributed signal evaluation.

    Manages data distribution and signal graph execution across a Dask cluster.

    Example:
        >>> from dask.distributed import Client
        >>> from qex.distributed import DistributedSignalEngine
        >>>
        >>> client = Client("scheduler:8786")
        >>> engine = DistributedSignalEngine(client)
        >>> engine.load_data(data)
        >>>
        >>> expressions = [
        ...     "rank(returns(60))",
        ...     "rank(returns(60) / volatility(60))",
        ...     "rank(-returns(5))",
        ... ]
        >>> results = engine.evaluate(expressions)
    """

    def __init__(self, client: Optional[Any] = None):
        """
        Initialize the engine.

        Args:
            client: Dask distributed Client (creates local if None)
        """
        self.client = client
        self._own_client = False
        self._data_future = None

        if client is None:
            try:
                from dask.distributed import Client
                self.client = Client(processes=False)
                self._own_client = True
            except ImportError:
                pass  # Will use local computation

    def load_data(self, data: Dict[str, pd.DataFrame]):
        """
        Scatter data to all workers.

        Call this once before evaluating signals.
        Data is cached on workers for efficient reuse.
        """
        if self.client is None:
            self._data = data
            return

        self._data_future = self.client.scatter(data, broadcast=True)

    def evaluate(
        self,
        expressions: List[str],
        names: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Evaluate multiple signal expressions efficiently.

        Args:
            expressions: List of qex expression strings
            names: Optional names for each expression

        Returns:
            Dict mapping names to result DataFrames
        """
        from .parser import qex

        if names is None:
            names = [f"signal_{i}" for i in range(len(expressions))]

        # Build unified graph
        graph = SignalGraph()
        for expr, name in zip(expressions, names):
            signal = qex(expr)
            graph.add_signal(signal, name=name)

        # Compute
        if self.client is not None and self._data_future is not None:
            return graph.compute_dask(
                self.client.gather(self._data_future),
                self.client
            )
        elif hasattr(self, '_data'):
            return graph.compute(self._data)
        else:
            raise ValueError("No data loaded. Call load_data() first.")

    def visualize(self, expressions: List[str], **kwargs) -> str:
        """Build graph from expressions and visualize."""
        from .parser import qex

        graph = SignalGraph()
        for i, expr in enumerate(expressions):
            signal = qex(expr)
            graph.add_signal(signal, name=f"signal_{i}")

        return graph.visualize(**kwargs)

    def close(self):
        """Close the Dask client if we created it."""
        if self._own_client and self.client is not None:
            self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
