# Distributed Signal Evaluation

Efficiently evaluate thousands of signals with shared sub-expression deduplication.

## Overview

When evaluating many signals, they often share common sub-expressions. For example, `rank(returns(60))` and `rank(returns(60) / volatility(60))` both compute `returns(60)`. The `SignalGraph` class builds a unified DAG that computes each unique sub-expression only once.

## Quick Start

```python
from qex import qex, SignalGraph

# Build a graph from multiple signals
graph = SignalGraph()
graph.add_signal(qex("rank(returns(60))"), name="momentum")
graph.add_signal(qex("rank(returns(60) / volatility(60))"), name="sharpe")
graph.add_signal(qex("rank(-returns(5))"), name="reversal")

# See what's shared
print(graph.stats())
# {'total_nodes': 7, 'output_signals': 3, 'leaf_nodes': 2, 'shared_nodes': 1}

# Compute all signals
results = graph.compute(data)
print(results.keys())  # dict_keys(['momentum', 'sharpe', 'reversal'])
```

## Visualization

Visualize the computation graph with graphviz:

```python
# Render to PNG
graph.visualize("signals.png")

# SVG for web
graph.visualize("signals.svg", format="svg")

# Horizontal layout
graph.visualize("signals.png", rankdir="LR")
```

Requirements:
```bash
pip install graphviz
# Plus system package:
# Linux: apt install graphviz
# macOS: brew install graphviz
```

Example output shows:
- **Green ellipses**: Leaf nodes (data access like `returns(60)`)
- **Blue boxes**: Output signals
- **White boxes**: Intermediate computations
- **Arrows**: Dependencies (child → parent)

## Distributed Evaluation with Dask

For large-scale evaluation across a cluster:

```python
from dask.distributed import Client
from qex import DistributedSignalEngine

# Connect to Dask cluster
client = Client("scheduler:8786")

# Create engine
engine = DistributedSignalEngine(client)

# Load data once (scattered to all workers)
engine.load_data(data)

# Evaluate many signals
expressions = [
    "rank(returns(60))",
    "rank(returns(60) / volatility(60))",
    "rank(-returns(5))",
    "rank(delta(returns(20), 5))",
    # ... thousands more
]

results = engine.evaluate(expressions)
```

### Data Distribution Strategies

**Scatter once (recommended)**:
```python
engine.load_data(data)  # Broadcasts to all workers
```

Data is transferred once and cached on each worker. All tasks reference the same copy.

**Shared storage** (for very large data):
```python
# Workers load from S3/GCS/NFS on demand
# Configure via Dask's standard mechanisms
```

## How Deduplication Works

The `SignalGraph` uses cache keys to identify identical sub-expressions:

```python
# These two signals
graph.add_signal(qex("rank(returns(60))"))
graph.add_signal(qex("rank(returns(60) / volatility(60))"))

# Share the same returns(60) node because
qex("returns(60)")._cache_key()  # ('Returns', 60, 'close')
```

Even `let` bindings that expand to the same expression are deduplicated:

```python
# Both produce identical graphs
qex("let mom = returns(60) in rank(mom)")
qex("rank(returns(60))")
```

## API Reference

### SignalGraph

```python
class SignalGraph:
    def add_signal(signal, name=None) -> str
        """Add a signal to the graph."""

    def stats() -> dict
        """Return graph statistics."""

    def visualize(filename, format=None, rankdir='TB', ...) -> str
        """Render graph with graphviz."""

    def compute(data, outputs=None) -> dict
        """Compute signals (single machine, cached)."""

    def compute_dask(data, client=None, outputs=None) -> dict
        """Compute signals with Dask distributed."""
```

### DistributedSignalEngine

```python
class DistributedSignalEngine:
    def __init__(client=None)
        """Initialize with optional Dask client."""

    def load_data(data)
        """Scatter data to workers."""

    def evaluate(expressions, names=None) -> dict
        """Evaluate multiple expressions efficiently."""

    def visualize(expressions, **kwargs) -> str
        """Build graph from expressions and visualize."""
```

## Performance Tips

1. **Maximize sharing**: Design signals with common building blocks
   ```python
   # Good: returns(60) computed once
   signals = [
       "rank(returns(60))",
       "zscore(returns(60))",
       "demean(returns(60))",
   ]

   # Less efficient: returns computed 3x
   signals = [
       "rank(returns(60))",
       "rank(returns(61))",
       "rank(returns(62))",
   ]
   ```

2. **Use `let` bindings**: Makes sharing explicit in expressions
   ```python
   expr = """
       let mom = returns(60), vol = volatility(60)
       in rank(mom / vol) * sign(mom)
   """
   ```

3. **Visualize first**: Check your graph before large runs
   ```python
   graph.visualize("check.png")
   print(f"Total nodes: {graph.stats()['total_nodes']}")
   ```

4. **Batch by similarity**: Group signals that share sub-expressions
