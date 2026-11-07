# API Reference

Everything a Python program uses from qex is importable from `qex`. The evaluation tools are also importable from
`evaluation`. This page says where each name is documented, and generates the reference for the rest from the code.

| Name | What it is | Documented in |
|------|------------|---------------|
| `qex`, `alpha`, `QexParser` | read an expression into a `Signal` | [Parsing](parsing.md) |
| `Signal`, `compute_context`, `get_context` | evaluate, weights, caching | [Signals](signal.md); the context below |
| `LazyData` | fields loaded on demand | [Large Datasets](../data/lazy.md) |
| `BookSettings`, `build_book`, `hold_untradable`, `turnover`, `fitness`, `margin_bps` | books | [Books](books.md) |
| `price_returns`, `forward_returns`, `MAX_PRICE_GAP` | returns across gaps | [Books](books.md#returns-and-forward-returns) |
| `Backtest`, `BacktestResult`, `QuantileAnalysis`, `QuantileResult` | evaluation | [Evaluation](evaluation.md) |
| `sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `max_drawdown`, `top_drawdowns`, `annualized_return`, `annualized_volatility`, `return_on_gmv` | metrics | [Evaluation](evaluation.md#metrics) |
| `cross_section_universe`, `universe_mask` | universes | [Universes from Python](universes.md) |
| `explain_missing`, `MissingReport`, `qex.adjust.check`, `qex.kinds.numbers_only` | diagnostics | [Diagnostics](diagnostics.md) |
| `FactorRiskModel`, `FactorDefinition`, `DEFAULT_STYLE_FACTORS`, `PRICE_ONLY_FACTORS` | risk model | [Risk Model](risk-model.md) |
| `SignalGraph`, `DistributedSignalEngine` | many signals, shared work, Dask | [Graphs and Dask](distributed.md), and below |

The operator functions (`rank`, `ts_mean`, ...) and node classes (`Rank`, `TsMean`, ...) are importable too, for
building a `Signal` in Python without text; what each operator does is in the
[Operator Reference](../language/reference.md).

## Compute context

::: qex.context.compute_context

::: qex.context.get_context

::: qex.context.ComputeContext

## Graphs

::: qex.distributed.SignalGraph

::: qex.distributed.DistributedSignalEngine
