"""qex and evaluation import in either order, and qex still re-exports the evaluation names."""

import subprocess
import sys

import pytest

EVALUATION_NAMES = [
    'Backtest', 'BacktestResult', 'BookSettings', 'build_book', 'QuantileAnalysis', 'QuantileResult',
    'sharpe_ratio', 'max_drawdown', 'top_drawdowns', 'return_on_gmv', 'annualized_return',
    'annualized_volatility', 'calmar_ratio', 'sortino_ratio', 'turnover', 'fitness', 'margin_bps',
]


def run(code):
    """Run code in a fresh interpreter, so no module is already imported."""
    return subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)


@pytest.mark.parametrize('code', [
    'import evaluation',
    'from evaluation import Backtest, build_book',
    'import evaluation; import qex',
    'import qex; import evaluation',
    'from qex import Backtest; from evaluation import Backtest as B; assert Backtest is B',
])
def test_import_order(code):
    result = run(code)
    assert result.returncode == 0, result.stderr


def test_qex_reexports_evaluation():
    import qex
    import evaluation
    for name in EVALUATION_NAMES:
        assert getattr(qex, name) is getattr(evaluation, name)
        assert name in qex.__all__
        assert name in dir(qex)


def test_star_import_includes_evaluation():
    result = run('from qex import *; Backtest; BookSettings; fitness')
    assert result.returncode == 0, result.stderr


def test_unknown_attribute_still_raises():
    import qex
    with pytest.raises(AttributeError):
        qex.not_a_name
