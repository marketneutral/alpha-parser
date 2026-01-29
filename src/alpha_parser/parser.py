"""Alpha expression parser."""

import ast
import operator
from typing import Any, Dict

import pandas as pd

from .signal import Signal
from .operators import Constant, is_valid
from .data import close, open, high, low, field
from .primitives import returns, volatility, volume, adv
from .timeseries import (
    ts_mean, ts_std, ts_sum, ts_max, ts_min, delay, delta, ts_rank, fill_forward,
    ts_corr, ts_cov, ewma, ts_argmax, ts_argmin, ts_skew, ts_kurt, decay_linear,
)
from .crosssection import rank, zscore, demean, quantile, winsorize, scale, truncate
from .conditional import where
from .groups import group_rank, group_demean, group_neutralize, group_count_valid
from .operators import log, abs_, sign, sqrt, power, max_, min_


class AlphaParser:
    """Parse string expressions into Signal objects."""

    def __init__(self):
        self.functions = {
            # Data access
            'close': close,
            'open': open,
            'high': high,
            'low': low,
            'field': field,

            # Primitives
            'returns': returns,
            'volatility': volatility,
            'volume': volume,
            'adv': adv,

            # Time-series ops
            'ts_mean': ts_mean,
            'ts_std': ts_std,
            'ts_sum': ts_sum,
            'ts_max': ts_max,
            'ts_min': ts_min,
            'delay': delay,
            'delta': delta,
            'ts_rank': ts_rank,
            'fill_forward': fill_forward,
            'ts_corr': ts_corr,
            'ts_cov': ts_cov,
            'ewma': ewma,
            'ts_argmax': ts_argmax,
            'ts_argmin': ts_argmin,
            'ts_skew': ts_skew,
            'ts_kurt': ts_kurt,
            'decay_linear': decay_linear,

            # Cross-sectional ops
            'rank': rank,
            'zscore': zscore,
            'demean': demean,
            'quantile': quantile,
            'winsorize': winsorize,
            'scale': scale,
            'truncate': truncate,

            # Conditional
            'where': where,

            # Group ops
            'group_rank': group_rank,
            'group_demean': group_demean,
            'group_neutralize': group_neutralize,
            'group_count_valid': group_count_valid,

            # Validity ops
            'is_valid': is_valid,

            # Math ops
            'log': log,
            'abs': abs_,
            'sign': sign,
            'sqrt': sqrt,
            'power': power,
            'max': max_,
            'min': min_,
        }

        self.binops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.Gt: operator.gt,
            ast.Lt: operator.lt,
            ast.GtE: operator.ge,
            ast.LtE: operator.le,
            ast.Eq: operator.eq,
            ast.NotEq: operator.ne,
            ast.BitAnd: operator.and_,
            ast.BitOr: operator.or_,
        }

        self.unaryops = {
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
            ast.Invert: operator.invert,
        }

    def parse(self, expression: str) -> Signal:
        """Parse string expression into Signal tree."""
        tree = ast.parse(expression, mode='eval')
        return self._visit(tree.body)

    def _visit(self, node: ast.AST) -> Signal:
        """Recursively visit AST nodes and build Signal tree."""

        if isinstance(node, ast.BinOp):
            left = self._visit(node.left)
            right = self._visit(node.right)
            op = self.binops[type(node.op)]
            return op(left, right)

        elif isinstance(node, ast.UnaryOp):
            operand = self._visit(node.operand)
            op = self.unaryops[type(node.op)]
            return op(operand)

        elif isinstance(node, ast.Compare):
            # Handle comparison operators (e.g., a < b, a > b)
            left = self._visit(node.left)
            # Only support single comparisons for now
            if len(node.ops) != 1:
                raise ValueError("Chained comparisons not supported")
            op = self.binops[type(node.ops[0])]
            right = self._visit(node.comparators[0])
            return op(left, right)

        elif isinstance(node, ast.Call):
            func_name = node.func.id
            if func_name not in self.functions:
                raise ValueError(f"Unknown function: {func_name}")

            args = [self._visit_arg(arg) for arg in node.args]

            return self.functions[func_name](*args)

        elif isinstance(node, (ast.Constant, ast.Num)):
            value = node.value if isinstance(node, ast.Constant) else node.n
            return Constant(value)

        elif isinstance(node, ast.Str):
            return node.s

        else:
            raise ValueError(f"Unsupported node type: {type(node)}")

    def _visit_arg(self, node: ast.AST) -> Any:
        """Visit argument node - could be Signal or literal."""
        if isinstance(node, ast.Call):
            return self._visit(node)
        elif isinstance(node, (ast.Constant, ast.Num)):
            return node.value if isinstance(node, ast.Constant) else node.n
        elif isinstance(node, ast.Str):
            return node.s
        else:
            return self._visit(node)


def alpha(expression: str) -> Signal:
    """Parse and return signal from string expression."""
    parser = AlphaParser()
    return parser.parse(expression)


def compute_weights(expression: str,
                    data: Dict[str, pd.DataFrame],
                    **kwargs) -> pd.DataFrame:
    """
    Parse expression and return portfolio weights.

    This is the main entry point for backtesting integration.
    """
    signal = alpha(expression)
    return signal.to_weights(data, **kwargs)
