"""Alpha expression parser."""

import ast
import operator
import re
from typing import Any, Dict, Optional

import pandas as pd

from .signal import Signal
from .operators import Constant, is_valid
from .data import close, open, high, low, field, day_of_week, day_of_month, month_of_year
from .primitives import returns, volatility, volume, adv
from .timeseries import (
    ts_mean, ts_std, ts_sum, ts_max, ts_min, delay, delta, ts_rank, fill_forward,
    ts_corr, ts_cov, ts_var, ewma, ewma_var, ewma_cov, ts_beta, ts_beta_ewma,
    ts_argmax, ts_argmin, ts_skew, ts_kurt, decay_linear,
    ts_mean_events, ts_std_events, ts_sum_events, ts_count_events,
)
from .crosssection import rank, zscore, demean, quantile, winsorize, scale, truncate
from .conditional import where
from .groups import group_rank, group_demean, group_count_valid, group_std, group_sum
from .operators import log, abs_, sign, sqrt, power, max_, min_


class AlphaParser:
    """Parse string expressions into Signal objects.

    Supports variable bindings with let...in syntax:
        let s = returns(20) in rank(delta(s, 20)) * s

    Multiple bindings can be comma-separated:
        let mom = returns(20), vol = volatility(60) in rank(mom / vol) * sign(mom)

    Later bindings can reference earlier ones:
        let mom = returns(20), sharpe = mom / volatility(60) in rank(sharpe) * sign(mom)
    """

    def __init__(self):
        self.variables: Dict[str, Signal] = {}
        self.functions = {
            # Data access
            'close': close,
            'open': open,
            'high': high,
            'low': low,
            'field': field,

            # Calendar
            'day_of_week': day_of_week,
            'day_of_month': day_of_month,
            'month_of_year': month_of_year,

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
            'ts_var': ts_var,
            'delay': delay,
            'delta': delta,
            'ts_rank': ts_rank,
            'fill_forward': fill_forward,
            'ts_corr': ts_corr,
            'ts_cov': ts_cov,
            'ewma': ewma,
            'ewma_var': ewma_var,
            'ewma_cov': ewma_cov,
            'ts_beta': ts_beta,
            'ts_beta_ewma': ts_beta_ewma,
            'ts_argmax': ts_argmax,
            'ts_argmin': ts_argmin,
            'ts_skew': ts_skew,
            'ts_kurt': ts_kurt,
            'decay_linear': decay_linear,

            # Event-based time-series ops (roll over N non-NaN values)
            'ts_mean_events': ts_mean_events,
            'ts_std_events': ts_std_events,
            'ts_sum_events': ts_sum_events,
            'ts_count_events': ts_count_events,

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
            'group_count_valid': group_count_valid,
            'group_std': group_std,
            'group_sum': group_sum,

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
        """Parse string expression into Signal tree.

        Handles let...in syntax for variable bindings before parsing.
        """
        # Reset variables for each parse
        self.variables = {}

        # Handle let...in syntax
        expression = expression.strip()
        if expression.startswith('let '):
            return self._parse_let(expression)

        tree = ast.parse(expression, mode='eval')
        return self._visit(tree.body)

    def _parse_let(self, expression: str) -> Signal:
        """Parse let...in expression with variable bindings.

        Syntax: let var1 = expr1, var2 = expr2 in body

        Args:
            expression: Full expression starting with 'let '

        Returns:
            Signal from evaluating body with variables bound
        """
        # Find 'in' keyword that separates bindings from body
        # Need to account for nested parentheses
        in_pos = self._find_in_keyword(expression)
        if in_pos == -1:
            raise ValueError("Invalid let expression: missing 'in' keyword")

        bindings_str = expression[4:in_pos].strip()  # Skip 'let '
        body_str = expression[in_pos + 3:].strip()   # Skip ' in '

        # Parse comma-separated bindings
        bindings = self._split_bindings(bindings_str)

        for name, expr in bindings:
            name = name.strip()
            expr = expr.strip()

            # Validate variable name
            if not name.isidentifier():
                raise ValueError(f"Invalid variable name: {name}")
            if name in self.functions:
                raise ValueError(f"Cannot shadow function name: {name}")

            # Parse the binding expression (may reference earlier variables)
            signal = self.parse(expr) if expr.startswith('let ') else self._parse_expr(expr)
            self.variables[name] = signal

        # Parse the body with all variables in scope
        return self._parse_expr(body_str)

    def _parse_expr(self, expression: str) -> Signal:
        """Parse an expression (without let handling)."""
        tree = ast.parse(expression, mode='eval')
        return self._visit(tree.body)

    def _find_in_keyword(self, expression: str) -> int:
        """Find the 'in' keyword that separates bindings from body.

        Must handle nested parentheses and avoid matching 'in' inside strings
        or function names like 'winsorize'.
        """
        depth = 0
        i = 4  # Start after 'let '

        while i < len(expression) - 1:
            char = expression[i]

            if char in '([{':
                depth += 1
            elif char in ')]}':
                depth -= 1
            elif depth == 0 and expression[i:i+3] == ' in':
                # Check it's a standalone 'in' (not part of another word)
                after = i + 3
                if after >= len(expression) or not expression[after].isalnum():
                    return i + 1  # Return position of 'in'

            i += 1

        return -1

    def _split_bindings(self, bindings_str: str) -> list:
        """Split comma-separated bindings, respecting parentheses.

        Returns list of (name, expression) tuples.
        """
        bindings = []
        current = []
        depth = 0

        for char in bindings_str:
            if char in '([{':
                depth += 1
                current.append(char)
            elif char in ')]}':
                depth -= 1
                current.append(char)
            elif char == ',' and depth == 0:
                # Split here
                binding = ''.join(current).strip()
                if binding:
                    name, expr = self._parse_binding(binding)
                    bindings.append((name, expr))
                current = []
            else:
                current.append(char)

        # Don't forget the last binding
        binding = ''.join(current).strip()
        if binding:
            name, expr = self._parse_binding(binding)
            bindings.append((name, expr))

        return bindings

    def _parse_binding(self, binding: str) -> tuple:
        """Parse a single 'name = expr' binding."""
        eq_pos = binding.find('=')
        if eq_pos == -1:
            raise ValueError(f"Invalid binding (missing '='): {binding}")

        name = binding[:eq_pos].strip()
        expr = binding[eq_pos + 1:].strip()
        return name, expr

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

        elif isinstance(node, ast.Name):
            # Variable reference
            name = node.id
            if name in self.variables:
                return self.variables[name]
            raise ValueError(f"Unknown variable: {name}")

        else:
            raise ValueError(f"Unsupported node type: {type(node)}")

    def _visit_arg(self, node: ast.AST) -> Any:
        """Visit argument node - could be Signal, variable, or literal."""
        if isinstance(node, ast.Call):
            return self._visit(node)
        elif isinstance(node, (ast.Constant, ast.Num)):
            return node.value if isinstance(node, ast.Constant) else node.n
        elif isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.Name):
            # Could be a variable reference
            name = node.id
            if name in self.variables:
                return self.variables[name]
            raise ValueError(f"Unknown variable: {name}")
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
