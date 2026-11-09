"""qex parser."""

import ast
import operator
import re
from typing import Any, Dict, Optional

import pandas as pd

from .signal import Signal
from .operators import Constant, NoneValue, is_valid, is_finite
from .data import close, open, high, low, field, market, day_of_week, day_of_month, month_of_year
from .primitives import returns, volatility, volume, adv
from .timeseries import (
    ts_mean, ts_std, ts_sum, ts_max, ts_min, delay, delta, ts_rank, fill_forward, latch, days_since,
    ts_corr, ts_cov, ts_var, ewma, ewma_var, ewma_cov, ts_beta, ts_beta_shrunk, ts_beta_ewma,
    ts_alpha, ts_alpha_shrunk, ts_resid, ts_resid_shrunk,
    ts_argmax, ts_argmin, ts_skew, ts_kurt, decay_linear, ts_quantile,
    ts_mean_events, ts_std_events, ts_sum_events, ts_count_events,
)
from .match import ts_match, ts_match_score
from .crosssection import (gaussrank, rank, zscore, demean, quantile, winsorize, scale, truncate, regression_neut, pca_factor,
    regression_alpha, regression_beta)
from .conditional import where, coalesce
from .groups import (group, group_rank, group_demean, group_count_valid, group_std, group_sum, group_quantile,
                     group_cross, identifier, group_gaussrank)
from .operators import log, exp, abs_, sign, sqrt, power, max_, min_


class QexParser:
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
            'market': market,

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
            'ts_quantile': ts_quantile,
            'fill_forward': fill_forward,
            'ts_corr': ts_corr,
            'ts_cov': ts_cov,
            'ewma': ewma,
            'ewma_var': ewma_var,
            'ewma_cov': ewma_cov,
            'ts_beta': ts_beta,
            'ts_beta_shrunk': ts_beta_shrunk,
            'ts_beta_ewma': ts_beta_ewma,
            'ts_alpha': ts_alpha,
            'ts_alpha_shrunk': ts_alpha_shrunk,
            'ts_resid': ts_resid,
            'ts_resid_shrunk': ts_resid_shrunk,
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
            'regression_neut': regression_neut,
            'regression_alpha': regression_alpha,
            'regression_beta': regression_beta,
            'ts_match': ts_match,
            'ts_match_score': ts_match_score,
            'identifier': identifier,
            'gaussrank': gaussrank,
            'group_gaussrank': group_gaussrank,
            'pca_factor': pca_factor,

            # Conditional
            'where': where,
            'coalesce': coalesce,
            'latch': latch,
            'days_since': days_since,

            # Group ops
            'group': group,
            'group_rank': group_rank,
            'group_demean': group_demean,
            'group_count_valid': group_count_valid,
            'group_std': group_std,
            'group_sum': group_sum,
            'group_quantile': group_quantile,
            'group_cross': group_cross,

            # Validity ops
            'is_valid': is_valid,
            'is_finite': is_finite,

            # Math ops
            'log': log,
            'exp': exp,
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
            ast.UAdd: lambda x: x,                     # +x is x
            ast.Invert: operator.invert,
        }

    def parse(self, expression: str) -> Signal:
        """Parse string expression into Signal tree.

        Handles let...in syntax for variable bindings before parsing.
        """
        # Reset variables for each parse
        self.variables = {}

        # Whitespace is insignificant: line breaks may fall anywhere, even after an operator
        expression = _collapse_whitespace(expression)

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
            if type(node.op) not in self.binops:
                raise ValueError(f"{_OP_NAMES.get(type(node.op), type(node.op).__name__)} is not a qex operator: qex has "
                                 f"+ - * / ** for arithmetic, comparisons, and & | ~ for logic")
            left = self._visit(node.left)
            right = self._visit(node.right)
            op = self.binops[type(node.op)]
            return op(left, right)

        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                raise ValueError("'not' is not qex: write ~x, as in ~(returns(5) > 0)")
            operand = self._visit(node.operand)
            op = self.unaryops[type(node.op)]
            return op(operand)

        elif isinstance(node, ast.BoolOp):
            word = "and" if isinstance(node.op, ast.And) else "or"
            sym = "&" if word == "and" else "|"
            raise ValueError(f"'{word}' is not qex: write {sym}, with each comparison in parentheses, "
                             f"as in (returns(5) > 0) {sym} (volume(1) > volume(20))")

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
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only operators are called in qex, by name, as in rank(x)")
            func_name = node.func.id
            if func_name not in self.functions:
                raise ValueError(f"Unknown function: {func_name}")
            if any(isinstance(a, ast.Starred) for a in node.args):
                raise ValueError(f"{func_name}() takes its arguments written out, not *unpacked")
            kwargs = self._keywords(func_name, node)

            args = _check_args(func_name, [self._visit_arg(arg) for arg in node.args])

            return self.functions[func_name](*args, **kwargs)

        elif isinstance(node, (ast.Constant, ast.Num)):
            value = node.value if isinstance(node, ast.Constant) else node.n
            if value is None:
                return NoneValue()
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
            raise ValueError(_unsupported(node))

    def _keywords(self, func_name: str, node: ast.Call) -> dict:
        """A call's keyword arguments: only operators listed in KEYWORDS take them, and only the names listed."""
        if not node.keywords:
            return {}
        allowed = KEYWORDS.get(func_name)
        first = node.keywords[0].arg
        if allowed is None:
            raise ValueError(f"{func_name}() takes its arguments by position, not by name: write "
                             f"{func_name}(x, 20), not {func_name}(x, {first or '...'}=20)")
        out = {}
        for kw in node.keywords:
            if kw.arg is None:
                raise ValueError(f"{func_name}() takes its arguments written out, not **unpacked")
            if kw.arg not in allowed:
                raise ValueError(f"{func_name}() has no argument named {kw.arg!r}; by name it takes "
                                 f"{', '.join(allowed)}")
            if kw.arg in out:
                raise ValueError(f"{func_name}()'s {kw.arg} is given twice")
            out[kw.arg] = self._visit_arg(kw.value)
        positional = len(node.args)
        order = POSITIONAL_ORDER.get(func_name, ())
        for name in out:
            if name in order and order.index(name) < positional:
                raise ValueError(f"{func_name}()'s {name} is given both by position and by name")
        return out

    def _visit_arg(self, node: ast.AST) -> Any:
        """Visit argument node - could be Signal, variable, or literal."""
        if isinstance(node, ast.Call):
            return self._visit(node)
        elif isinstance(node, ast.Constant) and node.value is None:
            return NoneValue()                         # None as an argument is the literal, a signal
        elif isinstance(node, (ast.Constant, ast.Num)):
            return node.value if isinstance(node, ast.Constant) else node.n
        elif (isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd))
              and isinstance(node.operand, ast.Constant)
              and isinstance(node.operand.value, (int, float))
              and not isinstance(node.operand.value, bool)):
            # A signed number literal (delay(x, -1), power(x, -2)) is a number, not a Neg signal
            value = node.operand.value
            return -value if isinstance(node.op, ast.USub) else value
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


def _strip_comments(expression: str) -> str:
    """Remove each `# comment` to the end of its line; a # inside a quoted string is kept."""
    out = []
    for line in expression.split("\n"):
        quote, i = None, 0
        while i < len(line):
            ch = line[i]
            if quote:
                if ch == "\\":
                    i += 1
                elif ch == quote:
                    quote = None
            elif ch in "'\"":
                quote = ch
            elif ch == "#":
                line = line[:i]
                break
            i += 1
        out.append(line)
    return "\n".join(out)


def _collapse_whitespace(expression: str) -> str:
    """Remove `#` comments, then turn every run of whitespace outside quoted strings into one space.

    Comments go first, line by line: joining the lines first would let a comment swallow the lines after it."""
    expression = _strip_comments(expression)
    parts = re.split(r"""('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")""", expression)
    return "".join(p if i % 2 else re.sub(r"\s+", " ", p) for i, p in enumerate(parts))


# What each operator's constant arguments must be, by position: a window of days (a whole number, at least 1), a lag
# (whole, at least 0), a count (whole, at least 1), a name (a quoted string), a half-life (a number above 0), a level in
# [0, 1], a winsorizing level in [0, 0.5], a positive number. "x" is an expression and is not checked here.
# Operators that take keyword arguments, and the names they take by name. Every other operator takes its arguments by
# position only. (Since 0.15.0; ts_match's five options would be unreadable by position.)
KEYWORDS = {
    "ts_match": ("metric", "within", "matching", "rebalance", "min_overlap"),
    "ts_match_score": ("metric", "within", "matching", "rebalance", "min_overlap", "stat"),
}
POSITIONAL_ORDER = {
    "ts_match": ("value", "key", "lookback", "metric", "within", "matching", "rebalance", "min_overlap"),
    "ts_match_score": ("key", "lookback", "metric", "within", "matching", "rebalance", "min_overlap", "stat"),
}

_ARG_RULES = {
    "returns": ("window", "name"), "volatility": ("window", "name"), "volume": ("window",), "adv": ("window",),
    "field": ("name",), "group": ("name",),
    **{f: ("x", "window") for f in ("ts_mean", "ts_std", "ts_sum", "ts_max", "ts_min", "ts_var", "ts_rank",
                                   "ts_argmax", "ts_argmin", "ts_skew", "ts_kurt", "decay_linear", "ts_count_events")},
    # ts_quantile, pca_factor, group_quantile and group_count_valid check their own arguments, with their own messages
    "ts_quantile": ("x", "x", "x", "count"),
    "delay": ("x", "lag"), "delta": ("x", "lag"),
    **{f: ("x", "x", "window", "count") for f in ("ts_corr", "ts_cov", "ts_beta")},
    # ts_alpha and ts_resid take any number of regressors, (y, x1, ..., xk, period[, min_periods]): checked there
    # the shrunk fits take (y, x, period[, min_periods][, groups]): a 4th argument is min_periods or the groups
    **{f: ("x", "x", "window", "count_or_groups", "groups")
       for f in ("ts_beta_shrunk", "ts_alpha_shrunk", "ts_resid_shrunk")},
    "ewma": ("x", "halflife"), "ewma_var": ("x", "halflife"), "ewma_cov": ("x", "x", "halflife"),
    "ts_beta_ewma": ("x", "x", "halflife"),
    **{f: ("x", "count", "count") for f in ("ts_mean_events", "ts_std_events", "ts_sum_events")},
    "fill_forward": ("x", "count"),
    "quantile": ("x", "count"), "winsorize": ("x", "winsor"), "truncate": ("x", "positive"),
}

_RULE_TEXT = {"window": "a whole number of days, at least 1", "lag": "a whole number of days, 0 or more",
              "count": "a whole number, at least 1", "name": "a name in quotes, as in 'sector'",
              "halflife": "a number of days above 0", "level": "a number from 0 to 1",
              "winsor": "a number from 0 to 0.5", "positive": "a number above 0",
              "count_or_groups": "a whole number, at least 1 (min_periods), or groups ('sector')"}

_ORDINAL = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}


def _check_args(func_name: str, args: list) -> list:
    """The arguments, each constant one checked against its operator's rule. Values pass through unchanged (20.0 stays
    20.0: an expression's cache key, which callers may hash as its identity, must not change). Raises ValueError naming
    the operator, the argument and what it must be."""
    rules = _ARG_RULES.get(func_name, ())
    out = list(args)
    for i, (rule, value) in enumerate(zip(rules, args)):
        if rule == "x":
            continue
        if rule in ("count_or_groups", "groups"):
            # a group's name or labels expression; the shrunk fits sort out min_periods from groups themselves,
            # with messages that say which comes first
            if isinstance(value, (str, Signal)) or rule == "groups":
                continue
            rule = "count"
        ok = True
        if rule == "name":
            ok = isinstance(value, str)
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            ok = False
        elif rule in ("window", "lag", "count"):
            # a negative lag is left to delay() and delta(), whose message explains the look-ahead
            ok = float(value).is_integer() and (rule == "lag" or value >= 1)
        elif rule in ("halflife", "positive"):
            ok = value > 0
        elif rule == "level":
            ok = 0 <= value <= 1
        elif rule == "winsor":
            ok = 0 <= value <= 0.5
        if not ok:
            shown = "an expression" if isinstance(value, Signal) else repr(value)
            raise ValueError(f"{func_name}()'s {_ORDINAL.get(i + 1, f'argument {i + 1}')} argument must be "
                             f"{_RULE_TEXT[rule]} (got {shown})")
    return out


_OP_NAMES = {ast.Mod: "'%'", ast.FloorDiv: "'//'", ast.MatMult: "'@'", ast.LShift: "'<<'", ast.RShift: "'>>'",
             ast.BitXor: "'^'"}

_NODE_NAMES = {ast.List: "a list [...]", ast.Tuple: "a tuple", ast.Dict: "a dict {...}", ast.Set: "a set {...}",
               ast.Subscript: "indexing x[...]", ast.Attribute: "an attribute x.y", ast.IfExp: "'if ... else'",
               ast.Lambda: "lambda", ast.ListComp: "a comprehension", ast.GeneratorExp: "a comprehension",
               ast.DictComp: "a comprehension", ast.SetComp: "a comprehension", ast.JoinedStr: "an f-string",
               ast.Await: "await", ast.NamedExpr: "':='", ast.Starred: "*unpacking"}


def _unsupported(node) -> str:
    what = _NODE_NAMES.get(type(node), type(node).__name__)
    hint = " (use where(cond, a, b))" if isinstance(node, ast.IfExp) else ""
    return f"{what} is not part of qex{hint}"


# Backwards compatibility alias
AlphaParser = QexParser


def qex(expression: str) -> Signal:
    """Parse and return signal from string expression.

    This is the primary entry point for qex expressions.

    Example:
        signal = qex("rank(-returns(20) / volatility(60))")
    """
    from .kinds import check                          # labels only where labels belong, before anything runs
    parser = QexParser()
    return check(parser.parse(expression))


# Backwards compatibility alias
alpha = qex


def compute_weights(expression: str,
                    data: Dict[str, pd.DataFrame],
                    **kwargs) -> pd.DataFrame:
    """
    Parse expression and return portfolio weights.

    This is the main entry point for backtesting integration.
    """
    signal = qex(expression)
    return signal.to_weights(data, **kwargs)
