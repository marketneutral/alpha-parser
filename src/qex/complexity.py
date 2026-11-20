"""How much an expression says: the operators written, and the parameters its estimators search for.

A score that charges for complexity should charge for what the author chose, not for how qex builds the graph.
SignalGraph's node count does the second: it counts internal nodes nobody writes (an estimator's fit, ts_match's
selection), so it moved when qex's internals did (0.20.0 put a fit node above ts_beta and took the hidden slope from
under ts_resid) while every number stayed the same. complexity() counts instead:

    operators   the distinct operations written (a shared subexpression once, as in the graph), not the internal
                nodes: ts_beta(y, x, 252) is one, ts_resid(y, x, 252) is one, pca_factor(r, 252, 2) is one,
                and estimate('ols', 'beta1', ...) is the same one as ts_beta
    dof         the degrees of freedom the expression's estimators learn from the data by searching for what works:
                the stump's threshold and the side each leaf takes (3). Descriptive statistics, a beta, a residual,
                a principal component, are not searched for and learn none. Each distinct fit counts once, however
                many of its outputs are read.

With data, complexity(signal, data) charges the dof the estimators actually used, averaged over their refits, where
an estimator measures it (realized_dof): a tree charges the splits it made and how much each leaf commits (|2w - 1|:
a leaf at the blend commits nothing), so a regularized tree that splits only when it must, or whose leaves hold back,
is charged less. An estimator that does not measure it is charged its nominal dof, as without data.

A model that learns more can fit more noise, so a score can charge for each dof as well as each operator. The stump
fits on the past only, so its own choices do not inflate a backtest the way an in-sample fit would; what a dof charge
prices is flexibility: a more flexible model gives an author's search more ways to find noise, and holds up less well.
"""
from dataclasses import dataclass

from .signal import Signal


@dataclass(frozen=True)
class Complexity:
    operators: int
    dof: float

    def charge(self, per_operator: float, per_dof: float) -> float:
        """The charge at these rates: per_operator * operators + per_dof * dof."""
        return per_operator * self.operators + per_dof * self.dof


def complexity(signal: Signal, data=None) -> Complexity:
    """The operators an expression writes and the dof its estimators learn (see the module's notes). With data, the
    dof each fit realized on it where its estimator measures that; evaluated in the caller's compute_context (and
    universe), so a fit the score has computed is not computed again."""
    from .estimate import EstimateFit
    from .missing import children
    seen, stack = {}, [signal]
    while stack:
        s = stack.pop()
        k = s._cache_key()
        if k in seen:
            continue
        seen[k] = s
        stack.extend(children(s))
    operators = sum(1 for s in seen.values() if getattr(type(s), "value_is_frame", True))
    dof = 0
    for s in seen.values():
        if not isinstance(s, EstimateFit):
            continue
        realized = s.evaluate(data).realized_dof if data is not None else None
        dof += s.estimator.dof_of(s) if realized is None else realized
    return Complexity(operators, dof)
