# The Art of Quant Expressions

## What is a quant expression?

A quant expression is a formula that transforms market data into a trading signal. It takes prices, volumes, and other observables as inputs and outputs a number for each stock on each day. That number represents conviction: how much you want to be long or short.

```
rank(-returns(20) / volatility(60))
```

This expression says: "Rank stocks by their risk-adjusted momentum over the past month, and bet against the winners." It's a mean-reversion signal. The entire logic fits in one line.

Quant expressions are declarative. You describe *what* you want to compute, not *how* to compute it. The engine handles the mechanics—rolling windows, cross-sectional ranks, missing data, alignment. You focus on the idea.

## Why write signals this way?

### Ideas become testable in seconds

The gap between "I wonder if..." and "let's see the backtest" should be measured in keystrokes, not hours. When you have a hunch—maybe stocks with high short-term momentum but low long-term momentum are about to reverse—you should be able to write it down and test it immediately:

```
rank(returns(5)) - rank(returns(60))
```

Done. Run the backtest. See if the idea has merit. Move on or iterate.

### Signals are readable

Six months from now, you'll look at your code. With a quant expression, the signal *is* the documentation:

```
let mom = returns(20), vol = volatility(60) in
rank(mom / vol) * sign(mom)
```

Compare this to 200 lines of pandas operations spread across three files. The expression captures intent. The pandas captures implementation details that obscure the idea.

### Composition is natural

Good signals often combine multiple effects. Maybe you want momentum, but only for liquid stocks, and you want to neutralize sector exposure:

```
group_demean(
    rank(returns(20)) * (adv(20) > 1000000),
    'sector'
)
```

Each operation snaps together like lego. Cross-sectional ranks, validity masks, group neutralization—they compose without friction.

### Mistakes become obvious

When signals are explicit, errors are visible. You can read the expression and ask: Does this make sense? Am I looking back in time? Am I accidentally peeking at future data? The expression is small enough to audit.

## Who writes quant expressions?

**Researchers** exploring new signals. You have an idea from a paper, a market observation, or pure intuition. You want to test it quickly, see if it generalizes, understand its behavior across regimes.

**Portfolio managers** defining their strategy. The signals you trade should be precise and auditable. A quant expression is a specification that can be reviewed, version-controlled, and stress-tested.

**Risk managers** monitoring exposures. Factor exposures are just signals. Beta, momentum, value, volatility—each can be expressed, measured, and tracked.

**Developers** building trading systems. When signals are data, not code, they can be stored in databases, edited through UIs, deployed without code changes. The system becomes more flexible.

## From intuition to expression

Most trading ideas start as fuzzy intuitions. The skill is translating them into precise math. Here's the process:

### Start with the story

Every signal encodes a belief about market behavior. Make the belief explicit:

- "Stocks that have fallen sharply tend to bounce back"
- "High momentum stocks continue to outperform"
- "When a stock's price diverges from its peers, it reverts"
- "Earnings surprises predict future returns"

The story is your north star. If the expression doesn't match the story, something is wrong.

### Identify the raw ingredients

What data do you need? Usually it's some combination of:

- **Price data**: returns, volatility, highs, lows
- **Volume data**: trading volume, dollar volume
- **Fundamental data**: earnings, book value, sales
- **Alternative data**: sentiment, events, signals from other models

Write down the ingredients before you write the expression.

### Build incrementally

Start with the core effect:

```
returns(5)
```

Then add transformations. Maybe you want ranks instead of raw values:

```
rank(returns(5))
```

Maybe you want to compare short-term to long-term:

```
rank(returns(5)) - rank(returns(60))
```

Maybe you want to scale by volatility:

```
(rank(returns(5)) - rank(returns(60))) / volatility(20)
```

Each step should make sense on its own. If you can't explain why you're adding something, don't add it.

### Think about edge cases

What happens when data is missing? What happens for illiquid stocks? What happens at market opens, closes, around earnings?

Robust signals handle edge cases explicitly:

```
let raw = returns(5) / volatility(20) in
rank(where(adv(20) > 500000, raw, 0))
```

This says: compute risk-adjusted momentum, but zero it out for illiquid stocks.

### Consider what you're betting against

Every long position implies a short position somewhere. When you rank stocks, you're saying the top ranks are better than the bottom ranks. Is that true? Why?

Mean reversion signals bet that extreme moves reverse:

```
rank(-returns(5))
```

You're long recent losers, short recent winners. That's a bet that short-term price movements are noise, not signal.

Momentum signals bet the opposite:

```
rank(returns(60))
```

You're long recent winners, short recent losers. That's a bet that trends persist.

Know which side of the trade you're on.

## Classic patterns

Some patterns appear again and again. They're worth knowing:

### Risk-adjusted momentum

Raw momentum is noisy. Dividing by volatility gives a Sharpe-ratio-like measure:

```
rank(returns(20) / volatility(60))
```

### Mean reversion with a filter

Don't fade every move. Fade moves that are large relative to history:

```
let z = (close() - ts_mean(close(), 20)) / ts_std(close(), 20) in
rank(-z) * (abs(z) > 2)
```

### Momentum with quality

Combine momentum with some measure of quality—low volatility, high liquidity, strong fundamentals:

```
rank(returns(60)) * rank(-volatility(60)) * rank(adv(20))
```

### Cross-sectional vs. time-series

Cross-sectional signals compare stocks to each other on the same day:

```
rank(returns(20))
```

Time-series signals compare a stock to its own history:

```
ts_rank(returns(20), 252)
```

Many successful strategies combine both.

## The feedback loop

Writing quant expressions is iterative. You write, test, observe, refine.

The expression is a hypothesis. The backtest is the experiment. The results tell you whether to iterate or move on.

Keep expressions simple at first. Complexity should be earned through evidence, not assumed through intuition. If a simple signal doesn't work, a complex version probably won't either.

And remember: the expression is a tool for thinking. Even if a signal doesn't trade well, the act of formalizing your intuition teaches you something about the market.

That's the art.
