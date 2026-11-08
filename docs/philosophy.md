# The Art of Quant Expressions

## What is a quant expression?

A quant expression is a formula that transforms market data into a trading signal. It takes prices, volumes, and other observables as inputs and outputs a number for each stock on each day. That number represents conviction: how much you want to be long or short.

```qex
rank(-returns(20) / volatility(60))
```

This expression says: "Rank stocks by their risk-adjusted momentum over the past month, and bet against the winners." It's a mean-reversion signal. The entire logic fits in one line.

Quant expressions are declarative. You describe *what* you want to compute, not *how* to compute it. The engine handles the mechanics—rolling windows, cross-sectional ranks, missing data, alignment. You focus on the idea.

## Why write signals this way?

### Ideas become testable in seconds

The gap between "I wonder if..." and "let's see the backtest" should be measured in keystrokes, not hours. When you have a hunch—maybe stocks with high short-term momentum but low long-term momentum are about to reverse—you should be able to write it down and test it immediately:

```qex
rank(returns(5)) - rank(returns(60))
```

Done. Run the backtest. See if the idea has merit. Move on or iterate.

### Signals are readable

Six months from now, you'll look at your code. With a quant expression, the signal *is* the documentation:

```qex
let mom = returns(20), vol = volatility(60) in
rank(mom / vol) * sign(mom)
```

Compare this to 200 lines of pandas operations spread across three files. The expression captures intent. The pandas captures implementation details that obscure the idea.

### Composition is natural

Good signals often combine multiple effects. Maybe you want momentum, but only for liquid stocks, and you want to neutralize sector exposure:

```qex
group_demean(
    where(adv(20) > 1000000, rank(returns(20)), None),
    'sector'
)
```

Each operation snaps together like lego. Cross-sectional ranks, validity masks, group neutralization—they compose without friction.

### No lookahead by construction

A common source of backtest errors is accidentally using future data. qex rules this out by design: every operation looks only backward in time, and `delay()` refuses a negative lag. There's no way to write an expression that peeks at tomorrow's price. (One exception lives in the data rather than
the language: an adjusted price *level* carries later corporate actions, and `qex.adjust.check` flags it; see
[Adjusted Data and Look-Ahead](language/adjusted-data.md).) `Backtest` pairs each day's weights with the *next*
day's return, so you never lag a signal to keep it honest.

You will still see `delay` in examples, for other reasons:

- **Measurement.** A factor's daily return regresses today's returns on characteristics known at yesterday's close
  (`regression_beta(returns(1), delay(z, 1))`): what a portfolio formed before the day earned during it.
- **Conventions.** Momentum skips the last month, `delay(returns(252), 20)`.
- **Hand-built P&L.** Inside an expression, `delay(w, 1) * returns(1)` is a position's P&L, the pairing `Backtest`
  does on its own.

One thing the language cannot decide for you is *when* you trade on a value. A signal computed from today's close can't be traded at that same close in practice; build the book with `delay=1` (see [Backtesting](python-examples/backtesting.md#book-settings-finding-alphas)) so that today's weights use only yesterday's data.

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

```qex
returns(5)
```

Then add transformations. Maybe you want ranks instead of raw values:

```qex
rank(returns(5))
```

Maybe you want to compare short-term to long-term:

```qex
rank(returns(5)) - rank(returns(60))
```

Maybe you want to scale by volatility:

```qex
(rank(returns(5)) - rank(returns(60))) / volatility(20)
```

Each step should make sense on its own. If you can't explain why you're adding something, don't add it.

### Think about edge cases

What happens when data is missing? What happens for illiquid stocks? What happens at market opens, closes, around earnings?

Robust signals handle edge cases explicitly:

```qex
let raw = returns(5) / volatility(20) in
where(adv(20) > 500000, rank(raw) - 0.5, None)
```

This says: rank risk-adjusted momentum, and hold no position in illiquid stocks. `None` is no value, so a book leaves those stocks out; a `0` would not, since a book demeans its signal and a `0` below or above the average is a position. (`None` before the rank, `rank(where(..., raw, None))`, works too; a `0` there would give every illiquid stock the same rank and still a weight.)

### Consider what you're betting against

Every long position implies a short position somewhere. When you rank stocks, you're saying the top ranks are better than the bottom ranks. Is that true? Why?

Mean reversion signals bet that extreme moves reverse:

```qex
rank(-returns(5))
```

You're long recent losers, short recent winners. That's a bet that short-term price movements are noise, not signal.

Momentum signals bet the opposite:

```qex
rank(returns(60))
```

You're long recent winners, short recent losers. That's a bet that trends persist.

Know which side of the trade you're on.

## Classic patterns

Some patterns appear again and again. They're worth knowing:

### Risk-adjusted momentum

Raw momentum is noisy. Dividing by volatility gives a Sharpe-ratio-like measure:

```qex
rank(returns(20) / volatility(60))
```

### Mean reversion with a filter

Don't fade every move. Fade moves that are large relative to history:

```qex
let z = (close() - ts_mean(close(), 20)) / ts_std(close(), 20)
in where(abs(z) > 2, rank(-z), None)
```

### Momentum with quality

Combine momentum with some measure of quality—low volatility, high liquidity, strong fundamentals:

```qex
rank(returns(60)) * rank(-volatility(60)) * rank(adv(20))
```

### Cross-sectional vs. time-series

Cross-sectional signals compare stocks to each other on the same day:

```qex
rank(returns(20))
```

Time-series signals compare a stock to its own history:

```qex
ts_rank(returns(20), 252)
```

Many successful strategies combine both.

## The feedback loop

Writing quant expressions is iterative. You write, test, observe, refine.

The expression is a hypothesis. The backtest is the experiment. The results tell you whether to iterate or move on.

Keep expressions simple at first. Complexity should be earned through evidence, not assumed through intuition. If a simple signal doesn't work, a complex version probably won't either.

And remember: the expression is a tool for thinking. Even if a signal doesn't trade well, the act of formalizing your intuition teaches you something about the market.

That's the art.
