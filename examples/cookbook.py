"""
Alpha Parser Cookbook
=====================

A comprehensive guide to building quantitative trading signals with alpha-parser.
Each section contains heavily commented examples that you can run directly.

This cookbook is organized by strategy type:
    1. Data Setup & Basics
    2. Momentum Strategies
    3. Mean Reversion Strategies
    4. Pairs Trading
    5. Factor-Neutral Signals
    6. Event-Driven Signals (PEAD)
    7. Risk Management Patterns
    8. Common Gotchas & Best Practices

Run this file directly to see all examples in action:
    PYTHONPATH=src python examples/cookbook.py

Or import specific sections in your own code.
"""

import numpy as np
import pandas as pd

from alpha_parser import alpha, compute_context, compute_weights


# =============================================================================
# SECTION 1: DATA SETUP & BASICS
# =============================================================================

def create_sample_data(n_days=252, n_tickers=10, seed=42):
    """
    Create sample market data for examples.

    Alpha-parser expects data as a dictionary of DataFrames:
        - Keys: 'close', 'open', 'high', 'low', 'volume', etc.
        - Values: DataFrames with DatetimeIndex (rows) and ticker columns

    All DataFrames must have the same index and columns.
    """
    np.random.seed(seed)
    dates = pd.date_range('2020-01-01', periods=n_days, freq='B')  # Business days
    tickers = [f'STOCK_{i}' for i in range(n_tickers)]

    # Generate correlated returns (more realistic than independent)
    # Market factor affects all stocks, idiosyncratic component is unique
    market_returns = np.random.randn(n_days) * 0.01

    close = pd.DataFrame(index=dates, columns=tickers)
    for i, ticker in enumerate(tickers):
        # Each stock has different beta to market (0.5 to 1.5)
        beta = 0.5 + i / n_tickers
        idio = np.random.randn(n_days) * 0.02
        returns = beta * market_returns + idio
        close[ticker] = 100 * np.exp(np.cumsum(returns))

    # Volume: lognormal with some correlation to abs(returns)
    volume = pd.DataFrame(
        np.random.lognormal(15, 0.5, (n_days, n_tickers)),
        index=dates,
        columns=tickers
    )

    # OHLC data (simplified - close is the reference)
    high = close * (1 + np.abs(np.random.randn(n_days, n_tickers) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n_days, n_tickers) * 0.01))
    open_ = close.shift(1).fillna(close)  # Previous close as open

    return {
        'close': close,
        'open': open_,
        'high': high,
        'low': low,
        'volume': volume,
    }


def basics_example():
    """
    BASICS: Parsing and evaluating signals.

    The alpha() function parses a string expression into a Signal object.
    Signal objects have two main methods:
        - evaluate(data) -> DataFrame of raw signal values
        - to_weights(data) -> DataFrame of portfolio weights
    """
    print("\n" + "="*70)
    print("SECTION 1: BASICS")
    print("="*70)

    data = create_sample_data(n_days=60, n_tickers=5)

    # ---------------------------------------------------------------------
    # Basic signal: 20-day momentum (returns)
    # ---------------------------------------------------------------------
    # returns(N) computes: (close / close.shift(N)) - 1
    signal = alpha("returns(20)")
    result = signal.evaluate(data)

    print("\n1.1 Basic momentum signal: returns(20)")
    print(f"    Shape: {result.shape}")
    print(f"    First valid row: {result.first_valid_index()}")
    # Note: First 20 rows are NaN because we need 20 days of history

    # ---------------------------------------------------------------------
    # Combining operations: Risk-adjusted momentum
    # ---------------------------------------------------------------------
    # Divide momentum by volatility to get Sharpe-like ratio
    # volatility(N) computes: rolling std of returns over N periods
    signal = alpha("returns(20) / volatility(20)")
    result = signal.evaluate(data)

    print("\n1.2 Risk-adjusted momentum: returns(20) / volatility(20)")
    print(f"    Sample values:\n{result.iloc[-1]}")

    # ---------------------------------------------------------------------
    # Cross-sectional ranking
    # ---------------------------------------------------------------------
    # rank() converts values to percentiles [0, 1] WITHIN each row
    # Higher value -> higher rank (ascending order)
    signal = alpha("rank(returns(20))")
    result = signal.evaluate(data)

    print("\n1.3 Cross-sectional rank: rank(returns(20))")
    print(f"    Row sums (should vary): {result.iloc[-5:].sum(axis=1).values}")
    # Note: rank() returns percentiles, not integers

    # ---------------------------------------------------------------------
    # Converting to portfolio weights
    # ---------------------------------------------------------------------
    # to_weights() normalizes signal to sum to desired exposure
    signal = alpha("rank(returns(20)) - 0.5")  # Center around 0
    weights = signal.to_weights(data, normalize=True)

    print("\n1.4 Portfolio weights from signal")
    print(f"    Sum of abs weights: {weights.iloc[-1].abs().sum():.4f}")
    print(f"    Net exposure: {weights.iloc[-1].sum():.4f}")

    # Long-only version
    weights_long = signal.to_weights(data, normalize=True, long_only=True)
    print(f"    Long-only min weight: {weights_long.iloc[-1].min():.4f}")


# =============================================================================
# SECTION 2: MOMENTUM STRATEGIES
# =============================================================================

def momentum_examples():
    """
    MOMENTUM STRATEGIES

    Momentum strategies bet that recent winners continue to outperform
    and recent losers continue to underperform. Key considerations:

    1. Lookback period: Typically 1-12 months (skip most recent month)
    2. Risk adjustment: Divide by volatility for better risk/return
    3. Cross-sectional: Rank stocks relative to each other
    """
    print("\n" + "="*70)
    print("SECTION 2: MOMENTUM STRATEGIES")
    print("="*70)

    data = create_sample_data(n_days=300, n_tickers=20)

    # ---------------------------------------------------------------------
    # Classic 12-1 Momentum
    # ---------------------------------------------------------------------
    # 12-month return, skipping most recent month (avoid reversal)
    # delay(x, N) shifts the signal back by N periods
    signal = alpha("delay(returns(252), 21)")  # ~12 months, skip 1 month

    print("\n2.1 Classic 12-1 Momentum")
    print("    Formula: delay(returns(252), 21)")
    print("    Rationale: Use 12-month return, but skip the most recent month")
    print("               to avoid short-term reversal effects.")

    # ---------------------------------------------------------------------
    # Risk-Adjusted Momentum (Sharpe Momentum)
    # ---------------------------------------------------------------------
    # Stocks with high returns AND low volatility are better
    signal = alpha("returns(60) / volatility(60)")

    print("\n2.2 Risk-Adjusted Momentum (Sharpe-like)")
    print("    Formula: returns(60) / volatility(60)")
    print("    Rationale: Prefer stocks with high return per unit risk.")

    # ---------------------------------------------------------------------
    # Ranked Momentum with Winsorization
    # ---------------------------------------------------------------------
    # winsorize() caps extreme values to reduce outlier impact
    # winsorize(x, 0.05) caps at 5th and 95th percentiles
    signal = alpha("rank(winsorize(returns(60), 0.05))")

    print("\n2.3 Winsorized Ranked Momentum")
    print("    Formula: rank(winsorize(returns(60), 0.05))")
    print("    Rationale: Cap extreme returns before ranking to reduce")
    print("               impact of outliers (M&A, earnings surprises).")

    # ---------------------------------------------------------------------
    # Acceleration: Change in Momentum
    # ---------------------------------------------------------------------
    # delta(x, N) computes x - delay(x, N)
    # Positive acceleration = momentum is increasing
    signal = alpha("delta(returns(20), 20)")

    print("\n2.4 Momentum Acceleration")
    print("    Formula: delta(returns(20), 20)")
    print("    Rationale: Bet on stocks where momentum is INCREASING,")
    print("               not just high. Captures trend acceleration.")

    # ---------------------------------------------------------------------
    # Multi-Horizon Momentum (Composite)
    # ---------------------------------------------------------------------
    # Combine multiple lookback periods for robustness
    signal = alpha(
        "0.25 * rank(returns(21)) + "   # 1 month
        "0.25 * rank(returns(63)) + "   # 3 months
        "0.25 * rank(returns(126)) + "  # 6 months
        "0.25 * rank(returns(252))"     # 12 months
    )

    print("\n2.5 Multi-Horizon Composite Momentum")
    print("    Formula: Equal-weight average of 1/3/6/12 month momentum ranks")
    print("    Rationale: Diversify across time horizons. Different horizons")
    print("               capture different market dynamics.")

    # Evaluate to show it works
    result = signal.evaluate(data)
    print(f"    Sample output (last row): {result.iloc[-1].values[:5].round(3)}")


# =============================================================================
# SECTION 3: MEAN REVERSION STRATEGIES
# =============================================================================

def mean_reversion_examples():
    """
    MEAN REVERSION STRATEGIES

    Mean reversion strategies bet that prices/signals will revert to their
    historical average. Key considerations:

    1. What "mean" to use: Simple average, EWMA, or moving average
    2. Normalization: Z-score tells you how many std devs from mean
    3. Time horizon: Short-term (days) vs medium-term (weeks/months)
    """
    print("\n" + "="*70)
    print("SECTION 3: MEAN REVERSION STRATEGIES")
    print("="*70)

    data = create_sample_data(n_days=200, n_tickers=15)

    # ---------------------------------------------------------------------
    # Simple Mean Reversion (Contrarian)
    # ---------------------------------------------------------------------
    # Negative of recent returns: buy losers, sell winners
    signal = alpha("-returns(5)")

    print("\n3.1 Simple Contrarian")
    print("    Formula: -returns(5)")
    print("    Rationale: Short-term reversal. Recent losers bounce,")
    print("               recent winners pull back.")

    # ---------------------------------------------------------------------
    # Bollinger Band Mean Reversion
    # ---------------------------------------------------------------------
    # Z-score of price relative to its moving average
    # (price - MA) / std -> how many std devs from the mean
    signal = alpha(
        "rank(-(close() - ts_mean(close(), 20)) / ts_std(close(), 20))"
    )

    print("\n3.2 Bollinger Band Signal")
    print("    Formula: rank(-(close - MA20) / std20)")
    print("    Rationale: Buy stocks that are oversold (below lower band),")
    print("               sell stocks that are overbought (above upper band).")
    print("    Note: Negative sign because LOW z-score = oversold = BUY")

    # ---------------------------------------------------------------------
    # RSI-style Mean Reversion
    # ---------------------------------------------------------------------
    # Compare up moves vs down moves
    # ts_sum of positive returns / ts_sum of all abs returns
    signal = alpha(
        "rank(-ts_mean(max(returns(1), 0), 14) / "
        "(ts_mean(max(returns(1), 0), 14) + ts_mean(max(-returns(1), 0), 14)))"
    )

    print("\n3.3 RSI-style Signal")
    print("    Formula: Ratio of up-moves to total moves (14-day)")
    print("    Rationale: Similar to RSI. High ratio = overbought = SELL.")
    print("               We negate and rank for a contrarian signal.")

    # ---------------------------------------------------------------------
    # EWMA Mean Reversion (More Responsive)
    # ---------------------------------------------------------------------
    # EWMA adapts faster to regime changes than simple MA
    # halflife=10 means 50% weight on last 10 observations
    signal = alpha(
        "rank(-(close() - ewma(close(), 10)) / sqrt(ewma_var(returns(1), 10)))"
    )

    print("\n3.4 EWMA Mean Reversion")
    print("    Formula: -(close - EWMA) / sqrt(EWMA_variance)")
    print("    Rationale: EWMA adapts faster than simple MA. Better for")
    print("               trending markets that shift regimes.")

    # ---------------------------------------------------------------------
    # Cross-Sectional Mean Reversion (Industry-Relative)
    # ---------------------------------------------------------------------
    # demean() subtracts the cross-sectional mean
    # zscore() also divides by cross-sectional std
    signal = alpha("-zscore(returns(5))")

    print("\n3.5 Cross-Sectional Mean Reversion")
    print("    Formula: -zscore(returns(5))")
    print("    Rationale: Z-score normalizes across stocks. Buy the")
    print("               relatively worst performers, sell the best.")
    print("    Note: This is CROSS-SECTIONAL (across stocks), not time-series.")

    result = signal.evaluate(data)
    print(f"    Sample z-scores: {result.iloc[-1].values[:5].round(3)}")


# =============================================================================
# SECTION 4: PAIRS TRADING
# =============================================================================

def pairs_trading_examples():
    """
    PAIRS TRADING

    Pairs trading exploits mean reversion between related securities.
    Key concepts:

    1. Spread: Difference (or ratio) between two related stocks
    2. Hedge ratio: How many shares of Y to hold per share of X
    3. Z-score: Normalized spread for entry/exit signals

    Alpha-parser supports pairs via GROUP operations:
        - group_demean: Subtract the pair's average
        - group_std: Compute volatility within each pair
    """
    print("\n" + "="*70)
    print("SECTION 4: PAIRS TRADING")
    print("="*70)

    # Create data with natural pairs
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='B')

    # Create 3 pairs with cointegrated prices
    # Pair 1: AAPL/MSFT (tech), Pair 2: XOM/CVX (energy), Pair 3: KO/PEP (staples)
    tickers = ['AAPL', 'MSFT', 'XOM', 'CVX', 'KO', 'PEP']

    close = pd.DataFrame(index=dates, columns=tickers)

    # Each pair shares a common factor + idiosyncratic noise
    for pair_stocks, base_price in [(['AAPL', 'MSFT'], 150),
                                      (['XOM', 'CVX'], 80),
                                      (['KO', 'PEP'], 55)]:
        common = np.cumsum(np.random.randn(252) * 0.01)
        for stock in pair_stocks:
            idio = np.cumsum(np.random.randn(252) * 0.005)
            close[stock] = base_price * np.exp(common + idio)

    volume = pd.DataFrame(
        np.random.lognormal(15, 0.3, (252, 6)),
        index=dates,
        columns=tickers
    )

    # CRITICAL: Define pair memberships
    # Each stock maps to its pair identifier
    pair_groups = pd.DataFrame(
        [['tech', 'tech', 'energy', 'energy', 'staples', 'staples']] * 252,
        index=dates,
        columns=tickers
    )

    data = {
        'close': close,
        'volume': volume,
        'pair': pair_groups,  # Group data for pairs
    }

    # ---------------------------------------------------------------------
    # Basic Pairs Spread
    # ---------------------------------------------------------------------
    # group_demean subtracts the pair's mean return
    # Result: +ve for outperformer, -ve for underperformer within pair
    signal = alpha("group_demean(returns(1), 'pair')")

    print("\n4.1 Basic Pair Spread (Returns)")
    print("    Formula: group_demean(returns(1), 'pair')")
    print("    Rationale: Within each pair, one stock gets +ve signal,")
    print("               the other gets -ve. They sum to zero per pair.")

    result = signal.evaluate(data)
    # Check that pairs sum to zero
    last_row = result.iloc[-1]
    tech_sum = last_row['AAPL'] + last_row['MSFT']
    print(f"    AAPL + MSFT spread sums to: {tech_sum:.6f} (should be ~0)")

    # ---------------------------------------------------------------------
    # Z-Score Normalized Spread
    # ---------------------------------------------------------------------
    # Normalize by rolling std within the pair
    # This gives comparable magnitude across different pairs
    signal = alpha(
        "group_demean(returns(5), 'pair') / group_std(returns(5), 'pair', 60)"
    )

    print("\n4.2 Z-Score Normalized Spread")
    print("    Formula: group_demean(returns, 'pair') / group_std(returns, 'pair', 60)")
    print("    Rationale: Normalize spread by its historical volatility.")
    print("               Z-score > 2 might indicate entry opportunity.")

    # ---------------------------------------------------------------------
    # Mean Reversion Signal
    # ---------------------------------------------------------------------
    # Trade AGAINST the spread: buy the laggard, sell the leader
    signal = alpha("-group_demean(returns(10), 'pair')")

    print("\n4.3 Mean Reversion Pairs Signal")
    print("    Formula: -group_demean(returns(10), 'pair')")
    print("    Rationale: Negative sign means we BUY the underperformer")
    print("               and SELL the outperformer, betting on reversion.")

    # ---------------------------------------------------------------------
    # Pairs with Dynamic Hedge Ratio (ADVANCED)
    # ---------------------------------------------------------------------
    # The simple spread assumes hedge ratio = 1.0
    # For better hedging, use rolling beta between the pair
    #
    # ts_beta(y, x, window) computes cov(y,x) / var(x)
    # This tells you: for every $1 move in x, y moves by $beta
    #
    # For pairs trading: hedge_ratio = beta
    # Hedged position: long Y, short (beta * X)

    print("\n4.4 Dynamic Hedge Ratio with Beta")
    print("    Formula: ts_beta(returns_Y, returns_X, 60)")
    print("    Rationale: Optimal hedge ratio from rolling regression.")
    print("               Hedged spread = Y - beta * X has lower variance.")

    # For EWMA beta (more responsive to regime changes):
    print("\n4.5 EWMA Hedge Ratio")
    print("    Formula: ts_beta_ewma(returns_Y, returns_X, 20)")
    print("    Rationale: EWMA weights recent data more heavily.")
    print("               Adapts faster when correlation structure changes.")

    # ---------------------------------------------------------------------
    # Conditional Pairs Trading
    # ---------------------------------------------------------------------
    # Only trade when spread is extreme (z-score > threshold)
    signal = alpha(
        "where("
        "  abs(group_demean(returns(10), 'pair') / group_std(returns(10), 'pair', 60)) > 1.5,"
        "  -group_demean(returns(5), 'pair'),"
        "  0"
        ")"
    )

    print("\n4.6 Conditional Pairs (Z-Score Threshold)")
    print("    Formula: where(|z-score| > 1.5, mean_reversion_signal, 0)")
    print("    Rationale: Only trade when spread is significantly stretched.")
    print("               Reduces turnover and false signals.")

    result = signal.evaluate(data)
    non_zero = (result.iloc[-20:] != 0).sum().sum()
    total = result.iloc[-20:].count().sum()
    print(f"    Active positions in last 20 days: {non_zero}/{total}")


# =============================================================================
# SECTION 5: FACTOR-NEUTRAL SIGNALS
# =============================================================================

def factor_neutral_examples():
    """
    FACTOR-NEUTRAL SIGNALS

    Factor-neutral signals remove exposure to common risk factors.
    This isolates the alpha from systematic risks.

    Common approaches:
    1. Industry/sector neutralization (group_demean by sector)
    2. Market beta neutralization (subtract beta * market)
    3. Style factor neutralization (size, value, momentum)
    """
    print("\n" + "="*70)
    print("SECTION 5: FACTOR-NEUTRAL SIGNALS")
    print("="*70)

    # Create data with sector groups
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='B')
    tickers = ['AAPL', 'MSFT', 'GOOG',  # Tech
               'XOM', 'CVX', 'COP',      # Energy
               'JPM', 'BAC', 'WFC',      # Finance
               'JNJ', 'PFE', 'MRK']      # Healthcare

    # Sector assignments
    sectors = ['tech'] * 3 + ['energy'] * 3 + ['finance'] * 3 + ['healthcare'] * 3
    sector_df = pd.DataFrame(
        [sectors] * len(dates),
        index=dates,
        columns=tickers
    )

    # Generate returns with sector factors
    close = pd.DataFrame(index=dates, columns=tickers)
    sector_factors = {
        'tech': np.cumsum(np.random.randn(200) * 0.015),
        'energy': np.cumsum(np.random.randn(200) * 0.012),
        'finance': np.cumsum(np.random.randn(200) * 0.010),
        'healthcare': np.cumsum(np.random.randn(200) * 0.008),
    }

    for i, ticker in enumerate(tickers):
        sector = sectors[i]
        idio = np.cumsum(np.random.randn(200) * 0.01)
        close[ticker] = 100 * np.exp(sector_factors[sector] + idio)

    volume = pd.DataFrame(
        np.random.lognormal(15, 0.3, (200, 12)),
        index=dates,
        columns=tickers
    )

    data = {
        'close': close,
        'volume': volume,
        'sector': sector_df,
    }

    # ---------------------------------------------------------------------
    # Industry-Neutral Momentum
    # ---------------------------------------------------------------------
    # group_demean by sector removes sector exposure
    # Result: momentum relative to sector peers
    signal = alpha("group_demean(returns(20), 'sector')")

    print("\n5.1 Sector-Neutral Momentum")
    print("    Formula: group_demean(returns(20), 'sector')")
    print("    Rationale: Removes sector effect. A tech stock is compared")
    print("               only to other tech stocks, not to energy stocks.")

    result = signal.evaluate(data)
    # Check sector sums are ~0
    last_row = result.iloc[-1]
    tech_sum = last_row[['AAPL', 'MSFT', 'GOOG']].sum()
    print(f"    Tech sector signal sum: {tech_sum:.6f} (should be ~0)")

    # ---------------------------------------------------------------------
    # Industry-Neutral with Ranking
    # ---------------------------------------------------------------------
    # Rank within sector for more stable signals
    signal = alpha("group_rank(returns(20), 'sector')")

    print("\n5.2 Sector-Neutral Ranked Momentum")
    print("    Formula: group_rank(returns(20), 'sector')")
    print("    Rationale: Rank within each sector. Top stock in each")
    print("               sector gets high rank regardless of sector returns.")

    # ---------------------------------------------------------------------
    # Double-Sorted Signal
    # ---------------------------------------------------------------------
    # First neutralize sector, then apply cross-sectional rank
    signal = alpha("rank(group_demean(returns(20), 'sector'))")

    print("\n5.3 Double-Sorted: Sector-Neutral then Ranked")
    print("    Formula: rank(group_demean(returns(20), 'sector'))")
    print("    Rationale: First remove sector effect, then rank across")
    print("               all stocks. Combines within and across sector.")

    # ---------------------------------------------------------------------
    # Beta-Neutral Signal (Market Hedge)
    # ---------------------------------------------------------------------
    # To make a signal beta-neutral, subtract beta * market_return
    #
    # Conceptually: residual = stock_return - beta * market_return
    # This removes market exposure from your signal

    print("\n5.4 Beta-Neutral Signal (Conceptual)")
    print("    Formula: returns(1) - ts_beta(returns(1), market_return, 60) * market_return")
    print("    Rationale: Subtract the portion of returns explained by market.")
    print("               What remains is idiosyncratic (alpha) component.")
    print("    Note: Requires market returns in your data (e.g., SPY).")

    # ---------------------------------------------------------------------
    # Multi-Factor Neutral (Advanced)
    # ---------------------------------------------------------------------
    print("\n5.5 Multi-Factor Neutral Signal (Advanced)")
    print("    Approach: Use FactorRiskModel to compute factor exposures,")
    print("              then construct signal from residuals.")
    print("    See: alpha_parser.FactorRiskModel for implementation.")


# =============================================================================
# SECTION 6: EVENT-DRIVEN SIGNALS (PEAD)
# =============================================================================

def event_driven_examples():
    """
    EVENT-DRIVEN SIGNALS

    Event-driven signals react to discrete events like earnings announcements.
    Key challenge: Data is SPARSE (most days have no event).

    Special operations for sparse data:
    - is_valid(x): Returns 1 where x is not NaN, 0 otherwise
    - fill_forward(x, N): Forward-fill NaN for up to N periods
    - ts_mean_events(x, N): Mean over past N non-NaN values (not N days!)
    - ts_count_events(x, N): Count non-NaN values in past N days
    """
    print("\n" + "="*70)
    print("SECTION 6: EVENT-DRIVEN SIGNALS (PEAD)")
    print("="*70)

    # Create data with sparse earnings surprises
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='B')
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']

    close = pd.DataFrame(
        100 * np.exp(np.cumsum(np.random.randn(252, 5) * 0.01, axis=0)),
        index=dates,
        columns=tickers
    )

    volume = pd.DataFrame(
        np.random.lognormal(15, 0.3, (252, 5)),
        index=dates,
        columns=tickers
    )

    # Sparse earnings surprises (only on earnings dates)
    # Most values are NaN, a few are actual surprises
    earnings_surprise = pd.DataFrame(
        np.nan,  # Default to NaN
        index=dates,
        columns=tickers
    )

    # Each stock reports ~4 times per year (quarterly)
    for ticker in tickers:
        # Random earnings dates
        earnings_dates = np.random.choice(252, size=4, replace=False)
        for ed in earnings_dates:
            # Surprise between -0.1 and +0.1
            earnings_surprise.iloc[ed][ticker] = np.random.uniform(-0.1, 0.1)

    data = {
        'close': close,
        'volume': volume,
        'earnings_surprise': earnings_surprise,  # SPARSE!
    }

    # ---------------------------------------------------------------------
    # Detecting Events
    # ---------------------------------------------------------------------
    # is_valid() returns 1 where not NaN, 0 otherwise
    signal = alpha("is_valid(field('earnings_surprise'))")

    print("\n6.1 Detecting Events with is_valid()")
    print("    Formula: is_valid(field('earnings_surprise'))")
    print("    Rationale: Returns 1 on earnings days, 0 otherwise.")
    print("               Use to identify when events occur.")

    result = signal.evaluate(data)
    event_count = result.sum().sum()
    print(f"    Total earnings events detected: {int(event_count)}")

    # ---------------------------------------------------------------------
    # Forward-Fill for Persistence
    # ---------------------------------------------------------------------
    # fill_forward() propagates the last known value
    # fill_forward(x, 60) carries the value for up to 60 days
    signal = alpha("fill_forward(field('earnings_surprise'), 60)")

    print("\n6.2 Forward-Fill Earnings Surprise")
    print("    Formula: fill_forward(field('earnings_surprise'), 60)")
    print("    Rationale: After earnings, hold the surprise value for 60 days.")
    print("               This lets us build signals that persist post-event.")

    # ---------------------------------------------------------------------
    # PEAD (Post-Earnings Announcement Drift)
    # ---------------------------------------------------------------------
    # Classic anomaly: stocks drift in direction of earnings surprise
    # Positive surprise -> positive drift for weeks after
    signal = alpha("rank(fill_forward(field('earnings_surprise'), 60))")

    print("\n6.3 PEAD Signal")
    print("    Formula: rank(fill_forward(earnings_surprise, 60))")
    print("    Rationale: Buy stocks with positive surprises (they drift up),")
    print("               sell stocks with negative surprises (they drift down).")

    # ---------------------------------------------------------------------
    # Standardized Unexpected Earnings (SUE)
    # ---------------------------------------------------------------------
    # Normalize surprise by historical surprise volatility
    # ts_std_events() computes std over past N EVENTS (not days)
    signal = alpha(
        "fill_forward("
        "  field('earnings_surprise') / ts_std_events(field('earnings_surprise'), 4),"
        "  60"
        ")"
    )

    print("\n6.4 Standardized Unexpected Earnings (SUE)")
    print("    Formula: surprise / ts_std_events(surprise, 4)")
    print("    Rationale: Normalize by volatility of past 4 surprises.")
    print("               A 5% surprise means more if std is 2% vs 10%.")
    print("    Note: ts_std_events uses past 4 EVENTS, not past 4 days!")

    # ---------------------------------------------------------------------
    # Conditional on Having Event
    # ---------------------------------------------------------------------
    # Only generate signal when we have a recent event
    signal = alpha(
        "where("
        "  ts_count_events(field('earnings_surprise'), 60) > 0,"
        "  fill_forward(field('earnings_surprise'), 60),"
        "  0"
        ")"
    )

    print("\n6.5 Conditional Event Signal")
    print("    Formula: where(ts_count_events(surprise, 60) > 0, ff(surprise), 0)")
    print("    Rationale: Only trade stocks with recent earnings.")
    print("               Avoids stale signals from old events.")


# =============================================================================
# SECTION 7: RISK MANAGEMENT PATTERNS
# =============================================================================

def risk_management_examples():
    """
    RISK MANAGEMENT PATTERNS

    Techniques to control risk in signals:
    1. Winsorization: Cap extreme values
    2. Truncation: Hard limits on position sizes
    3. Scaling: Control total portfolio exposure
    4. Volatility targeting: Adjust for realized volatility
    """
    print("\n" + "="*70)
    print("SECTION 7: RISK MANAGEMENT PATTERNS")
    print("="*70)

    data = create_sample_data(n_days=200, n_tickers=20)

    # ---------------------------------------------------------------------
    # Winsorization (Cap Extremes)
    # ---------------------------------------------------------------------
    # winsorize(x, 0.05) caps at 5th and 95th percentiles
    # Reduces impact of outliers without removing them entirely
    signal = alpha("winsorize(returns(20), 0.05)")

    print("\n7.1 Winsorization")
    print("    Formula: winsorize(returns(20), 0.05)")
    print("    Rationale: Cap extreme returns at 5th/95th percentile.")
    print("               Reduces impact of outliers (M&A, delistings).")

    raw = alpha("returns(20)").evaluate(data)
    winsorized = signal.evaluate(data)

    print(f"    Raw max: {raw.max().max():.4f}, Winsorized max: {winsorized.max().max():.4f}")

    # ---------------------------------------------------------------------
    # Truncation (Hard Position Limits)
    # ---------------------------------------------------------------------
    # truncate(x, max_weight) clips to [-max_weight, max_weight]
    # Use AFTER converting to weights
    signal = alpha("truncate(rank(returns(20)) - 0.5, 0.05)")

    print("\n7.2 Truncation (Position Limits)")
    print("    Formula: truncate(signal, 0.05)")
    print("    Rationale: Hard limit on maximum position size.")
    print("               Prevents concentrated bets.")

    result = signal.evaluate(data)
    print(f"    Max absolute value: {result.abs().max().max():.4f}")

    # ---------------------------------------------------------------------
    # Scaling (Control Leverage)
    # ---------------------------------------------------------------------
    # scale() makes absolute values sum to 1
    signal = alpha("scale(rank(returns(20)) - 0.5)")

    print("\n7.3 Scaling (GMV = 1)")
    print("    Formula: scale(signal)")
    print("    Rationale: Gross market value sums to 1.")
    print("               Controls total leverage/exposure.")

    result = signal.evaluate(data)
    gmv = result.iloc[-1].abs().sum()
    print(f"    Gross Market Value: {gmv:.4f}")

    # ---------------------------------------------------------------------
    # Volatility Targeting
    # ---------------------------------------------------------------------
    # Divide signal by recent volatility to target constant risk
    signal = alpha(
        "scale(rank(returns(20)) - 0.5) / ewma_var(returns(1), 20)"
    )

    print("\n7.4 Volatility Targeting")
    print("    Formula: signal / ewma_var(returns, 20)")
    print("    Rationale: Scale down positions in volatile stocks.")
    print("               Targets equal risk contribution per position.")

    # ---------------------------------------------------------------------
    # Combining Multiple Risk Controls
    # ---------------------------------------------------------------------
    signal = alpha(
        "truncate("
        "  scale("
        "    winsorize(rank(returns(20)) - 0.5, 0.05)"
        "  ),"
        "  0.1"
        ")"
    )

    print("\n7.5 Combined Risk Controls")
    print("    Formula: truncate(scale(winsorize(signal, 0.05)), 0.1)")
    print("    Rationale: Layer multiple controls:")
    print("               1. Winsorize to reduce outlier impact")
    print("               2. Scale to control total exposure")
    print("               3. Truncate to limit max position")


# =============================================================================
# SECTION 8: COMMON GOTCHAS & BEST PRACTICES
# =============================================================================

def gotchas_and_best_practices():
    """
    COMMON GOTCHAS & BEST PRACTICES

    Things that catch people off guard when using alpha-parser.
    """
    print("\n" + "="*70)
    print("SECTION 8: COMMON GOTCHAS & BEST PRACTICES")
    print("="*70)

    data = create_sample_data(n_days=100, n_tickers=5)

    # =====================================================================
    # GOTCHA 1: rank() and quantile() are ASCENDING
    # =====================================================================
    print("\n8.1 GOTCHA: rank() and quantile() are ASCENDING")
    print("    Higher value -> Higher rank")
    print("    ")
    print("    If you want to BUY high-momentum stocks:")
    print("      CORRECT:   rank(returns(20))")
    print("      WRONG:     rank(-returns(20))  # This buys LOW momentum!")
    print("    ")
    print("    If you want to BUY low P/E stocks (value):")
    print("      CORRECT:   rank(-pe_ratio)  # Negate so low P/E -> high rank")
    print("      WRONG:     rank(pe_ratio)   # This buys HIGH P/E (growth)!")

    # =====================================================================
    # GOTCHA 2: Window warmup produces NaN
    # =====================================================================
    print("\n8.2 GOTCHA: Window warmup produces NaN")
    print("    ts_mean(x, 20) has NaN for first 19 rows")
    print("    returns(20) has NaN for first 20 rows")
    print("    ")
    print("    Nested operations compound this:")
    print("      ts_mean(returns(20), 10) -> NaN for first 29 rows")
    print("    ")
    print("    BEST PRACTICE: Check result.first_valid_index()")

    signal = alpha("ts_mean(returns(20), 10)")
    result = signal.evaluate(data)
    print(f"    Example: first valid row is index {result.first_valid_index()}")

    # =====================================================================
    # GOTCHA 3: Cross-sectional vs Time-series operations
    # =====================================================================
    print("\n8.3 GOTCHA: Cross-sectional vs Time-series operations")
    print("    ")
    print("    CROSS-SECTIONAL (work across stocks within each day):")
    print("      rank(), zscore(), demean(), quantile(), scale()")
    print("      These normalize ACROSS columns (tickers)")
    print("    ")
    print("    TIME-SERIES (work across time for each stock):")
    print("      ts_mean(), ts_std(), ewma(), delay(), returns()")
    print("      These compute ALONG rows (dates)")
    print("    ")
    print("    EXAMPLE:")
    print("      zscore(returns(20))  # Rank returns across stocks")
    print("      (returns - ts_mean(returns, 60)) / ts_std(returns, 60)  # Z-score over time")

    # =====================================================================
    # GOTCHA 4: ts_beta argument order matters
    # =====================================================================
    print("\n8.4 GOTCHA: ts_beta() argument order matters")
    print("    ts_beta(Y, X, window) regresses Y on X")
    print("    Result = cov(Y, X) / var(X)")
    print("    ")
    print("    For hedge ratio of stock vs market:")
    print("      ts_beta(stock_returns, market_returns, 60)")
    print("    NOT:")
    print("      ts_beta(market_returns, stock_returns, 60)  # This is inverted!")

    # =====================================================================
    # GOTCHA 5: EWMA halflife vs rolling period
    # =====================================================================
    print("\n8.5 GOTCHA: EWMA halflife vs rolling period")
    print("    ewma(x, halflife=10) and ts_mean(x, period=20) are NOT equivalent")
    print("    ")
    print("    halflife=10 means: weight decays by 50% every 10 periods")
    print("    Effective window is ~3x halflife (30 periods for halflife=10)")
    print("    ")
    print("    RULE OF THUMB: halflife ≈ period / 3")
    print("      ewma(x, 10) ≈ ts_mean(x, 30) in terms of lookback")

    # =====================================================================
    # GOTCHA 6: Group data must be provided
    # =====================================================================
    print("\n8.6 GOTCHA: Group operations need group data")
    print("    group_demean(returns(5), 'sector') requires:")
    print("      data['sector'] = DataFrame of sector assignments")
    print("    ")
    print("    The group DataFrame must have:")
    print("      - Same index as price data")
    print("      - Same columns as price data")
    print("      - Values are group identifiers (strings or ints)")

    # =====================================================================
    # BEST PRACTICE: Use compute_context() for caching
    # =====================================================================
    print("\n8.7 BEST PRACTICE: Use compute_context() for caching")
    print("    Wrap multiple evaluations in compute_context():")
    print("    ")
    print("    with compute_context():")
    print("        signal1 = alpha('returns(20)')  # Computed once")
    print("        signal2 = alpha('rank(returns(20))')  # Reuses returns(20)")
    print("        result1 = signal1.evaluate(data)")
    print("        result2 = signal2.evaluate(data)")
    print("    ")
    print("    This avoids recomputing shared sub-expressions.")

    with compute_context():
        signal1 = alpha("returns(20)")
        signal2 = alpha("rank(returns(20))")
        result1 = signal1.evaluate(data)
        result2 = signal2.evaluate(data)
    print("    (Caching demonstration complete)")

    # =====================================================================
    # BEST PRACTICE: Center signals before converting to weights
    # =====================================================================
    print("\n8.8 BEST PRACTICE: Center signals before to_weights()")
    print("    rank() returns values in [0, 1]")
    print("    This means ALL positions are long!")
    print("    ")
    print("    For long/short portfolio, center first:")
    print("      rank(x) - 0.5  # Now ranges from -0.5 to +0.5")
    print("    ")
    print("    Or use demean():")
    print("      demean(rank(x))  # Automatically centers")


# =============================================================================
# MAIN: Run all examples
# =============================================================================

if __name__ == '__main__':
    """Run all cookbook examples."""

    print("="*70)
    print("ALPHA PARSER COOKBOOK")
    print("="*70)
    print("\nThis cookbook demonstrates common patterns for building")
    print("quantitative trading signals with alpha-parser.")
    print("\nRunning all examples...\n")

    # Run each section
    basics_example()
    momentum_examples()
    mean_reversion_examples()
    pairs_trading_examples()
    factor_neutral_examples()
    event_driven_examples()
    risk_management_examples()
    gotchas_and_best_practices()

    print("\n" + "="*70)
    print("COOKBOOK COMPLETE")
    print("="*70)
    print("\nFor more information, see:")
    print("  - README.md for API reference")
    print("  - tests/test_operators.py for additional examples")
    print("  - CLAUDE.md for quick reference")
