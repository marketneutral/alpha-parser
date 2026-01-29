"""Multi-factor risk model.

This module implements a multi-factor risk model using alpha-parser signals
for factor exposure computation. The model decomposes stock returns into
systematic (factor) risk and idiosyncratic (specific) risk.

Example:
    from alpha_parser import LazyData
    from alpha_parser.risk import FactorRiskModel
    import pandas as pd

    data = LazyData({
        'close': lambda: pd.read_parquet('data/fmp/close.parquet'),
        'volume': lambda: pd.read_parquet('data/fmp/volume.parquet'),
        'market_cap': lambda: pd.read_parquet('data/fmp/market_cap.parquet'),
        'book_to_price': lambda: pd.read_parquet('data/fmp/book_to_price.parquet'),
        'sector': lambda: pd.read_parquet('data/fmp/sector.parquet'),
    })

    # Fit the model
    model = FactorRiskModel()
    model.fit(data, lookback=252)

    # Get factor exposures for a specific date
    exposures = model.get_exposures(data, date='2024-01-15')

    # Compute portfolio risk
    weights = pd.Series({'AAPL': 0.1, 'MSFT': 0.1, ...})
    total_risk, factor_risk, specific_risk = model.portfolio_risk(weights)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .parser import alpha
from .context import compute_context


@dataclass
class FactorDefinition:
    """Definition of a style factor."""
    name: str
    expression: str
    description: str = ""


# Default style factor definitions
# Note: Using rank() instead of zscore() for robustness to outliers
DEFAULT_STYLE_FACTORS = [
    FactorDefinition(
        name="size",
        expression="rank(field('market_cap'))",
        description="Market capitalization rank",
    ),
    FactorDefinition(
        name="value",
        expression="rank(winsorize(field('book_to_price'), 0.05))",
        description="Book-to-price ratio rank",
    ),
    FactorDefinition(
        name="momentum",
        expression="rank(returns(252) - returns(21))",
        description="12-month minus 1-month returns",
    ),
    FactorDefinition(
        name="volatility",
        expression="rank(volatility(60))",
        description="60-day realized volatility",
    ),
    FactorDefinition(
        name="liquidity",
        expression="rank(ts_mean(volume(1), 20))",
        description="20-day average dollar volume",
    ),
    FactorDefinition(
        name="short_term_reversal",
        expression="rank(-returns(5))",
        description="5-day reversal",
    ),
]


# Simplified factor definitions that work without fundamental data
PRICE_ONLY_FACTORS = [
    FactorDefinition(
        name="momentum",
        expression="rank(returns(252) - returns(21))",
        description="12-month minus 1-month returns",
    ),
    FactorDefinition(
        name="volatility",
        expression="rank(volatility(60))",
        description="60-day realized volatility",
    ),
    FactorDefinition(
        name="liquidity",
        expression="rank(ts_mean(volume(1), 20))",
        description="20-day average dollar volume",
    ),
    FactorDefinition(
        name="short_term_reversal",
        expression="rank(-returns(5))",
        description="5-day reversal",
    ),
    FactorDefinition(
        name="medium_term_reversal",
        expression="rank(-returns(20))",
        description="20-day reversal",
    ),
]


@dataclass
class RiskModelResults:
    """Results from fitting the risk model."""
    factor_returns: pd.DataFrame  # T x K factor returns
    factor_covariance: pd.DataFrame  # K x K factor covariance
    specific_variance: pd.Series  # N x 1 specific variance per stock
    r_squared: pd.Series  # T x 1 R-squared per date
    factor_exposures: pd.DataFrame  # Last available exposures (N x K)


class FactorRiskModel:
    """Multi-factor risk model.

    The model decomposes stock returns into:
        r_i = sum_k(X_ik * f_k) + epsilon_i

    Where:
        r_i = return of stock i
        X_ik = exposure of stock i to factor k
        f_k = return of factor k
        epsilon_i = specific (idiosyncratic) return

    Attributes:
        style_factors: List of style factor definitions
        include_industries: Whether to include industry dummy factors
        industry_column: Name of industry column in data
        half_life: Half-life for exponential weighting (days)
    """

    def __init__(
        self,
        style_factors: Optional[List[FactorDefinition]] = None,
        include_industries: bool = True,
        industry_column: str = 'sector',
        half_life: int = 63,  # ~3 months
    ):
        self.style_factors = style_factors or DEFAULT_STYLE_FACTORS
        self.include_industries = include_industries
        self.industry_column = industry_column
        self.half_life = half_life

        # Results populated after fit()
        self.results_: Optional[RiskModelResults] = None
        self._style_signals = {}
        self._industries: List[str] = []

    def _compute_style_exposures(
        self,
        data,
        factors: List[FactorDefinition],
    ) -> Dict[str, pd.DataFrame]:
        """Compute style factor exposures using alpha-parser signals."""
        exposures = {}

        with compute_context():
            for factor in factors:
                try:
                    signal = alpha(factor.expression)
                    exposure = signal.evaluate(data)
                    exposures[factor.name] = exposure
                except Exception as e:
                    print(f"Warning: Could not compute factor '{factor.name}': {e}")
                    continue

        return exposures

    def _compute_industry_exposures(
        self,
        data,
        reference_df: pd.DataFrame,
    ) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """Compute industry dummy exposures."""
        from .data import resolve_data
        data = resolve_data(data)

        if self.industry_column not in data:
            return {}, []

        industry_df = data[self.industry_column]

        # Get unique industries
        all_industries = set()
        for col in industry_df.columns:
            all_industries.update(industry_df[col].dropna().unique())
        industries = sorted(all_industries)

        # Create dummy DataFrames for each industry
        exposures = {}
        for industry in industries:
            dummy = (industry_df == industry).astype(float)
            dummy = dummy.reindex(index=reference_df.index, columns=reference_df.columns)
            exposures[f"ind_{industry}"] = dummy

        return exposures, industries

    def _compute_returns(self, data) -> pd.DataFrame:
        """Compute daily returns from close prices."""
        from .data import resolve_data
        data = resolve_data(data)
        close = data['close']
        returns = close.pct_change()
        return returns

    def _exponential_weights(self, n: int) -> np.ndarray:
        """Compute exponential decay weights."""
        decay = np.log(2) / self.half_life
        weights = np.exp(-decay * np.arange(n)[::-1])
        return weights / weights.sum()

    def fit(
        self,
        data,
        lookback: int = 252,
        min_observations: int = 10,
    ) -> 'FactorRiskModel':
        """Fit the risk model using cross-sectional regression.

        Args:
            data: Data dict or LazyData with price and factor data
            lookback: Number of days to use for estimation
            min_observations: Minimum observations required per stock

        Returns:
            self (fitted model)
        """
        print("Computing returns...")
        returns = self._compute_returns(data)

        # Trim to lookback period
        returns = returns.iloc[-lookback:]
        dates = returns.index

        print(f"Computing style factor exposures...")
        style_exposures = self._compute_style_exposures(data, self.style_factors)

        # Align style exposures to returns dates
        for name, exp_df in style_exposures.items():
            style_exposures[name] = exp_df.reindex(index=dates)

        # Industry exposures
        industry_exposures = {}
        if self.include_industries:
            print("Computing industry exposures...")
            industry_exposures, self._industries = self._compute_industry_exposures(
                data, returns
            )
            for name, exp_df in industry_exposures.items():
                industry_exposures[name] = exp_df.reindex(index=dates)

        # Combine all exposures
        all_exposures = {**style_exposures, **industry_exposures}
        factor_names = list(all_exposures.keys())

        print(f"Running cross-sectional regressions for {len(dates)} dates...")
        print(f"  Style factors: {list(style_exposures.keys())}")
        if self.include_industries:
            print(f"  Industry factors: {len(industry_exposures)} industries")

        # Cross-sectional regression for each date
        factor_returns_list = []
        r_squared_list = []

        for i, date in enumerate(dates):
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{len(dates)}")

            # Get returns for this date
            y = returns.loc[date].dropna()
            if len(y) < min_observations:
                continue

            # Build exposure matrix for this date
            X_data = {}
            for factor_name in factor_names:
                exp_series = all_exposures[factor_name].loc[date]
                X_data[factor_name] = exp_series

            X = pd.DataFrame(X_data)
            X = X.loc[y.index]  # Align to stocks with returns

            # Drop rows with any NaN
            valid_mask = X.notna().all(axis=1) & y.notna()
            X_valid = X.loc[valid_mask]
            y_valid = y.loc[valid_mask]

            if len(y_valid) < min_observations:
                continue

            # Add constant for market factor
            X_with_const = sm.add_constant(X_valid, has_constant='add')

            try:
                # WLS regression (could add weights here)
                model = sm.OLS(y_valid, X_with_const)
                result = model.fit()

                # Extract factor returns
                factor_ret = result.params.drop('const', errors='ignore')
                factor_ret['market'] = result.params.get('const', 0)
                factor_ret.name = date

                factor_returns_list.append(factor_ret)
                r_squared_list.append(result.rsquared)

            except Exception as e:
                print(f"  Warning: Regression failed for {date}: {e}")
                continue

        if not factor_returns_list:
            raise ValueError("No successful regressions - check data quality")

        # Combine factor returns
        factor_returns = pd.DataFrame(factor_returns_list)
        factor_returns.index.name = 'date'

        # Compute factor covariance with exponential weighting
        print("Estimating factor covariance...")
        weights = self._exponential_weights(len(factor_returns))
        weighted_returns = factor_returns.mul(np.sqrt(weights), axis=0)

        # Annualize covariance (252 trading days)
        factor_cov = weighted_returns.cov() * 252

        # Estimate specific variance
        print("Estimating specific risk...")
        specific_var = self._estimate_specific_variance(
            returns, all_exposures, factor_returns, dates
        )

        # Get last available exposures
        last_date = dates[-1]
        last_exposures = pd.DataFrame({
            name: all_exposures[name].loc[last_date]
            for name in factor_names
        })

        self.results_ = RiskModelResults(
            factor_returns=factor_returns,
            factor_covariance=factor_cov,
            specific_variance=specific_var,
            r_squared=pd.Series(r_squared_list, index=factor_returns.index),
            factor_exposures=last_exposures,
        )

        print(f"Done! Model fitted with {len(factor_returns)} dates")
        print(f"  Average R-squared: {np.mean(r_squared_list):.3f}")

        return self

    def _estimate_specific_variance(
        self,
        returns: pd.DataFrame,
        exposures: Dict[str, pd.DataFrame],
        factor_returns: pd.DataFrame,
        dates: pd.DatetimeIndex,
    ) -> pd.Series:
        """Estimate specific (idiosyncratic) variance for each stock."""
        # Compute residuals
        residuals_list = []

        for date in factor_returns.index:
            y = returns.loc[date]
            f = factor_returns.loc[date]

            # Predicted returns
            predicted = pd.Series(0.0, index=y.index)
            for factor_name in f.index:
                if factor_name == 'market':
                    predicted += f['market']
                elif factor_name in exposures:
                    exp = exposures[factor_name].loc[date]
                    predicted += exp * f[factor_name]

            residual = y - predicted
            residuals_list.append(residual)

        residuals_df = pd.DataFrame(residuals_list)

        # Compute variance with exponential weighting
        weights = self._exponential_weights(len(residuals_df))
        weighted_residuals = residuals_df.mul(np.sqrt(weights), axis=0)

        # Annualized specific variance
        specific_var = weighted_residuals.var() * 252

        return specific_var

    def get_exposures(
        self,
        data,
        date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get factor exposures for a specific date.

        Args:
            data: Data dict or LazyData
            date: Date string (default: most recent)

        Returns:
            DataFrame with stocks as rows, factors as columns
        """
        style_exposures = self._compute_style_exposures(data, self.style_factors)

        if date is None:
            # Use last available date
            date = list(style_exposures.values())[0].index[-1]
        else:
            date = pd.Timestamp(date)

        # Get style exposures for date
        result = pd.DataFrame({
            name: exp_df.loc[date]
            for name, exp_df in style_exposures.items()
        })

        # Add industry exposures
        if self.include_industries:
            industry_exposures, _ = self._compute_industry_exposures(
                data, list(style_exposures.values())[0]
            )
            for name, exp_df in industry_exposures.items():
                if date in exp_df.index:
                    result[name] = exp_df.loc[date]

        return result

    def portfolio_risk(
        self,
        weights: pd.Series,
        exposures: Optional[pd.DataFrame] = None,
    ) -> Tuple[float, float, float]:
        """Compute portfolio risk decomposition.

        Args:
            weights: Portfolio weights indexed by ticker
            exposures: Factor exposures (default: use fitted exposures)

        Returns:
            Tuple of (total_risk, factor_risk, specific_risk) as annualized std
        """
        if self.results_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if exposures is None:
            exposures = self.results_.factor_exposures

        # Align weights to exposures
        common_stocks = weights.index.intersection(exposures.index)
        w = weights.loc[common_stocks].values
        X = exposures.loc[common_stocks].values

        # Portfolio factor exposures
        portfolio_exposures = X.T @ w  # K x 1

        # Factor covariance (only for factors in exposures)
        factor_names = exposures.columns.tolist()
        cov_factors = [f for f in factor_names if f in self.results_.factor_covariance.index]
        F = self.results_.factor_covariance.loc[cov_factors, cov_factors].values

        # Map portfolio exposures to covariance matrix order
        exp_idx = [factor_names.index(f) for f in cov_factors]
        p_exp = portfolio_exposures[exp_idx]

        # Factor variance
        factor_var = p_exp.T @ F @ p_exp

        # Specific variance
        specific_var = self.results_.specific_variance.loc[common_stocks].values
        specific_var = np.nan_to_num(specific_var, nan=np.nanmean(specific_var))
        specific_portfolio_var = (w ** 2) @ specific_var

        # Total variance
        total_var = factor_var + specific_portfolio_var

        # Return as standard deviation (annualized)
        return (
            np.sqrt(total_var),
            np.sqrt(factor_var),
            np.sqrt(specific_portfolio_var),
        )

    def factor_attribution(
        self,
        weights: pd.Series,
        exposures: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Compute risk attribution to each factor.

        Args:
            weights: Portfolio weights
            exposures: Factor exposures (default: use fitted)

        Returns:
            Series with variance contribution of each factor
        """
        if self.results_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if exposures is None:
            exposures = self.results_.factor_exposures

        # Align weights
        common_stocks = weights.index.intersection(exposures.index)
        w = weights.loc[common_stocks].values
        X = exposures.loc[common_stocks].values

        # Portfolio exposures
        portfolio_exp = X.T @ w

        # Marginal contribution to risk
        factor_names = exposures.columns.tolist()
        cov_factors = [f for f in factor_names if f in self.results_.factor_covariance.index]

        attribution = {}
        for i, factor in enumerate(factor_names):
            if factor in cov_factors:
                idx = cov_factors.index(factor)
                exp_idx = [factor_names.index(f) for f in cov_factors]
                p_exp = portfolio_exp[exp_idx]
                F = self.results_.factor_covariance.loc[cov_factors, cov_factors].values

                # Contribution = exposure_k * sum_j(F_kj * exposure_j)
                contrib = p_exp[idx] * (F[idx, :] @ p_exp)
                attribution[factor] = contrib

        return pd.Series(attribution)

    @property
    def factor_covariance(self) -> pd.DataFrame:
        """Get the fitted factor covariance matrix."""
        if self.results_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.results_.factor_covariance

    @property
    def factor_returns(self) -> pd.DataFrame:
        """Get the estimated factor returns time series."""
        if self.results_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.results_.factor_returns

    @property
    def specific_risk(self) -> pd.Series:
        """Get specific risk (annualized std) for each stock."""
        if self.results_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return np.sqrt(self.results_.specific_variance)
