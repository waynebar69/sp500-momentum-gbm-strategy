"""Trading strategy logic for the S&P 500 momentum GBM project.

This module links the database layer to the quantitative model. It is
responsible for:
- extracting historical prices for a given trading day
- calibrating the GBM model on a rolling lookback window
- forecasting future price movement
- estimating downside risk via expected shortfall
- converting the signal into a target position size

The implementation is intentionally simple, transparent, and explainable,
which is appropriate for a Masters-level coursework project and for a
portfolio piece that may be reviewed by recruiters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.database import DEFAULT_INSTRUMENT, DatabaseManager
from src.gbm import GBM, TRADING_DAYS_PER_YEAR


@dataclass
class StrategyConfig:
    """Configuration for the momentum strategy.

    Parameters
    ----------
    instrument : str, optional
        Instrument identifier used in the database.
    lookback_days : int, optional
        Number of historical daily prices used to calibrate the GBM model.
    horizon_days : int, optional
        Number of trading days ahead for the forecast.
    forecast_confidence : float, optional
        Central confidence level used for the forecast interval.
    expected_shortfall_alpha : float, optional
        Confidence level used for expected shortfall.
    max_leverage : float, optional
        Maximum notional exposure as a fraction of portfolio value.
    min_history : int, optional
        Minimum number of prices required before a signal is generated.
    trade_lot_size : int, optional
        Position targets are rounded to the nearest lot size.
    allow_short : bool, optional
        Whether the strategy may take short positions.
    """

    instrument: str = DEFAULT_INSTRUMENT
    lookback_days: int = 125
    horizon_days: int = 10
    forecast_confidence: float = 0.90
    expected_shortfall_alpha: float = 0.95
    max_leverage: float = 1.0
    min_history: int = 30
    trade_lot_size: int = 1
    allow_short: bool = False


class MomentumStrategy:
    """Momentum strategy driven by a GBM forecast and expected shortfall.

    Parameters
    ----------
    database : DatabaseManager
        Database manager used to fetch prices and positions.
    config : StrategyConfig | None, optional
        Strategy configuration. If omitted, default settings are used.
    model_random_state : int, optional
        Random seed used for the GBM bootstrap calibration.
    model_bootstrap_samples : int, optional
        Number of bootstrap samples used in GBM calibration.
    """

    def __init__(
        self,
        database: DatabaseManager,
        config: StrategyConfig | None = None,
        model_random_state: int = 42,
        model_bootstrap_samples: int = 5000,
    ) -> None:
        self.database = database
        self.config = config or StrategyConfig()
        self.model_random_state = model_random_state
        self.model_bootstrap_samples = model_bootstrap_samples

    def analyse(self, which_day: str) -> Dict[str, float]:
        """Analyse the market on a given day and generate a trading signal.

        Parameters
        ----------
        which_day : str
            Trading day in ISO date format.

        Returns
        -------
        dict
            Dictionary containing forecast, expected shortfall, and target size.
        """
        prices = self.database.get_prices_up_to_day(
            which_day=which_day,
            lookback=self.config.lookback_days,
            instrument=self.config.instrument,
        )

        if len(prices) < self.config.min_history:
            latest_price = self.database.get_latest_price(
                which_day=which_day,
                instrument=self.config.instrument,
            )
            quantity, cash = self.database.get_latest_position(
                which_day=which_day,
                instrument=self.config.instrument,
            )
            portfolio_value = cash + quantity * latest_price

            return {
                "which_day": which_day,
                "latest_price": latest_price,
                "mu": np.nan,
                "sigma": np.nan,
                "expected_price": np.nan,
                "forecast_lower": np.nan,
                "forecast_upper": np.nan,
                "expected_shortfall": np.nan,
                "signal_strength": 0.0,
                "target_quantity": quantity,
                "portfolio_value": portfolio_value,
            }

        price_array = np.asarray(prices, dtype=float)
        latest_price = float(price_array[-1])
        dt = 1.0 / TRADING_DAYS_PER_YEAR
        horizon = self.config.horizon_days / TRADING_DAYS_PER_YEAR

        model = GBM(
            n_bootstrap=self.model_bootstrap_samples,
            random_state=self.model_random_state,
        )
        model.calibrate(price_array, dt=dt)

        forecast = model.forecast(
            latest=latest_price,
            horizon=horizon,
            confidence=self.config.forecast_confidence,
        )
        expected_shortfall = model.expected_shortfall(
            horizon=horizon,
            alpha=self.config.expected_shortfall_alpha,
        )

        target_quantity = self.position_size(
            which_day=which_day,
            latest_price=latest_price,
            forecast_price=forecast["expected_price"],
            expected_shortfall=expected_shortfall,
        )

        quantity, cash = self.database.get_latest_position(
            which_day=which_day,
            instrument=self.config.instrument,
        )
        portfolio_value = cash + quantity * latest_price
        mu, sigma = model.parameters()

        return {
            "which_day": which_day,
            "latest_price": latest_price,
            "mu": mu,
            "sigma": sigma,
            "expected_price": forecast["expected_price"],
            "forecast_lower": forecast["interval"][0],
            "forecast_upper": forecast["interval"][1],
            "expected_shortfall": expected_shortfall,
            "signal_strength": self._signal_strength(
                latest_price=latest_price,
                forecast_price=forecast["expected_price"],
                expected_shortfall=expected_shortfall,
            ),
            "target_quantity": target_quantity,
            "portfolio_value": portfolio_value,
        }

    def position_size(
        self,
        which_day: str,
        latest_price: float,
        forecast_price: float,
        expected_shortfall: float,
    ) -> float:
        """Convert a forecast and risk estimate into a target position size.

        The sizing rule is deliberately transparent. It compares the expected
        upside from the forecast to the downside tail risk measured by expected
        shortfall. Larger positive signals receive larger allocations, subject
        to a leverage cap and available portfolio value.

        Parameters
        ----------
        which_day : str
            Trading day in ISO date format.
        latest_price : float
            Latest available price on or before the trading day.
        forecast_price : float
            Expected future price from the GBM model.
        expected_shortfall : float
            Expected shortfall expressed as a positive relative loss.

        Returns
        -------
        float
            Target quantity to hold after trading on `which_day`.
        """
        quantity, cash = self.database.get_latest_position(
            which_day=which_day,
            instrument=self.config.instrument,
        )
        portfolio_value = cash + quantity * latest_price

        if portfolio_value <= 0 or latest_price <= 0:
            return 0.0

        signal_strength = self._signal_strength(
            latest_price=latest_price,
            forecast_price=forecast_price,
            expected_shortfall=expected_shortfall,
        )

        if not self.config.allow_short:
            signal_strength = max(signal_strength, 0.0)
        else:
            signal_strength = np.clip(signal_strength, -self.config.max_leverage, self.config.max_leverage)

        target_notional = (
            np.clip(signal_strength, 0.0, self.config.max_leverage) * portfolio_value
            if not self.config.allow_short
            else signal_strength * portfolio_value
        )
        raw_target_quantity = target_notional / latest_price
        return self._round_to_lot_size(raw_target_quantity)

    def _signal_strength(
        self,
        latest_price: float,
        forecast_price: float,
        expected_shortfall: float,
    ) -> float:
        """Return a risk-adjusted momentum score.

        The numerator measures forecast upside as a percentage of the latest
        price. The denominator uses expected shortfall as a relative downside
        risk measure.
        """
        expected_return = (forecast_price - latest_price) / latest_price
        risk_scale = max(expected_shortfall, 1e-8)
        return float(expected_return / risk_scale)

    def _round_to_lot_size(self, quantity: float) -> float:
        """Round the target quantity to the configured trade lot size."""
        lot_size = self.config.trade_lot_size
        if lot_size <= 0:
            raise ValueError("trade_lot_size must be positive")

        rounded_quantity = lot_size * round(quantity / lot_size)
        return float(rounded_quantity)


def analyse_day(
    database: DatabaseManager,
    which_day: str,
    config: StrategyConfig | None = None,
) -> Dict[str, float]:
    """Convenience function to analyse a single day using default strategy settings."""
    strategy = MomentumStrategy(database=database, config=config)
    return strategy.analyse(which_day)
