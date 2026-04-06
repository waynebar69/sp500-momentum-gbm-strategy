"""Backtesting engine for the S&P 500 momentum GBM strategy.

This module orchestrates the end-to-end trading workflow:
- prepare the SQLite database from CSV data
- analyse each trading day in the test period
- rebalance the portfolio to the target position
- record updated portfolio states in the database
- build a portfolio history DataFrame for evaluation

The implementation is intentionally transparent and suitable for both a
coursework notebook and a portfolio-quality GitHub repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from src.database import (
    DEFAULT_CSV_PATH,
    DEFAULT_DB_PATH,
    DEFAULT_INSTRUMENT,
    DEFAULT_INITIAL_CASH,
    DatabaseManager,
)
from src.strategy import MomentumStrategy, StrategyConfig


@dataclass
class BacktestConfig:
    """Configuration for running the backtest.

    Parameters
    ----------
    db_path : str | Path, optional
        SQLite database path.
    csv_path : str | Path, optional
        CSV file containing historical prices.
    instrument : str, optional
        Instrument identifier used in the database.
    initial_cash : float, optional
        Starting cash balance.
    begin_on : str, optional
        First trading day of the test period.
    end_on : str | None, optional
        Last trading day of the test period. If None, use all remaining days.
    allow_fractional_trades : bool, optional
        Whether target quantities may be fractional.
    """

    db_path: str | Path = DEFAULT_DB_PATH
    csv_path: str | Path = DEFAULT_CSV_PATH
    instrument: str = DEFAULT_INSTRUMENT
    initial_cash: float = DEFAULT_INITIAL_CASH
    begin_on: str = "2020-06-01"
    end_on: str | None = "2021-05-31"
    allow_fractional_trades: bool = False


class Backtester:
    """Run and evaluate the momentum strategy backtest."""

    def __init__(
        self,
        backtest_config: BacktestConfig | None = None,
        strategy_config: StrategyConfig | None = None,
    ) -> None:
        self.backtest_config = backtest_config or BacktestConfig()
        self.strategy_config = strategy_config or StrategyConfig()

    def prepare(self) -> int:
        """Create and populate the SQLite database for a fresh run."""
        with DatabaseManager(self.backtest_config.db_path) as database:
            return database.prepare_database(
                csv_path=self.backtest_config.csv_path,
                instrument=self.backtest_config.instrument,
                initial_cash=self.backtest_config.initial_cash,
            )

    def run(self, prepare_database_first: bool = True) -> pd.DataFrame:
        """Run the backtest and return a portfolio history DataFrame.

        Parameters
        ----------
        prepare_database_first : bool, optional
            If True, rebuild the database from the CSV before running.

        Returns
        -------
        pd.DataFrame
            Daily backtest history including holdings, cash, and returns.
        """
        if prepare_database_first:
            self.prepare()

        with DatabaseManager(self.backtest_config.db_path) as database:
            strategy = MomentumStrategy(database=database, config=self.strategy_config)
            trading_days = self._get_backtest_days(database)

            for which_day in trading_days:
                analysis = strategy.analyse(which_day)
                latest_price = analysis["latest_price"]
                target_quantity = analysis["target_quantity"]

                current_quantity, current_cash = database.get_latest_position(
                    which_day=which_day,
                    instrument=self.backtest_config.instrument,
                )

                if not self.backtest_config.allow_fractional_trades:
                    target_quantity = round(target_quantity)
                    current_quantity = round(current_quantity)

                trade_quantity = target_quantity - current_quantity

                if trade_quantity == 0:
                    continue

                new_cash = current_cash - trade_quantity * latest_price

                database.insert_position(
                    time_of_trade=which_day,
                    quantity=target_quantity,
                    cash=new_cash,
                    instrument=self.backtest_config.instrument,
                )

            return self.portfolio_history()

    def portfolio_history(self) -> pd.DataFrame:
        """Build a daily portfolio history DataFrame from the database."""
        with DatabaseManager(self.backtest_config.db_path) as database:
            prices = self._load_prices(database)
            positions = self._load_positions(database)

        positions = positions.sort_values("time_of_trade").rename(
            columns={"time_of_trade": "theday"}
        )
        prices = prices.sort_values("theday")

        merged = pd.merge_asof(
            prices,
            positions,
            on="theday",
            by="instrument",
            direction="backward",
        )

        merged["quantity"] = merged["quantity"].astype(float)
        merged["cash"] = merged["cash"].astype(float)
        merged["holdings_value"] = merged["quantity"] * merged["price"]
        merged["portfolio_value"] = merged["cash"] + merged["holdings_value"]
        merged["daily_return"] = merged["portfolio_value"].pct_change().fillna(0.0)
        merged["benchmark_return"] = merged["price"].pct_change().fillna(0.0)
        merged["cum_strategy_return"] = (
            1.0 + merged["daily_return"]
        ).cumprod() - 1.0
        merged["cum_benchmark_return"] = (
            1.0 + merged["benchmark_return"]
        ).cumprod() - 1.0

        mask = merged["theday"] >= self.backtest_config.begin_on
        if self.backtest_config.end_on is not None:
            mask &= merged["theday"] <= self.backtest_config.end_on

        return merged.loc[mask].reset_index(drop=True)

    def performance_summary(self) -> pd.Series:
        """Return headline performance metrics for the backtest."""
        history = self.portfolio_history()

        mean_daily_return = history["daily_return"].mean()
        std_daily_return = history["daily_return"].std(ddof=1)
        sharpe_ratio = (
            (mean_daily_return / std_daily_return) * (252**0.5)
            if std_daily_return > 0
            else 0.0
        )

        total_return = history["portfolio_value"].iloc[-1] / history["portfolio_value"].iloc[0] - 1.0
        benchmark_total_return = (
            history["price"].iloc[-1] / history["price"].iloc[0] - 1.0
        )

        return pd.Series(
            {
                "start_date": history["theday"].iloc[0],
                "end_date": history["theday"].iloc[-1],
                "initial_portfolio_value": history["portfolio_value"].iloc[0],
                "final_portfolio_value": history["portfolio_value"].iloc[-1],
                "strategy_total_return": total_return,
                "benchmark_total_return": benchmark_total_return,
                "mean_daily_return": mean_daily_return,
                "std_daily_return": std_daily_return,
                "sharpe_ratio": sharpe_ratio,
                "best_day": history.loc[history["daily_return"].idxmax(), "theday"],
                "worst_day": history.loc[history["daily_return"].idxmin(), "theday"],
                "best_day_return": history["daily_return"].max(),
                "worst_day_return": history["daily_return"].min(),
            }
        )

    def _get_backtest_days(self, database: DatabaseManager) -> list[str]:
        """Return the trading days in the configured backtest window."""
        all_days = database.get_all_trading_days(start_date=self.backtest_config.begin_on)

        if self.backtest_config.end_on is None:
            return all_days

        return [day for day in all_days if day <= self.backtest_config.end_on]

    def _load_prices(self, database: DatabaseManager) -> pd.DataFrame:
        """Load prices from SQLite into a DataFrame."""
        query = """
            SELECT theday, instrument, price
            FROM prices
            WHERE instrument = ?
            ORDER BY theday ASC;
        """
        return pd.read_sql_query(
            query,
            database.connection,
            params=(self.backtest_config.instrument,),
        )

    def _load_positions(self, database: DatabaseManager) -> pd.DataFrame:
        """Load positions from SQLite into a DataFrame."""
        query = """
            SELECT time_of_trade, instrument, quantity, cash
            FROM positions
            WHERE instrument = ?
            ORDER BY time_of_trade ASC;
        """
        return pd.read_sql_query(
            query,
            database.connection,
            params=(self.backtest_config.instrument,),
        )


def run_backtest(
    db_path: str | Path = DEFAULT_DB_PATH,
    csv_path: str | Path = DEFAULT_CSV_PATH,
    begin_on: str = "2020-06-01",
    end_on: Optional[str] = "2021-05-31",
    initial_cash: float = DEFAULT_INITIAL_CASH,
    strategy_config: StrategyConfig | None = None,
) -> pd.DataFrame:
    """Convenience function to run the full backtest in one call."""
    backtester = Backtester(
        backtest_config=BacktestConfig(
            db_path=db_path,
            csv_path=csv_path,
            begin_on=begin_on,
            end_on=end_on,
            initial_cash=initial_cash,
        ),
        strategy_config=strategy_config,
    )
    return backtester.run(prepare_database_first=True)
