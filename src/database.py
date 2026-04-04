"""SQLite database utilities for the S&P 500 momentum strategy.

This module provides a small data-access layer for the project. It is designed
for both coursework reproducibility and portfolio-quality structure.

Main responsibilities
---------------------
- create the SQLite database schema
- load S&P 500 prices from a CSV file
- initialise the positions table
- provide helper queries for prices, dates, and positions
- record strategy trades safely using parameterised SQL
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

DEFAULT_DB_PATH = Path("data/SP500.db")
DEFAULT_CSV_PATH = Path("data/SP500.csv")
DEFAULT_INSTRUMENT = "SP500"
DEFAULT_INITIAL_CASH = 10_000.0


class DatabaseManager:
    """Manage the SQLite database used by the strategy.

    Parameters
    ----------
    db_path : str | Path, optional
        Path to the SQLite database file.
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row

    def close(self) -> None:
        """Close the SQLite connection."""
        self.connection.close()

    def __enter__(self) -> "DatabaseManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def create_tables(self) -> None:
        """Create the prices and positions tables if they do not already exist."""
        cursor = self.connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS prices (
                theday TEXT NOT NULL,
                instrument TEXT NOT NULL,
                price REAL NOT NULL,
                PRIMARY KEY (theday, instrument)
            );
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS positions (
                time_of_trade TEXT NOT NULL,
                instrument TEXT NOT NULL,
                quantity REAL NOT NULL,
                cash REAL NOT NULL
            );
            """
        )

        self.connection.commit()

    def reset_tables(self) -> None:
        """Drop and recreate the prices and positions tables."""
        cursor = self.connection.cursor()
        cursor.execute("DROP TABLE IF EXISTS prices;")
        cursor.execute("DROP TABLE IF EXISTS positions;")
        self.connection.commit()
        self.create_tables()

    def load_prices_from_csv(
        self,
        csv_path: str | Path = DEFAULT_CSV_PATH,
        instrument: str = DEFAULT_INSTRUMENT,
    ) -> int:
        """Load daily prices from CSV into the prices table.

        Parameters
        ----------
        csv_path : str | Path, optional
            Path to the CSV file containing historical prices.
        instrument : str, optional
            Instrument label stored in the database.

        Returns
        -------
        int
            Number of rows inserted.
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None:
                raise ValueError("CSV file must contain a header row")

            rows = []
            for row in reader:
                date_value = self._extract_date(row)
                price_value = self._extract_price(row)
                rows.append((date_value, instrument, price_value))

        cursor = self.connection.cursor()
        cursor.executemany(
            """
            INSERT OR REPLACE INTO prices (theday, instrument, price)
            VALUES (?, ?, ?);
            """,
            rows,
        )
        self.connection.commit()
        return len(rows)

    def initialise_positions(
        self,
        initial_cash: float = DEFAULT_INITIAL_CASH,
        instrument: str = DEFAULT_INSTRUMENT,
        initial_date: str = "1900-01-01",
    ) -> None:
        """Initialise the positions table with a flat starting position."""
        if initial_cash < 0:
            raise ValueError("initial_cash must be non-negative")

        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM positions;")
        cursor.execute(
            """
            INSERT INTO positions (time_of_trade, instrument, quantity, cash)
            VALUES (?, ?, ?, ?);
            """,
            (initial_date, instrument, 0.0, float(initial_cash)),
        )
        self.connection.commit()

    def prepare_database(
        self,
        csv_path: str | Path = DEFAULT_CSV_PATH,
        instrument: str = DEFAULT_INSTRUMENT,
        initial_cash: float = DEFAULT_INITIAL_CASH,
    ) -> int:
        """Fully prepare the database for a fresh backtest run.

        This drops any existing tables, recreates the schema, loads price data,
        and inserts the initial flat position.

        Returns
        -------
        int
            Number of price rows loaded.
        """
        self.reset_tables()
        row_count = self.load_prices_from_csv(csv_path=csv_path, instrument=instrument)
        self.initialise_positions(initial_cash=initial_cash, instrument=instrument)
        return row_count

    def get_all_trading_days(self, start_date: Optional[str] = None) -> List[str]:
        """Return all available trading days, optionally filtered from a start date."""
        cursor = self.connection.cursor()

        if start_date is None:
            cursor.execute(
                """
                SELECT theday
                FROM prices
                ORDER BY theday ASC;
                """
            )
        else:
            cursor.execute(
                """
                SELECT theday
                FROM prices
                WHERE theday >= ?
                ORDER BY theday ASC;
                """,
                (start_date,),
            )

        return [row["theday"] for row in cursor.fetchall()]

    def get_prices_up_to_day(
        self,
        which_day: str,
        lookback: int,
        instrument: str = DEFAULT_INSTRUMENT,
    ) -> List[float]:
        """Return up to `lookback` historical prices before a given day."""
        if lookback <= 0:
            raise ValueError("lookback must be positive")

        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT price
            FROM prices
            WHERE instrument = ?
              AND theday < ?
            ORDER BY theday DESC
            LIMIT ?;
            """,
            (instrument, which_day, lookback),
        )

        rows = cursor.fetchall()
        return [row["price"] for row in reversed(rows)]

    def get_latest_price(
        self,
        which_day: str,
        instrument: str = DEFAULT_INSTRUMENT,
    ) -> float:
        """Return the latest price available on or before a given day."""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT price
            FROM prices
            WHERE instrument = ?
              AND theday <= ?
            ORDER BY theday DESC
            LIMIT 1;
            """,
            (instrument, which_day),
        )
        row = cursor.fetchone()

        if row is None:
            raise ValueError(f"No price found on or before {which_day}")

        return float(row["price"])

    def get_latest_position(
        self,
        which_day: str,
        instrument: str = DEFAULT_INSTRUMENT,
    ) -> Tuple[float, float]:
        """Return the latest quantity and cash balance before a given day."""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            SELECT quantity, cash
            FROM positions
            WHERE instrument = ?
              AND time_of_trade < ?
            ORDER BY time_of_trade DESC
            LIMIT 1;
            """,
            (instrument, which_day),
        )
        row = cursor.fetchone()

        if row is None:
            raise ValueError(f"No position found before {which_day}")

        return float(row["quantity"]), float(row["cash"])

    def insert_position(
        self,
        time_of_trade: str,
        quantity: float,
        cash: float,
        instrument: str = DEFAULT_INSTRUMENT,
    ) -> None:
        """Insert a new portfolio state into the positions table."""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT INTO positions (time_of_trade, instrument, quantity, cash)
            VALUES (?, ?, ?, ?);
            """,
            (time_of_trade, instrument, float(quantity), float(cash)),
        )
        self.connection.commit()

    def count_rows(self, table_name: str) -> int:
        """Return the number of rows in a supported table."""
        if table_name not in {"prices", "positions"}:
            raise ValueError("table_name must be either 'prices' or 'positions'")

        cursor = self.connection.cursor()
        cursor.execute(f"SELECT COUNT(*) AS row_count FROM {table_name};")
        row = cursor.fetchone()
        return int(row["row_count"])

    @staticmethod
    def _extract_date(row: dict) -> str:
        """Extract the date field from a CSV row."""
        for key in ("Date", "date", "theday"):
            if key in row and row[key] not in (None, ""):
                return str(row[key]).strip()
        raise ValueError("CSV row is missing a recognised date column")

    @staticmethod
    def _extract_price(row: dict) -> float:
        """Extract the closing price field from a CSV row."""
        for key in ("Close", "close", "Adj Close", "AdjClose", "price"):
            if key in row and row[key] not in (None, ""):
                return float(row[key])
        raise ValueError("CSV row is missing a recognised price column")


def prepare_database(
    db_path: str | Path = DEFAULT_DB_PATH,
    csv_path: str | Path = DEFAULT_CSV_PATH,
    instrument: str = DEFAULT_INSTRUMENT,
    initial_cash: float = DEFAULT_INITIAL_CASH,
) -> int:
    """Convenience function to create and populate the project database."""
    with DatabaseManager(db_path=db_path) as db:
        return db.prepare_database(
            csv_path=csv_path,
            instrument=instrument,
            initial_cash=initial_cash,
        )
