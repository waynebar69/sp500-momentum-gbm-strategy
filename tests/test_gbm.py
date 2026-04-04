"""Unit tests for the GBM model.

These tests validate the core quantitative logic used in the momentum strategy:
- GBM simulation output shape
- bootstrap calibration behaviour
- forecast interval consistency
- expected shortfall positivity
- input validation
"""

from __future__ import annotations

import numpy as np
import pytest

from src.gbm import GBM


def test_simulate_returns_correct_shape() -> None:
    """simulate should return an array of shape (n_steps + 1, n_paths)."""
    model = GBM(mu=0.08, sigma=0.20, random_state=42)
    paths = model.simulate(n_steps=10, n_paths=5, dt=1 / 252, s0=100.0)

    assert paths.shape == (11, 5)
    assert np.allclose(paths[0], 100.0)
    assert np.all(paths > 0.0)


def test_calibrate_produces_finite_parameters() -> None:
    """Calibration should produce finite, non-negative GBM parameters."""
    rng = np.random.default_rng(123)
    log_returns = rng.normal(loc=0.0004, scale=0.01, size=252)
    prices = 100.0 * np.exp(np.cumsum(np.insert(log_returns, 0, 0.0)))

    model = GBM(n_bootstrap=2000, random_state=42)
    model.calibrate(prices, dt=1 / 252)

    mu, sigma = model.parameters()

    assert np.isfinite(mu)
    assert np.isfinite(sigma)
    assert sigma >= 0.0


def test_forecast_returns_ordered_confidence_interval() -> None:
    """Forecast interval should contain the expected price in a sensible range."""
    model = GBM(mu=0.10, sigma=0.20, random_state=42)
    result = model.forecast(latest=100.0, horizon=10 / 252, confidence=0.95)

    lower, upper = result["interval"]
    expected_price = result["expected_price"]

    assert lower > 0.0
    assert upper > lower
    assert lower <= expected_price <= upper


def test_expected_shortfall_is_positive() -> None:
    """Expected shortfall should be positive for a non-zero volatility process."""
    model = GBM(mu=0.08, sigma=0.25, random_state=42)
    es = model.expected_shortfall(horizon=10 / 252, alpha=0.95)

    assert np.isfinite(es)
    assert es > 0.0


@pytest.mark.parametrize(
    "prices",
    [
        np.array([100.0]),
        np.array([100.0, 0.0, 101.0]),
        np.array([100.0, -1.0, 101.0]),
    ],
)
def test_calibrate_rejects_invalid_price_series(prices: np.ndarray) -> None:
    """Calibration should reject invalid price trajectories."""
    model = GBM()

    with pytest.raises(ValueError):
        model.calibrate(prices, dt=1 / 252)


def test_forecast_rejects_invalid_confidence_level() -> None:
    """Forecast should reject invalid confidence values."""
    model = GBM(mu=0.05, sigma=0.15)

    with pytest.raises(ValueError):
        model.forecast(latest=100.0, horizon=10 / 252, confidence=1.5)


def test_expected_shortfall_rejects_invalid_alpha() -> None:
    """Expected shortfall should reject invalid alpha values."""
    model = GBM(mu=0.05, sigma=0.15)

    with pytest.raises(ValueError):
        model.expected_shortfall(horizon=10 / 252, alpha=0.0)


def test_zero_volatility_generates_deterministic_path() -> None:
    """With zero volatility, GBM paths should evolve deterministically."""
    model = GBM(mu=0.12, sigma=0.0, random_state=42)
    paths = model.simulate(n_steps=3, n_paths=2, dt=1 / 252, s0=100.0)

    expected = np.array(
        [
            [100.0, 100.0],
            [100.0 * np.exp(0.12 / 252), 100.0 * np.exp(0.12 / 252)],
            [100.0 * np.exp(2 * 0.12 / 252), 100.0 * np.exp(2 * 0.12 / 252)],
            [100.0 * np.exp(3 * 0.12 / 252), 100.0 * np.exp(3 * 0.12 / 252)],
        ]
    )

    assert np.allclose(paths, expected)
