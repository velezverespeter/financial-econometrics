"""
GARCH(1,1) Volatility Modeling & Forecasting
-----------------------------------------------
Fits a GARCH(1,1) model to daily equity index returns to capture volatility
clustering — the well-documented tendency for large price moves to be
followed by more large moves, and calm periods to persist. Extracts the
fitted conditional volatility series, compares it against simple rolling
realized volatility, and produces a forward volatility forecast.

Usage:
    python garch_model.py --ticker SPY
    python garch_model.py --ticker SPY --fallback

Data source:
    Live daily price history pulled via yfinance.

    If unreachable, the model falls back to a SYNTHETIC return series
    simulated from a GARCH(1,1) data-generating process using textbook-
    representative parameters for a broad equity index (see
    FALLBACK_GARCH_PARAMS below). This is standard practice for validating
    a GARCH implementation (simulate from known parameters, confirm the
    fitted model recovers them) but is explicitly NOT real historical data,
    and is labeled as such in all output.

Author: Peter Velez Vereš
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

TRADING_DAYS = 252
RANDOM_SEED = 42

# ── Fallback calibration (used only if live data is unreachable) ────────────
# Textbook-representative GARCH(1,1) parameters for a broad equity index,
# consistent with commonly cited academic estimates (high persistence,
# alpha + beta approx. 0.95-0.99 is typical for daily equity index returns).
# Calibrated here to target a long-run annualized volatility of ~16%,
# in line with SPY's typical historical volatility.
FALLBACK_GARCH_PARAMS = {
    "omega_annualized_target_vol": 0.16,
    "alpha": 0.09,   # ARCH term — reaction to recent shocks
    "beta": 0.89,    # GARCH term — persistence of past volatility
    "daily_mean_return": 0.0004,  # approx. 10% annualized
}


def fetch_live_returns(ticker: str, years: int = 5) -> pd.Series:
    """Pull live daily adjusted close prices via yfinance and return daily
    percentage returns (scaled to percent, as is standard for GARCH fitting).
    Raises on failure so the caller can fall back cleanly."""
    import yfinance as yf

    data = yf.download(ticker, period=f"{years}y", auto_adjust=True)["Close"]
    if data.empty:
        raise ValueError("Live price data empty.")
    returns = 100 * data.pct_change().dropna()
    returns.name = ticker
    return returns.squeeze()


def generate_fallback_returns(n_days: int = TRADING_DAYS * 5) -> pd.Series:
    """Simulate a synthetic return series from a GARCH(1,1) data-generating
    process using textbook-representative parameters. Reproducible via a
    fixed random seed. Returns are scaled to percent, matching the live path."""
    rng = np.random.default_rng(RANDOM_SEED)

    alpha = FALLBACK_GARCH_PARAMS["alpha"]
    beta = FALLBACK_GARCH_PARAMS["beta"]
    target_annual_vol = FALLBACK_GARCH_PARAMS["omega_annualized_target_vol"]
    daily_mean = FALLBACK_GARCH_PARAMS["daily_mean_return"]

    long_run_daily_var = (target_annual_vol / np.sqrt(TRADING_DAYS)) ** 2
    omega = long_run_daily_var * (1 - alpha - beta)

    sigma2 = np.zeros(n_days)
    eps = np.zeros(n_days)
    sigma2[0] = long_run_daily_var

    for t in range(1, n_days):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * rng.standard_normal()

    returns_pct = 100 * (daily_mean + eps)
    idx = pd.RangeIndex(n_days)
    return pd.Series(returns_pct, index=idx, name="SYNTHETIC")


def get_returns(ticker: str, use_fallback: bool = False) -> tuple:
    """Returns (returns_series, is_synthetic)."""
    if use_fallback:
        print("Using synthetic returns simulated from a GARCH(1,1) process with")
        print("textbook-representative parameters (dated 2026-07-19).")
        print("NOTE: this is NOT real historical price data — for demonstration only.\n")
        return generate_fallback_returns(), True
    try:
        returns = fetch_live_returns(ticker)
        print(f"Pulled live 5-year daily price history for {ticker} via yfinance.\n")
        return returns, False
    except Exception as e:
        print(f"Live data fetch failed ({e}).")
        print("Falling back to synthetic GARCH(1,1)-simulated returns.")
        print("NOTE: this is NOT real historical price data — for demonstration only.\n")
        return generate_fallback_returns(), True


def fit_garch(returns: pd.Series):
    """Fit a GARCH(1,1) model with a constant mean and normal innovations."""
    model = arch_model(returns, mean="Constant", vol="Garch", p=1, q=1, dist="normal")
    fitted = model.fit(disp="off")
    return fitted


def run_analysis(ticker: str, use_fallback: bool = False, forecast_horizon: int = 21):
    returns, is_synthetic = get_returns(ticker, use_fallback=use_fallback)
    fitted = fit_garch(returns)

    params = fitted.params
    omega, alpha, beta = params["omega"], params["alpha[1]"], params["beta[1]"]
    persistence = alpha + beta
    long_run_daily_vol = np.sqrt(omega / (1 - persistence)) if persistence < 1 else np.nan
    long_run_annual_vol = long_run_daily_vol * np.sqrt(TRADING_DAYS) / 100

    conditional_vol = fitted.conditional_volatility  # daily, in percent
    conditional_vol_annualized = conditional_vol * np.sqrt(TRADING_DAYS) / 100

    # Rolling realized volatility (21-day window) for comparison
    rolling_vol = returns.rolling(21).std() * np.sqrt(TRADING_DAYS) / 100

    # Forward forecast
    forecast = fitted.forecast(horizon=forecast_horizon, reindex=False)
    forecast_var = forecast.variance.values[-1]  # daily variance, percent^2
    forecast_vol_annualized = np.sqrt(forecast_var) * np.sqrt(TRADING_DAYS) / 100

    label = "SYNTHETIC" if is_synthetic else ticker
    print(f"{'='*70}")
    print(f"GARCH(1,1) MODEL — {label}")
    print(f"{'='*70}\n")

    print("Fitted Parameters")
    print(f"  Omega (constant):        {omega:.6f}")
    print(f"  Alpha (ARCH term):       {alpha:.4f}")
    print(f"  Beta (GARCH term):       {beta:.4f}")
    print(f"  Persistence (a+b):       {persistence:.4f}")
    print(f"  Long-run Annual Vol:     {long_run_annual_vol:.2%}\n")

    print(f"Current Conditional Volatility (annualized): {conditional_vol_annualized.iloc[-1]:.2%}")
    print(f"Current 21-Day Rolling Realized Vol (annualized): {rolling_vol.iloc[-1]:.2%}\n")

    print(f"Forward Volatility Forecast ({forecast_horizon} trading days)")
    print(f"  Day 1:   {forecast_vol_annualized[0]:.2%}")
    print(f"  Day {forecast_horizon//2}:  {forecast_vol_annualized[forecast_horizon//2 - 1]:.2%}")
    print(f"  Day {forecast_horizon}:  {forecast_vol_annualized[-1]:.2%}")
    print(f"{'='*70}\n")

    return {
        "ticker": ticker,
        "is_synthetic": is_synthetic,
        "returns": returns,
        "fitted": fitted,
        "conditional_vol_annualized": conditional_vol_annualized,
        "rolling_vol": rolling_vol,
        "forecast_vol_annualized": forecast_vol_annualized,
        "persistence": persistence,
        "long_run_annual_vol": long_run_annual_vol,
        "params": {"omega": omega, "alpha": alpha, "beta": beta},
    }


def plot_returns(result: dict, output_path: str = "outputs/returns_series.png"):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(result["returns"].values, color="#1672B0", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Daily Return (%)")
    label = "Synthetic GARCH(1,1) Series" if result["is_synthetic"] else result["ticker"]
    ax.set_title(f"Daily Returns — {label} (Volatility Clustering)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Chart saved to {output_path}")
    plt.close()


def plot_volatility(result: dict, output_path: str = "outputs/conditional_volatility.png"):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(result["conditional_vol_annualized"].values, color="#1672B0",
             linewidth=1.2, label="GARCH(1,1) Conditional Volatility")
    ax.plot(result["rolling_vol"].values, color="#D8A03E",
             linewidth=1.0, linestyle="--", alpha=0.8, label="21-Day Rolling Realized Volatility")
    ax.axhline(result["long_run_annual_vol"], color="#888888", linewidth=1,
               linestyle=":", label=f"Long-Run Volatility ({result['long_run_annual_vol']:.1%})")
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Annualized Volatility")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    label = "Synthetic GARCH(1,1) Series" if result["is_synthetic"] else result["ticker"]
    ax.set_title(f"Conditional vs. Realized Volatility — {label}")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Chart saved to {output_path}")
    plt.close()


def plot_forecast(result: dict, output_path: str = "outputs/volatility_forecast.png"):
    fig, ax = plt.subplots(figsize=(9, 5))
    horizon = len(result["forecast_vol_annualized"])
    days = np.arange(1, horizon + 1)

    recent_history = result["conditional_vol_annualized"].values[-60:]
    hist_days = np.arange(-len(recent_history) + 1, 1)

    ax.plot(hist_days, recent_history, color="#1672B0", linewidth=1.2, label="Historical Conditional Volatility")
    ax.plot(days, result["forecast_vol_annualized"], color="#C14444", linewidth=1.5,
             linestyle="--", marker="o", markersize=3, label=f"{horizon}-Day Forecast")
    ax.axvline(0, color="black", linewidth=0.7, alpha=0.5)

    ax.set_xlabel("Trading Days (0 = Forecast Origin)")
    ax.set_ylabel("Annualized Volatility")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    label = "Synthetic GARCH(1,1) Series" if result["is_synthetic"] else result["ticker"]
    ax.set_title(f"Volatility Forecast — {label}")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Chart saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit a GARCH(1,1) volatility model.")
    parser.add_argument("--ticker", type=str, default="SPY", help="Ticker to model")
    parser.add_argument(
        "--fallback", action="store_true",
        help="Use synthetic GARCH-simulated returns instead of a live API call",
    )
    args = parser.parse_args()

    result = run_analysis(args.ticker, use_fallback=args.fallback)
    plot_returns(result)
    plot_volatility(result)
    plot_forecast(result)
