"""
Backtest engine for XAUUSD session-liquidity strategy.

Provides bar-by-bar trade simulation with no look-ahead bias,
comprehensive performance metrics, and equity curve computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import pandas as pd


@dataclass
class Trade:
    """Represents a single completed trade with full metadata."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    direction: str  # 'long' or 'short'
    stop_loss: float
    take_profit: float
    risk_r: float  # SL distance in price terms
    pnl_r: float  # P&L expressed as multiples of risk_r
    result: str  # 'win', 'loss', or 'breakeven'
    metadata: dict = field(default_factory=dict)


class BacktestEngine:
    """Bar-by-bar backtest engine for XAUUSD strategies.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with a DatetimeIndex and columns:
        open, high, low, close, volume.
    spread : float
        Spread in price points (default 0.30 for XAUUSD).
    commission_per_lot : float
        Round-trip commission per standard lot (default 7.0 USD).
    """

    MAX_TRADE_BARS: int = 48  # 2 days on 1H bars

    def __init__(
        self,
        df: pd.DataFrame,
        spread: float = 0.30,
        commission_per_lot: float = 7.0,
    ) -> None:
        self.df = df
        self.spread = spread
        self.commission_per_lot = commission_per_lot

        # Pre-extract numpy arrays for fast bar-by-bar simulation.
        self._timestamps: np.ndarray = df.index.values
        self._opens: np.ndarray = df["open"].values
        self._highs: np.ndarray = df["high"].values
        self._lows: np.ndarray = df["low"].values
        self._closes: np.ndarray = df["close"].values
        self._n_bars: int = len(df)

    @staticmethod
    def _to_utc(ts) -> pd.Timestamp:
        """Convert a numpy datetime64 or Timestamp to UTC-aware Timestamp."""
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            return t.tz_localize("UTC")
        return t

    # ------------------------------------------------------------------
    # Trade simulation
    # ------------------------------------------------------------------

    def simulate_trade(
        self,
        entry_idx: int,
        direction: str,
        stop_loss: float,
        take_profit: float,
    ) -> Trade:
        """Simulate a single trade bar-by-bar from *entry_idx* forward.

        The entry is assumed to happen at the **open** of the bar at
        *entry_idx*.  Spread is added/subtracted on entry depending on
        direction.

        For the bar at *entry_idx* itself we use it only as the entry
        price source; the SL/TP scan starts from the **next** bar
        (entry_idx + 1) to avoid using the entry bar's range as an
        exit signal.

        If both SL and TP could be hit on the same bar the engine
        conservatively assumes the SL was hit first.

        After *MAX_TRADE_BARS* bars without an exit the trade is closed
        at the market (close of the last bar in the window).

        Parameters
        ----------
        entry_idx : int
            Positional index into ``self.df`` where the trade opens.
        direction : str
            ``'long'`` or ``'short'``.
        stop_loss : float
            Stop-loss price level.
        take_profit : float
            Take-profit price level.

        Returns
        -------
        Trade
            The completed trade record.
        """
        if direction not in ("long", "short"):
            raise ValueError(f"direction must be 'long' or 'short', got '{direction}'")

        # --- Entry price with spread adjustment ---
        raw_entry = self._opens[entry_idx]
        if direction == "long":
            # Buy at the ask (open + half spread).
            entry_price = raw_entry + self.spread / 2.0
        else:
            # Sell at the bid (open - half spread).
            entry_price = raw_entry - self.spread / 2.0

        # Risk in price points (always positive).
        risk_r = abs(entry_price - stop_loss)
        if risk_r == 0:
            risk_r = 1e-9  # avoid division by zero

        entry_time = self._to_utc(self._timestamps[entry_idx])

        # --- Bar-by-bar scan ---
        end_idx = min(entry_idx + 1 + self.MAX_TRADE_BARS, self._n_bars)

        exit_price: float
        exit_time: pd.Timestamp
        hit: Optional[str] = None  # 'sl', 'tp', or None (timeout)

        for i in range(entry_idx + 1, end_idx):
            bar_high = self._highs[i]
            bar_low = self._lows[i]

            sl_hit: bool
            tp_hit: bool

            if direction == "long":
                sl_hit = bar_low <= stop_loss
                tp_hit = bar_high >= take_profit
            else:
                sl_hit = bar_high >= stop_loss
                tp_hit = bar_low <= take_profit

            if sl_hit and tp_hit:
                # Conservative: SL hit first.
                hit = "sl"
                exit_price = stop_loss
                exit_time = self._to_utc(self._timestamps[i])
                break
            elif sl_hit:
                hit = "sl"
                exit_price = stop_loss
                exit_time = self._to_utc(self._timestamps[i])
                break
            elif tp_hit:
                hit = "tp"
                exit_price = take_profit
                exit_time = self._to_utc(self._timestamps[i])
                break
        else:
            # Neither SL nor TP hit within MAX_TRADE_BARS — close at market.
            last_idx = end_idx - 1
            exit_price = self._closes[last_idx]
            exit_time = self._to_utc(self._timestamps[last_idx])

        # --- P&L in R multiples ---
        if direction == "long":
            raw_pnl = exit_price - entry_price
        else:
            raw_pnl = entry_price - exit_price

        pnl_r = raw_pnl / risk_r

        # --- Classify result ---
        if hit == "tp":
            result = "win"
        elif hit == "sl":
            result = "loss"
        else:
            # Timeout exit — classify by sign of pnl.
            if pnl_r > 0:
                result = "win"
            elif pnl_r < 0:
                result = "loss"
            else:
                result = "breakeven"

        return Trade(
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            direction=direction,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_r=risk_r,
            pnl_r=pnl_r,
            result=result,
        )

    # ------------------------------------------------------------------
    # Strategy runner
    # ------------------------------------------------------------------

    def run_strategy(
        self,
        strategy_func: Callable[[int, pd.DataFrame, "BacktestEngine"], Optional[dict]],
    ) -> List[Trade]:
        """Run a strategy function over the entire DataFrame.

        ``strategy_func`` is called for every bar with the signature::

            strategy_func(current_idx, df_up_to_current, engine) -> dict | None

        It must return ``None`` (no signal) or a dict with keys::

            {
                "direction": "long" | "short",
                "stop_loss": float,
                "take_profit": float,
                "metadata": dict,   # optional
            }

        While a trade is open no new signals are generated (flat-only
        entry).

        Parameters
        ----------
        strategy_func : callable
            Signal generator conforming to the interface above.

        Returns
        -------
        list[Trade]
            All completed trades in chronological order.
        """
        trades: List[Trade] = []
        current_trade_exit_idx: int = -1  # positional index of last exit bar

        # Pre-build a lightweight slice view to avoid creating a new
        # DataFrame on every bar.  The strategy's pre-computed data is
        # indexed by timestamp, so passing the full df with the current
        # index is sufficient.
        full_df = self.df

        for idx in range(self._n_bars):
            # Skip if we are still inside an open trade.
            if idx <= current_trade_exit_idx:
                continue

            # Pass the full DataFrame up to the current bar (inclusive).
            # Using iloc slice only when needed for correctness.
            df_slice = full_df.iloc[: idx + 1]

            signal = strategy_func(idx, df_slice, self)
            if signal is None:
                continue

            direction: str = signal["direction"]
            stop_loss: float = signal["stop_loss"]
            take_profit: float = signal["take_profit"]
            metadata: dict = signal.get("metadata", {})

            # Entry on the NEXT bar's open (bar[idx+1]).
            # The strategy evaluates bar[idx]'s completed OHLC, so we can
            # only enter after the bar closes — i.e. at bar[idx+1].open.
            entry_idx = idx + 1
            if entry_idx >= self._n_bars:
                continue  # no room to enter

            trade = self.simulate_trade(entry_idx, direction, stop_loss, take_profit)
            trade.metadata = metadata

            trades.append(trade)

            # Mark the exit bar so we skip until it's done.
            # Find the positional index of exit_time in the DataFrame.
            exit_loc = self.df.index.get_indexer(
                [trade.exit_time], method="nearest"
            )[0]
            current_trade_exit_idx = exit_loc

        return trades

    # ------------------------------------------------------------------
    # Performance metrics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_metrics(trades: List[Trade]) -> dict:
        """Compute comprehensive performance metrics from a list of trades.

        Parameters
        ----------
        trades : list[Trade]
            Completed trades (output of :meth:`run_strategy` or manual list).

        Returns
        -------
        dict
            Dictionary of performance statistics.
        """
        if not trades:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "avg_winner_r": 0.0,
                "avg_loser_r": 0.0,
                "expectancy_r": 0.0,
                "profit_factor": 0.0,
                "max_consecutive_wins": 0,
                "max_consecutive_losses": 0,
                "max_drawdown_r": 0.0,
                "sharpe_ratio": 0.0,
                "total_r": 0.0,
                "avg_trades_per_month": 0.0,
                "calmar_ratio": 0.0,
            }

        r_values = np.array([t.pnl_r for t in trades])
        results = [t.result for t in trades]

        total_trades = len(trades)
        wins = sum(1 for r in results if r == "win")
        losses = sum(1 for r in results if r == "loss")
        win_rate = (wins / total_trades) * 100.0 if total_trades else 0.0

        winner_rs = [t.pnl_r for t in trades if t.result == "win"]
        loser_rs = [t.pnl_r for t in trades if t.result == "loss"]

        avg_winner_r = float(np.mean(winner_rs)) if winner_rs else 0.0
        avg_loser_r = float(np.mean(loser_rs)) if loser_rs else 0.0

        # Expectancy: win_rate * avg_winner - loss_rate * |avg_loser|
        wr = wins / total_trades if total_trades else 0.0
        lr = 1.0 - wr
        expectancy_r = wr * avg_winner_r - lr * abs(avg_loser_r)

        # Profit factor: gross_profit / gross_loss
        gross_profit = sum(r for r in r_values if r > 0)
        gross_loss = abs(sum(r for r in r_values if r < 0))
        profit_factor = (
            float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")
        )

        # Consecutive wins / losses
        max_consecutive_wins = _max_consecutive(results, "win")
        max_consecutive_losses = _max_consecutive(results, "loss")

        # Drawdown
        cumulative = np.cumsum(r_values)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_drawdown_r = float(np.max(drawdowns)) if len(drawdowns) else 0.0

        # Sharpe ratio on R returns (risk-free = 0)
        if len(r_values) > 1 and np.std(r_values, ddof=1) > 0:
            sharpe_ratio = float(
                np.mean(r_values) / np.std(r_values, ddof=1)
            )
        else:
            sharpe_ratio = 0.0

        total_r = float(np.sum(r_values))

        # Average trades per month
        first_time = trades[0].entry_time
        last_time = trades[-1].exit_time
        span_days = max((last_time - first_time).total_seconds() / 86400.0, 1.0)
        span_months = span_days / 30.44  # average days per month
        avg_trades_per_month = total_trades / span_months if span_months > 0 else 0.0

        # Calmar ratio
        calmar_ratio = (
            total_r / max_drawdown_r if max_drawdown_r > 0 else float("inf")
        )

        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 2),
            "avg_winner_r": round(avg_winner_r, 4),
            "avg_loser_r": round(avg_loser_r, 4),
            "expectancy_r": round(expectancy_r, 4),
            "profit_factor": round(profit_factor, 4),
            "max_consecutive_wins": max_consecutive_wins,
            "max_consecutive_losses": max_consecutive_losses,
            "max_drawdown_r": round(max_drawdown_r, 4),
            "sharpe_ratio": round(sharpe_ratio, 4),
            "total_r": round(total_r, 4),
            "avg_trades_per_month": round(avg_trades_per_month, 2),
            "calmar_ratio": round(calmar_ratio, 4),
        }


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def compute_equity_curve(trades: List[Trade]) -> pd.Series:
    """Return a cumulative R-curve indexed by trade exit time.

    Parameters
    ----------
    trades : list[Trade]
        Completed trades sorted chronologically.

    Returns
    -------
    pd.Series
        Cumulative sum of ``pnl_r`` indexed by each trade's
        ``exit_time``.  The series name is ``"cumulative_r"``.
    """
    if not trades:
        return pd.Series(dtype=float, name="cumulative_r")

    exit_times = [t.exit_time for t in trades]
    pnl_rs = [t.pnl_r for t in trades]

    cumulative = np.cumsum(pnl_rs)

    return pd.Series(data=cumulative, index=exit_times, name="cumulative_r")


def _max_consecutive(results: List[str], target: str) -> int:
    """Return the longest streak of *target* in *results*."""
    max_streak = 0
    current = 0
    for r in results:
        if r == target:
            current += 1
            if current > max_streak:
                max_streak = current
        else:
            current = 0
    return max_streak
