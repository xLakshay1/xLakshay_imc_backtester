from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import pandas as pd


PRODUCT = "INTARIAN_PEPPER_ROOT"
DATASETS = {
    "R1": Path("/Users/lakshaykumar/Downloads/ROUND1"),
    "R2": Path("/Users/lakshaykumar/Downloads/ROUND2"),
}


@dataclass
class SessionSeries:
    dataset: str
    day: int
    path: Path
    frame: pd.DataFrame
    intercept: float
    max_abs_residual: float


def parse_day(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def two_sided_product_rows(path: Path, product: str) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=";")
    frame = frame[frame["product"] == product].copy()
    frame = frame[
        frame["bid_price_1"].notna()
        & frame["ask_price_1"].notna()
        & frame["mid_price"].notna()
        & (frame["mid_price"] > 0)
    ].copy()
    return frame.sort_values("timestamp").reset_index(drop=True)


def snap(value: float, grid: float) -> float:
    return round(value / grid) * grid


def detect_intercept(
    frame: pd.DataFrame,
    slope_per_timestamp: float,
    grid: float,
    warm_samples: int,
) -> float:
    sample = frame.head(warm_samples)
    raw_values = [
        float(row.mid_price) - slope_per_timestamp * float(row.timestamp)
        for row in sample.itertuples(index=False)
    ]
    return snap(median(raw_values), grid)


def load_sessions(
    slope_per_timestamp: float,
    intercept_grid: float,
    warm_samples: int,
) -> list[SessionSeries]:
    sessions: list[SessionSeries] = []

    for dataset, data_dir in DATASETS.items():
        for path in sorted(data_dir.glob("prices_round_*_day_*.csv")):
            frame = two_sided_product_rows(path, PRODUCT)
            if frame.empty:
                continue
            intercept = detect_intercept(
                frame,
                slope_per_timestamp=slope_per_timestamp,
                grid=intercept_grid,
                warm_samples=warm_samples,
            )
            residual = frame["mid_price"] - (
                intercept + slope_per_timestamp * frame["timestamp"]
            )
            sessions.append(
                SessionSeries(
                    dataset=dataset,
                    day=parse_day(path),
                    path=path,
                    frame=frame,
                    intercept=intercept,
                    max_abs_residual=float(residual.abs().max()),
                )
            )

    return sessions


def max_consecutive_deviations(
    frame: pd.DataFrame,
    intercept: float,
    slope_per_timestamp: float,
    warmup_timestamp: int,
    gap: float,
    injected_shift: float = 0.0,
    injection_timestamp: int | None = None,
) -> tuple[int, int | None]:
    max_run = 0
    run = 0
    first_fire_timestamp: int | None = None

    for row in frame.itertuples(index=False):
        timestamp = int(row.timestamp)
        if timestamp < warmup_timestamp:
            run = 0
            continue

        observed_mid = float(row.mid_price)
        if injection_timestamp is not None and timestamp >= injection_timestamp:
            observed_mid += injected_shift

        expected_mid = intercept + slope_per_timestamp * timestamp
        is_deviation = abs(observed_mid - expected_mid) >= gap
        if is_deviation:
            run += 1
            if first_fire_timestamp is None:
                first_fire_timestamp = timestamp
        else:
            run = 0
        max_run = max(max_run, run)

    return max_run, first_fire_timestamp


def select_threshold(
    sessions: list[SessionSeries],
    slope_per_timestamp: float,
    warmup_timestamp: int,
    gaps: list[float],
    confirms: list[int],
) -> tuple[float, int, dict[tuple[float, int], int]]:
    max_runs: dict[tuple[float, int], int] = {}
    for gap in gaps:
        normal_max_run = max(
            max_consecutive_deviations(
                session.frame,
                session.intercept,
                slope_per_timestamp,
                warmup_timestamp,
                gap,
            )[0]
            for session in sessions
        )
        for confirm in confirms:
            max_runs[(gap, confirm)] = normal_max_run
            if normal_max_run < confirm:
                return gap, confirm, max_runs

    gap = gaps[-1]
    confirm = confirms[-1]
    return gap, confirm, max_runs


def false_positive_rate_proxy(
    sessions: list[SessionSeries],
    slope_per_timestamp: float,
    warmup_timestamp: int,
    gap: float,
    confirm: int,
) -> tuple[int, int, float, float]:
    deviating_ticks = 0
    total_ticks = 0
    max_run = 0

    for session in sessions:
        session_max, _ = max_consecutive_deviations(
            session.frame,
            session.intercept,
            slope_per_timestamp,
            warmup_timestamp,
            gap,
        )
        max_run = max(max_run, session_max)
        for row in session.frame.itertuples(index=False):
            timestamp = int(row.timestamp)
            if timestamp < warmup_timestamp:
                continue
            total_ticks += 1
            expected = session.intercept + slope_per_timestamp * timestamp
            if abs(float(row.mid_price) - expected) >= gap:
                deviating_ticks += 1

    per_tick_rate = deviating_ticks / total_ticks if total_ticks else 0.0
    fp_per_million_ticks = (per_tick_rate ** confirm) * 1_000_000
    return deviating_ticks, max_run, per_tick_rate, fp_per_million_ticks


def print_intercepts(sessions: list[SessionSeries]) -> None:
    print("Intercept detection")
    print("SET DAY ROWS DETECTED_INTERCEPT MAX_ABS_RESIDUAL FILE")
    for session in sessions:
        print(
            f"{session.dataset:>2} {session.day:>3} {len(session.frame):>5} "
            f"{session.intercept:>18.1f} {session.max_abs_residual:>16.2f} "
            f"{session.path.name}"
        )
    print()


def print_synthetic_checks(
    sessions: list[SessionSeries],
    slope_per_timestamp: float,
    warmup_timestamp: int,
    gap: float,
    confirm: int,
    injection_timestamp: int,
) -> None:
    shifts = [
        -3 * gap,
        -2 * gap,
        -1.5 * gap,
        1.5 * gap,
        2 * gap,
        3 * gap,
    ]

    print("Synthetic regime-change detection")
    print("SHIFT WORST_FIRE_DELAY_TICKS MISSED_SESSIONS")
    for shift in shifts:
        worst_delay = 0
        missed = 0
        for session in sessions:
            run = 0
            fire_timestamp = None
            for row in session.frame.itertuples(index=False):
                timestamp = int(row.timestamp)
                if timestamp < warmup_timestamp:
                    run = 0
                    continue
                observed = float(row.mid_price)
                if timestamp >= injection_timestamp:
                    observed += shift
                expected = session.intercept + slope_per_timestamp * timestamp
                if abs(observed - expected) >= gap:
                    run += 1
                    if run >= confirm:
                        fire_timestamp = timestamp
                        break
                else:
                    run = 0

            if fire_timestamp is None:
                missed += 1
            else:
                delay_ticks = max(0, math.ceil((fire_timestamp - injection_timestamp) / 100))
                worst_delay = max(worst_delay, delay_ticks)

        print(f"{shift:>7.1f} {worst_delay:>22} {missed:>15}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate live intercept and strict regime-change checks for Pepper."
    )
    parser.add_argument("--slope-per-timestamp", type=float, default=0.001)
    parser.add_argument("--intercept-grid", type=float, default=1000.0)
    parser.add_argument("--warm-samples", type=int, default=25)
    parser.add_argument("--warmup-timestamp", type=int, default=50_000)
    parser.add_argument("--injection-timestamp", type=int, default=500_000)
    parser.add_argument(
        "--gaps",
        type=float,
        nargs="+",
        default=[10, 12, 15, 18, 20, 25, 30, 35, 40, 50],
    )
    parser.add_argument("--confirms", type=int, nargs="+", default=[3, 4, 5, 6, 8, 10])
    args = parser.parse_args()

    sessions = load_sessions(
        slope_per_timestamp=args.slope_per_timestamp,
        intercept_grid=args.intercept_grid,
        warm_samples=args.warm_samples,
    )
    if not sessions:
        raise RuntimeError("No Pepper sessions found")

    print_intercepts(sessions)

    gap, confirm, max_runs = select_threshold(
        sessions=sessions,
        slope_per_timestamp=args.slope_per_timestamp,
        warmup_timestamp=args.warmup_timestamp,
        gaps=args.gaps,
        confirms=args.confirms,
    )

    deviating_ticks, max_run, per_tick_rate, fp_per_million_ticks = false_positive_rate_proxy(
        sessions=sessions,
        slope_per_timestamp=args.slope_per_timestamp,
        warmup_timestamp=args.warmup_timestamp,
        gap=gap,
        confirm=confirm,
    )

    print("Selected strict detector")
    print(f"GAP: {gap}")
    print(f"CONFIRM: {confirm}")
    print(f"WARMUP_TIMESTAMP: {args.warmup_timestamp}")
    print(f"normal_deviating_ticks: {deviating_ticks}")
    print(f"normal_max_consecutive_run: {max_run}")
    print(f"normal_per_tick_deviation_rate: {per_tick_rate:.8f}")
    print(f"rough_false_fires_per_1m_ticks: {fp_per_million_ticks:.10f}")
    print()

    print_synthetic_checks(
        sessions=sessions,
        slope_per_timestamp=args.slope_per_timestamp,
        warmup_timestamp=args.warmup_timestamp,
        gap=gap,
        confirm=confirm,
        injection_timestamp=args.injection_timestamp,
    )

    print("Threshold scan sample")
    print("GAP CONFIRM NORMAL_MAX_RUN ZERO_FP")
    for (candidate_gap, candidate_confirm), normal_max_run in sorted(max_runs.items()):
        print(
            f"{candidate_gap:>4.0f} {candidate_confirm:>7} "
            f"{normal_max_run:>14} {str(normal_max_run < candidate_confirm):>7}"
        )


if __name__ == "__main__":
    main()
