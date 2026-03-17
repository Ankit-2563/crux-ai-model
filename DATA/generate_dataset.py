"""
EV Battery Dataset Generator
==============================
Generates a 10M-row synthetic EV battery monitoring dataset
suitable for training a Decision Tree classifier.

Constraints (hard-enforced):
  - temperature : [0, 80)   °C   — strictly below 80
  - voltage     : [8.0, 13.0) V  — strictly below 13 V
  - current     : [0, 18]   A    — never exceeds 18 A

Label logic (mutually exclusive, priority-ordered):
  CRITICAL → temperature ≥ 75  OR  voltage < 8.5  OR  current >= 17.5
  ALERT    → temperature ≥ 70  OR  voltage < 9.5  OR  current > 15
  NORMAL   → temperature < 40  AND voltage ≥ 11.5 AND current ≤ 10

Author : <your-name>
Version: 4.2.0
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
SEED        = 42
N_ROWS      = 10_000_000
OUTPUT_PATH = Path("ev_battery_10M.csv")

# Hard physical limits (inclusive min, exclusive max where noted)
TEMP_MIN,    TEMP_MAX    =  0.0,  80.0   # °C  — strictly below 80
VOLTAGE_MIN, VOLTAGE_MAX =  8.0,  13.0   # V   — strictly below 13
CURRENT_MIN, CURRENT_MAX =  0.0,  18.0   # A   — max 18 A (inclusive)

# ── CRITICAL thresholds (most severe — immediate danger) ──
CRITICAL_TEMP_THRESH    = 75.0   # >= → CRITICAL
CRITICAL_VOLT_THRESH    =  8.5   # <  → CRITICAL
CRITICAL_CURR_THRESH    = 17.5   # >= → CRITICAL
#
# NOTE: previously 18.0, which equalled CURRENT_MAX.  Because
# rng.uniform's upper bound is exclusive and float32 almost never
# rounds up to exactly 18.0, the band [18.0, 18.0] was effectively
# empty and the tree had zero CRITICAL-via-current training examples.
# Lowered to 17.5 to give a real [17.5, 18.0] band (~2.7 % of rows).

# ── ALERT thresholds (warning zone — between normal and critical) ──
ALERT_TEMP_THRESH       = 70.0   # >= → ALERT
ALERT_VOLT_THRESH       =  9.5   # <  → ALERT
ALERT_CURR_THRESH       = 15.0   # >  → ALERT

# ── NORMAL thresholds (all three must be satisfied) ──
NORMAL_TEMP_THRESH      = 40.0   # <  → candidate for NORMAL
NORMAL_VOLT_THRESH      = 11.5   # >= → candidate for NORMAL
NORMAL_CURR_THRESH      = 10.0   # <= → candidate for NORMAL

CHUNK_SIZE  = 500_000            # rows processed per chunk (memory safety)

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def _assign_labels(temp: np.ndarray,
                   volt: np.ndarray,
                   curr: np.ndarray) -> np.ndarray:
    """
    Vectorised label assignment — avoids Python-level loops entirely.

    Priority (highest to lowest):
      1. CRITICAL  — immediate danger, overrides everything
      2. ALERT     — warning zone, overrides NORMAL
      3. NORMAL    — all conditions are nominal

    Thresholds:
      CRITICAL : temp >= 75   OR  volt < 8.5  OR  current >= 17.5
      ALERT    : temp >= 70   OR  volt < 9.5  OR  current >  15
      NORMAL   : temp <  40   AND volt >= 11.5 AND current <= 10
    """
    # Start everything as NORMAL
    labels = np.full(len(temp), "NORMAL", dtype=object)

    # NORMAL mask — all three conditions must hold
    normal_mask = (
        (temp < NORMAL_TEMP_THRESH) &
        (volt >= NORMAL_VOLT_THRESH) &
        (curr <= NORMAL_CURR_THRESH)
    )
    # Anything not satisfying NORMAL conditions becomes ALERT
    labels[~normal_mask] = "ALERT"

    # ALERT overwrites NORMAL where warning thresholds are breached
    alert_mask = (
        (temp >= ALERT_TEMP_THRESH) |
        (volt <  ALERT_VOLT_THRESH) |
        (curr >  ALERT_CURR_THRESH)
    )
    labels[alert_mask] = "ALERT"

    # CRITICAL overwrites everything where danger thresholds are breached
    critical_mask = (
        (temp >= CRITICAL_TEMP_THRESH) |
        (volt <  CRITICAL_VOLT_THRESH) |
        (curr >= CRITICAL_CURR_THRESH)
    )
    labels[critical_mask] = "CRITICAL"

    return labels


def _validate_constraints(df: pd.DataFrame) -> None:
    """Assert all physical constraints are satisfied; raise on violation."""
    violations = []

    if df["temperature"].max() >= TEMP_MAX:
        violations.append(f"temperature ≥ {TEMP_MAX} found")
    if df["temperature"].min() < TEMP_MIN:
        violations.append(f"temperature < {TEMP_MIN} found")

    if df["voltage"].max() >= VOLTAGE_MAX:
        violations.append(f"voltage ≥ {VOLTAGE_MAX} found")
    if df["voltage"].min() < VOLTAGE_MIN:
        violations.append(f"voltage < {VOLTAGE_MIN} found")

    if df["current"].max() > CURRENT_MAX:
        violations.append(f"current > {CURRENT_MAX} found")
    if df["current"].min() < CURRENT_MIN:
        violations.append(f"current < {CURRENT_MIN} found")

    power_check = np.allclose(
        df["power"].values,
        (df["voltage"] * df["current"]).values,
        rtol=1e-5,
    )
    if not power_check:
        violations.append("power ≠ voltage × current")

    if violations:
        raise ValueError("Constraint violations detected:\n  " +
                         "\n  ".join(violations))

    log.info("✅  All physical constraints verified.")


def _log_class_distribution(labels: np.ndarray) -> None:
    unique, counts = np.unique(labels, return_counts=True)
    log.info("Class distribution:")
    for cls, cnt in zip(unique, counts):
        log.info("  %-10s  %9d  (%.2f%%)", cls, cnt, 100 * cnt / len(labels))


# ──────────────────────────────────────────────
# Main generator
# ──────────────────────────────────────────────
def generate_dataset(
    n_rows: int      = N_ROWS,
    seed: int        = SEED,
    output: Path     = OUTPUT_PATH,
    chunk_size: int  = CHUNK_SIZE,
) -> pd.DataFrame:
    """
    Generate the EV battery dataset and write it to *output* as CSV.

    Returns the complete DataFrame (held in memory).  For very large
    datasets you may want to drop the return value and rely on the CSV.
    """
    log.info("Generating %s rows  |  seed=%d  |  output=%s",
             f"{n_rows:,}", seed, output)

    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()

    # ── Sample features ──
    # Strategy:
    #   1. Sample in float64  (rng.uniform upper-bound is exclusive)
    #   2. Cast to float32
    #   3. Clip to hard limits — guards against float32 rounding at boundaries
    temperature = np.clip(
        rng.uniform(TEMP_MIN,    TEMP_MAX,    n_rows).astype(np.float32),
        np.float32(TEMP_MIN),
        np.nextafter(np.float32(TEMP_MAX),    -np.inf),   # strictly < 80
    )
    voltage = np.clip(
        rng.uniform(VOLTAGE_MIN, VOLTAGE_MAX, n_rows).astype(np.float32),
        np.float32(VOLTAGE_MIN),
        np.nextafter(np.float32(VOLTAGE_MAX), -np.inf),   # strictly < 13
    )
    current = np.clip(
        rng.uniform(CURRENT_MIN, CURRENT_MAX, n_rows).astype(np.float32),
        np.float32(CURRENT_MIN),
        np.float32(CURRENT_MAX),                           # inclusive 18
    )

    # Power derived AFTER clipping — always consistent with stored features
    # Power range: [0*8, 13*18) = [0, 234) W
    power = (voltage * current).astype(np.float32)

    # ── Vectorised label assignment ──
    battery_state = _assign_labels(temperature, voltage, current)

    # ── Assemble DataFrame ──
    df = pd.DataFrame({
        "temperature"  : temperature,
        "voltage"      : voltage,
        "current"      : current,
        "power"        : power,
        "battery_state": battery_state,
    })

    elapsed_gen = time.perf_counter() - t0
    log.info("Data generated in %.2f s", elapsed_gen)

    # ── Validate ──
    _validate_constraints(df)
    _log_class_distribution(battery_state)

    # ── Write CSV in chunks ──
    log.info("Writing CSV to %s …", output)
    t1 = time.perf_counter()
    for i, start in enumerate(range(0, n_rows, chunk_size)):
        chunk = df.iloc[start: start + chunk_size]
        chunk.to_csv(
            output,
            index=False,
            mode="w" if i == 0 else "a",
            header=(i == 0),
        )
    elapsed_write = time.perf_counter() - t1
    log.info("CSV written in %.2f s  |  size=%.1f MB",
             elapsed_write, output.stat().st_size / 1e6)

    log.info("✅  Dataset ready: %s", output.resolve())
    return df


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    df = generate_dataset()

    print("\nSample (first 5 rows):")
    print(df.head().to_string(index=False))
    print("\nDtypes:")
    print(df.dtypes.to_string())
    print(f"\nShape: {df.shape}")
    print(f"\nFeature ranges:")
    print(f"  temperature : [{df['temperature'].min():.2f}, {df['temperature'].max():.2f}] °C")
    print(f"  voltage     : [{df['voltage'].min():.2f}, {df['voltage'].max():.2f}] V")
    print(f"  current     : [{df['current'].min():.2f}, {df['current'].max():.2f}] A")
    print(f"  power       : [{df['power'].min():.2f}, {df['power'].max():.2f}] W")
