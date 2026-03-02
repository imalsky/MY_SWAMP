#!/usr/bin/env python3
"""
plot_mala.py

Plot all results from a completed `run_mala.py` run.

This script NEVER runs SWAMP and NEVER runs inference. It only reads saved outputs from
OUT_DIR and creates plots under OUT_DIR/plots.

This version adds much more defensive validation + verbose, terminal-friendly logging to help
diagnose common failure modes:
- missing / mismatched files
- wrong array shapes or missing NPZ keys
- non-finite data (NaN/Inf) poisoning plots
- posterior samples with unexpected ranges / degeneracies
- SMC diagnostics indicating weight collapse or stalled tempering
- optional file load failures (PPC, maps, diagnostics)

No CLI args by design: edit OUT_DIR below if needed (or override via SWAMP_PLOT_OUT_DIR).
"""

from __future__ import annotations

import inspect
import json
import logging
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Force a non-interactive backend for headless / HPC environments.
import matplotlib

matplotlib.use("Agg")  # must be set before importing pyplot
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG
# =============================================================================

# You can override the output directory without editing the file:
#   SWAMP_PLOT_OUT_DIR=/path/to/outputs ./plot_mala.py
OUT_DIR = Path(os.environ.get("SWAMP_PLOT_OUT_DIR", "swamp_jaxoplanet_retrieval_outputs"))
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Logging verbosity can be overridden without editing the file:
#   SWAMP_PLOT_LOG_LEVEL=DEBUG ./plot_mala.py
_LOG_LEVEL_NAME = os.environ.get("SWAMP_PLOT_LOG_LEVEL", "INFO").upper()
_LOG_LEVEL = getattr(logging, _LOG_LEVEL_NAME, logging.INFO)


# =============================================================================
# LOGGING
# =============================================================================

log_path = OUT_DIR / "plot.log"
logging.basicConfig(
    level=_LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_path, mode="w")],
    force=True,
)
logger = logging.getLogger("swamp_plot")


# =============================================================================
# Diagnostics helpers
# =============================================================================


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _describe_path(p: Path) -> str:
    try:
        st = p.stat()
    except FileNotFoundError:
        return "(missing)"
    size_kb = st.st_size / 1024.0
    mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"size={size_kb:.1f} KiB, mtime={mtime}"


def _tail_text_lines(path: Path, *, n_lines: int = 30, max_bytes: int = 64_000) -> List[str]:
    """Return up to the last `n_lines` lines of a text file (best-effort).

    This is meant for debugging context (e.g., showing the tail of OUT_DIR/run.log) without
    slurping arbitrarily large log files into memory.
    """
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            if end <= 0:
                return []
            read_size = min(int(max_bytes), int(end))
            f.seek(end - read_size, os.SEEK_SET)
            chunk = f.read(read_size)
    except Exception as e:
        logger.debug(f"Could not read tail of {path}: {e}", exc_info=True)
        return []

    try:
        text = chunk.decode("utf-8", errors="replace")
    except Exception:
        text = chunk.decode(errors="replace")

    lines = text.splitlines()
    return lines[-int(n_lines) :]


def log_environment() -> None:
    logger.info(f"=== plot_mala diagnostics start ({_utc_ts()}) ===")
    logger.info(f"OUT_DIR={OUT_DIR.resolve()}")
    logger.info(f"PLOTS_DIR={PLOTS_DIR.resolve()}")
    logger.info(f"CWD={Path.cwd().resolve()}")
    logger.info(f"Python={sys.version.splitlines()[0]}")
    logger.info(f"Platform={platform.platform()}")
    logger.info(f"NumPy={np.__version__}")
    logger.info(f"Matplotlib={matplotlib.__version__}, backend={matplotlib.get_backend()}")
    logger.info(f"Log level={_LOG_LEVEL_NAME}")
    if OUT_DIR.exists():
        try:
            files = sorted([p.name for p in OUT_DIR.iterdir()])
            logger.info(f"OUT_DIR contains {len(files)} entries: {files}")
        except Exception as e:
            logger.warning(f"Could not list OUT_DIR entries: {e}")

        run_log = OUT_DIR / "run.log"
        if run_log.exists():
            logger.info(f"Found run.log: {_describe_path(run_log)}")
            tail = _tail_text_lines(run_log, n_lines=25)
            if tail:
                logger.info("run.log tail (last 25 lines):")
                for line in tail:
                    logger.info(f"run.log| {line}")
            else:
                logger.info("run.log tail: (empty or unreadable)")
        else:
            logger.info("run.log not found in OUT_DIR.")
    else:
        logger.error(
            "OUT_DIR does not exist. This plot script only reads outputs; run the inference script first, "
            "or set SWAMP_PLOT_OUT_DIR to the directory that contains config.json/observations.npz."
        )


def _finite_mask(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if not np.issubdtype(x.dtype, np.number):
        return np.ones(x.shape, dtype=bool)
    return np.isfinite(x)


def log_array_stats(name: str, x: Any, *, max_quantile_elems: int = 2_000_000) -> None:
    """Log shape/dtype and basic finite/min/max stats.

    For very large arrays, we may sub-sample for quantiles to avoid excessive cost.
    """
    try:
        arr = np.asarray(x)
    except Exception as e:
        logger.warning(f"{name}: could not convert to ndarray for stats ({e})")
        return

    logger.info(f"{name}: dtype={arr.dtype}, shape={arr.shape}")
    if not np.issubdtype(arr.dtype, np.number):
        return

    flat = arr.reshape(-1)
    finite = np.isfinite(flat)
    n = flat.size
    n_fin = int(finite.sum())
    n_bad = n - n_fin
    if n == 0:
        logger.warning(f"{name}: empty array")
        return

    logger.info(f"{name}: finite={n_fin}/{n} ({100.0 * n_fin / max(n,1):.2f}%), nonfinite={n_bad}")
    if n_fin == 0:
        return

    v = flat[finite]
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    mean = float(np.mean(v))
    std = float(np.std(v))
    logger.info(f"{name}: min={vmin:.6g}, max={vmax:.6g}, mean={mean:.6g}, std={std:.6g}")

    # Quantiles: potentially sub-sample if huge.
    if v.size > max_quantile_elems:
        rng = np.random.default_rng(0)
        idx = rng.choice(v.size, size=max_quantile_elems, replace=False)
        vq = v[idx]
        logger.debug(f"{name}: quantiles computed on a random subsample of {max_quantile_elems} values")
    else:
        vq = v

    try:
        q01, q05, q50, q95, q99 = np.quantile(vq, [0.01, 0.05, 0.5, 0.95, 0.99])
        logger.info(f"{name}: q01={q01:.6g}, q05={q05:.6g}, q50={q50:.6g}, q95={q95:.6g}, q99={q99:.6g}")
    except Exception as e:
        logger.debug(f"{name}: could not compute quantiles ({e})")


def _require_file(path: Path, *, hint: str) -> None:
    if not path.exists():
        msg = f"Missing required file: {path} ({hint})"
        logger.error(msg)
        raise FileNotFoundError(msg)
    logger.info(f"Found {path.name}: {_describe_path(path)}")


def load_json_required(path: Path) -> Dict[str, Any]:
    _require_file(path, hint="run_mala.py should write this")
    try:
        obj = json.loads(path.read_text())
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON at {path}: {e}") from e
    if not isinstance(obj, dict):
        raise TypeError(f"Expected JSON object (dict) in {path}, got {type(obj)}")
    return obj


def load_npz_required(path: Path, *, required_keys: Sequence[str], allow_pickle: bool = False) -> np.lib.npyio.NpzFile:
    _require_file(path, hint="run_mala.py should write this")
    try:
        npz = np.load(path, allow_pickle=allow_pickle)
    except Exception as e:
        raise RuntimeError(f"Failed to load NPZ at {path}: {e}") from e

    keys = list(npz.files)
    logger.info(f"Loaded {path.name}: keys={keys}")
    missing = [k for k in required_keys if k not in keys]
    if missing:
        npz.close()
        raise KeyError(f"{path.name} missing keys {missing}. Available keys={keys}")
    return npz


def load_npz_optional(path: Path, *, allow_pickle: bool = False) -> Optional[np.lib.npyio.NpzFile]:
    if not path.exists():
        logger.info(f"Optional file not present: {path.name}")
        return None
    try:
        npz = np.load(path, allow_pickle=allow_pickle)
    except Exception:
        logger.exception(f"Optional file exists but could not be loaded: {path}")
        return None
    logger.info(f"Loaded optional {path.name}: keys={list(npz.files)}")
    return npz


def validate_1d_same_length(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray) -> None:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"Length mismatch: {name_a} has {a.shape[0]} elems, {name_b} has {b.shape[0]} elems")


def check_monotonic_increasing(name: str, x: np.ndarray) -> None:
    x = np.asarray(x).reshape(-1)
    if x.size < 2:
        return
    if not np.all(np.isfinite(x)):
        logger.warning(f"{name}: contains non-finite values; cannot check monotonicity reliably")
        return
    if np.any(np.diff(x) < 0):
        logger.warning(f"{name}: NOT monotonic increasing (this can indicate corrupted time arrays)")
    else:
        logger.info(f"{name}: monotonic increasing")


# =============================================================================
# LOAD FILES (with aggressive diagnostics)
# =============================================================================

log_environment()

cfg_path = OUT_DIR / "config.json"
cfg: Dict[str, Any] = load_json_required(cfg_path)
logger.info(f"Loaded config.json with {len(cfg)} keys")

cfg_out_dir = cfg.get("out_dir", None)
if cfg_out_dir is not None:
    try:
        cfg_out = Path(str(cfg_out_dir)).expanduser().resolve()
        out_res = OUT_DIR.resolve()
        if cfg_out != out_res:
            logger.warning(
                "config.json out_dir does not match OUT_DIR used by plot_mala.py. "
                f"config out_dir={cfg_out}, plot OUT_DIR={out_res}. "
                "If you changed cfg.out_dir in run_mala.py, set SWAMP_PLOT_OUT_DIR accordingly."
            )
        else:
            logger.info("config.json out_dir matches plot OUT_DIR.")
    except Exception as e:
        logger.warning(f"Could not interpret config.json out_dir={cfg_out_dir!r} as a path: {e}")

# Apply DPI if present
if "fig_dpi" in cfg:
    try:
        plt.rcParams["figure.dpi"] = int(cfg["fig_dpi"])
        logger.info(f"Matplotlib figure.dpi set to {plt.rcParams['figure.dpi']}")
    except Exception:
        logger.exception("Failed to apply fig_dpi from config.json; continuing with matplotlib default.")

obs_path = OUT_DIR / "observations.npz"
obs = load_npz_required(
    obs_path,
    required_keys=("times_days", "flux_true", "flux_obs", "obs_sigma", "orbital_period_days"),
)
times_days = np.asarray(obs["times_days"])
flux_true = np.asarray(obs["flux_true"])
flux_obs = np.asarray(obs["flux_obs"])
obs_sigma = float(obs["obs_sigma"])
orbital_period_days = float(obs["orbital_period_days"])
obs.close()

log_array_stats("times_days", times_days)
log_array_stats("flux_true", flux_true)
log_array_stats("flux_obs", flux_obs)
logger.info(f"obs_sigma={obs_sigma:.6g}")
logger.info(f"orbital_period_days={orbital_period_days:.6g}")

if not (math.isfinite(obs_sigma) and obs_sigma > 0.0):
    raise ValueError(f"obs_sigma must be finite and > 0. Got {obs_sigma!r}")
if not (math.isfinite(orbital_period_days) and orbital_period_days > 0.0):
    raise ValueError(f"orbital_period_days must be finite and > 0. Got {orbital_period_days!r}")

validate_1d_same_length("times_days", times_days, "flux_true", flux_true)
validate_1d_same_length("times_days", times_days, "flux_obs", flux_obs)
check_monotonic_increasing("times_days", times_days)

samples_path = OUT_DIR / "posterior_samples.npz"
samps = load_npz_required(samples_path, required_keys=("param_names", "samples"), allow_pickle=True)
param_names = [str(x) for x in samps["param_names"].tolist()]
param_labels = [str(x) for x in samps["param_labels"].tolist()] if "param_labels" in samps.files else param_names
samples = np.asarray(samps["samples"])  # (chains, draws, dim)
samps.close()

logger.info(f"Inferred parameters from posterior_samples.npz: {param_names}")
if len(param_labels) != len(param_names):
    logger.warning(
        f"param_labels length ({len(param_labels)}) != param_names length ({len(param_names)}); using param_names."
    )
    param_labels = param_names

if samples.ndim != 3:
    raise ValueError(f"posterior_samples['samples'] must have shape (chains, draws, dim); got {samples.shape}")
if samples.shape[-1] != len(param_names):
    logger.warning(
        f"samples dim={samples.shape[-1]} but len(param_names)={len(param_names)}. "
        "This usually indicates a corrupted posterior_samples.npz."
    )
logger.info(f"Loaded posterior samples cube: shape={samples.shape} (chains, draws, dim)")

log_array_stats("samples", samples)

# Optional: SMC diagnostics
extra_path = OUT_DIR / "mcmc_extra_fields.npz"
extra = load_npz_optional(extra_path)

# Optional: posterior predictive quantiles
ppc_quant_path = OUT_DIR / "posterior_predictive_quantiles.npz"
ppc_q: Optional[Dict[str, np.ndarray]] = None
q = load_npz_optional(ppc_quant_path)
if q is not None:
    required = ("p05", "p50", "p95")
    missing = [k for k in required if k not in q.files]
    if missing:
        logger.warning(
            f"posterior_predictive_quantiles.npz missing keys {missing}; expected {required}. Ignoring PPC file."
        )
    else:
        p05 = np.asarray(q["p05"])
        p50 = np.asarray(q["p50"])
        p95 = np.asarray(q["p95"])
        log_array_stats("ppc_p05", p05)
        log_array_stats("ppc_p50", p50)
        log_array_stats("ppc_p95", p95)

        # Shape checks: must match times_days length.
        try:
            validate_1d_same_length("times_days", times_days, "ppc_p50", p50)
            validate_1d_same_length("times_days", times_days, "ppc_p05", p05)
            validate_1d_same_length("times_days", times_days, "ppc_p95", p95)
        except Exception:
            logger.exception("PPC arrays do not match observation times; ignoring PPC file.")
        else:
            # Quantile ordering check
            if np.any(p05 > p50) or np.any(p50 > p95):
                logger.warning("PPC quantiles violate ordering (p05<=p50<=p95) at some times.")
            ppc_q = {"p05": p05, "p50": p50, "p95": p95}
            logger.info("PPC quantiles will be overlaid on phase curve plot.")

    q.close()

# Optional: maps
maps_path = OUT_DIR / "maps_truth_and_posterior_summary.npz"
maps = load_npz_optional(maps_path)
if maps is not None:
    for k in ("lon", "lat", "phi_truth", "T_truth", "I_truth", "phi_post", "T_post", "I_post"):
        if k not in maps.files:
            logger.warning(f"maps file missing key {k!r}; some plots may be skipped.")
    # Log a subset of arrays (avoid spamming huge logs)
    if "phi_truth" in maps.files:
        log_array_stats("maps.phi_truth", maps["phi_truth"])
    if "phi_post" in maps.files:
        log_array_stats("maps.phi_post", maps["phi_post"])
    if "T_truth" in maps.files:
        log_array_stats("maps.T_truth", maps["T_truth"])
    if "T_post" in maps.files:
        log_array_stats("maps.T_post", maps["T_post"])
    if "I_truth" in maps.files:
        log_array_stats("maps.I_truth", maps["I_truth"])
    if "I_post" in maps.files:
        log_array_stats("maps.I_post", maps["I_post"])


# =============================================================================
# Helpers used by plotting
# =============================================================================


def flatten_chain_draw(x: np.ndarray) -> np.ndarray:
    """(chains, draws, ...) -> (chains*draws, ...)"""
    x = np.asarray(x)
    return x.reshape((-1,) + x.shape[2:])


def save_fig(fig: plt.Figure, filename: str) -> None:
    path = PLOTS_DIR / filename
    # tight_layout can fail for some figures; don't let it kill the whole script.
    try:
        fig.tight_layout()
    except Exception:
        logger.debug(f"tight_layout failed for {filename}; saving without tight_layout.", exc_info=True)
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved {path}")


def finite_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x).reshape(-1)
    return x[np.isfinite(x)]


def orders_of_magnitude_span(lo: float, hi: float) -> float:
    if lo <= 0.0 or hi <= 0.0:
        return 0.0
    return float(np.log10(hi) - np.log10(lo))


def should_use_log_axis(
    values: np.ndarray,
    *,
    orders_threshold: float,
    explicit_bounds: Optional[Tuple[float, float]] = None,
) -> bool:
    """Heuristic: use log axis only if range spans many orders and values are positive."""
    if explicit_bounds is not None:
        lo, hi = explicit_bounds
        if lo <= 0.0 or hi <= 0.0:
            return False
        return orders_of_magnitude_span(float(lo), float(hi)) >= float(orders_threshold)

    v = finite_1d(values)
    if v.size == 0:
        return False
    vmin = float(np.min(v))
    vmax = float(np.max(v))
    if vmin <= 0.0:
        return False
    return orders_of_magnitude_span(vmin, vmax) >= float(orders_threshold)


def quantile_summary(v: np.ndarray) -> Tuple[float, float, float]:
    v = finite_1d(v)
    if v.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    q16, q50, q84 = np.quantile(v, [0.16, 0.50, 0.84])
    return float(q16), float(q50), float(q84)


def format_summary_line(name: str, truth: Optional[float], q16: float, q50: float, q84: float) -> str:
    if not (math.isfinite(q16) and math.isfinite(q50) and math.isfinite(q84)):
        return f"{name}: (no finite posterior samples)"
    plus = q84 - q50
    minus = q50 - q16
    if truth is None or (not math.isfinite(truth)):
        return f"{name}: median={q50:.6g} (+{plus:.3g}/-{minus:.3g})"
    delta = q50 - truth
    return f"{name}: truth={truth:.6g} | median={q50:.6g} (+{plus:.3g}/-{minus:.3g}) | median-truth={delta:.3g}"


def get_param_meta_from_cfg(cfg_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Extract inferred-parameter metadata from config.json if present."""
    meta: Dict[str, Any] = {}
    for k in (
        "inferred_param_names",
        "inferred_param_labels",
        "inferred_param_prior_types",
        "inferred_param_prior_lo",
        "inferred_param_prior_hi",
        "inferred_param_truth",
        "log_axis_orders_threshold",
    ):
        if k in cfg_obj:
            meta[k] = cfg_obj[k]
    return meta


def _safe_float_array(x: Any, *, name: str) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(x, dtype=float)
    except Exception as e:
        logger.warning(f"Could not coerce {name} to float array: {e}")
        return None
    return arr


# =============================================================================
# Parameter metadata / consistency checks
# =============================================================================

param_meta = get_param_meta_from_cfg(cfg)

cfg_param_names = [str(x) for x in param_meta.get("inferred_param_names", [])]
cfg_param_labels = [str(x) for x in param_meta.get("inferred_param_labels", [])]
cfg_prior_types = [str(x) for x in param_meta.get("inferred_param_prior_types", [])]
cfg_prior_lo = _safe_float_array(param_meta.get("inferred_param_prior_lo", []), name="inferred_param_prior_lo")
cfg_prior_hi = _safe_float_array(param_meta.get("inferred_param_prior_hi", []), name="inferred_param_prior_hi")
cfg_truth = _safe_float_array(param_meta.get("inferred_param_truth", []), name="inferred_param_truth")

if cfg_param_names:
    if cfg_param_names != param_names:
        logger.warning(
            "Parameter name mismatch between config.json and posterior_samples.npz. "
            f"config.json names={cfg_param_names}, posterior names={param_names}. "
            "Proceeding using posterior_samples.npz names."
        )
    else:
        logger.info("Parameter names match between config.json and posterior_samples.npz.")
        if cfg_param_labels and len(cfg_param_labels) == len(param_labels):
            param_labels = cfg_param_labels
        prior_types = cfg_prior_types if len(cfg_prior_types) == len(param_names) else ["uniform"] * len(param_names)
        prior_lo = cfg_prior_lo if (cfg_prior_lo is not None and cfg_prior_lo.size == len(param_names)) else None
        prior_hi = cfg_prior_hi if (cfg_prior_hi is not None and cfg_prior_hi.size == len(param_names)) else None
        truth_vals = cfg_truth if (cfg_truth is not None and cfg_truth.size == len(param_names)) else None
else:
    prior_types = ["uniform"] * len(param_names)
    prior_lo = None
    prior_hi = None
    truth_vals = None

orders_threshold = float(cfg.get("log_axis_orders_threshold", 3.0))
logger.info(f"log_axis_orders_threshold={orders_threshold:.3g}")

# Log per-parameter posterior summary (very useful when plots fail)
flat_all = flatten_chain_draw(samples)  # (N, D)
if flat_all.ndim == 2 and flat_all.shape[1] >= 1:
    n_all, d_all = flat_all.shape
    logger.info(f"Posterior flat view: N={n_all}, D={d_all}")
    nonfinite_rows = int(np.sum(~np.isfinite(flat_all).all(axis=1)))
    if nonfinite_rows:
        logger.warning(f"Posterior contains {nonfinite_rows}/{n_all} rows with non-finite values.")
    for j, name in enumerate(param_names[:d_all]):
        v = flat_all[:, j]
        v_fin = v[np.isfinite(v)]
        if v_fin.size == 0:
            logger.warning(f"Posterior[{name}]: no finite samples.")
            continue
        q16, q50, q84 = np.quantile(v_fin, [0.16, 0.5, 0.84])
        msg = f"Posterior[{name}]: median={q50:.6g}, q16={q16:.6g}, q84={q84:.6g}, std={np.std(v_fin):.3g}"
        if truth_vals is not None and j < truth_vals.size and math.isfinite(float(truth_vals[j])):
            msg += f", truth={float(truth_vals[j]):.6g}, median-truth={float(q50 - truth_vals[j]):.3g}"
        logger.info(msg)

        # Prior bound sanity (if known)
        if prior_lo is not None and prior_hi is not None and j < prior_lo.size:
            lo = float(prior_lo[j])
            hi = float(prior_hi[j])
            out_lo = int(np.sum(v_fin < lo))
            out_hi = int(np.sum(v_fin > hi))
            if out_lo or out_hi:
                logger.warning(
                    f"Posterior[{name}] has samples outside prior bounds: lo<{lo:.6g} count={out_lo}, "
                    f"hi>{hi:.6g} count={out_hi}. This should not happen if the transform is correct."
                )
else:
    logger.warning("Posterior samples array is not 2D after flattening; skipping per-parameter summaries.")


# =============================================================================
# Plotting routines
# =============================================================================


def plot_phase_curve() -> None:
    logger.info("Plotting phase_curve.png")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(times_days, flux_obs, ".", ms=3, label="observed", alpha=0.65)
    ax.plot(times_days, flux_true, "-", lw=2, label="truth (noise-free)")

    if ppc_q is not None:
        ax.plot(times_days, ppc_q["p50"], "-", lw=2, label="posterior median")
        ax.fill_between(times_days, ppc_q["p05"], ppc_q["p95"], alpha=0.25, label="90% PPC band")

    # Mark transit and secondary eclipse (approx)
    t0 = float(cfg.get("time_transit_days", 0.0))
    ax.axvline(t0, ls="--", lw=1, alpha=0.6)
    ax.axvline(t0 + 0.5 * orbital_period_days, ls="--", lw=1, alpha=0.6)

    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Planet flux (relative)")
    ax.set_title("Thermal phase curve (SWAMP + starry)")
    ax.legend(loc="best", fontsize=9)
    save_fig(fig, "phase_curve.png")


def plot_phase_curve_residuals() -> None:
    logger.info("Plotting phase_curve_residuals.png")
    model = ppc_q["p50"] if ppc_q is not None else flux_true
    resid = flux_obs - model
    log_array_stats("phase_curve_residuals", resid)

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(times_days, resid, ".", ms=3, alpha=0.7)
    ax.axhline(0.0, lw=1)
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Residual (obs - model)")
    ax.set_title("Residuals")
    save_fig(fig, "phase_curve_residuals.png")


def plot_posterior_1d_and_overlay_priors() -> None:
    """For each inferred parameter: histogram posterior; overlay analytic prior density."""
    logger.info("Plotting per-parameter 1D posteriors")
    flat = flatten_chain_draw(samples)  # (N, D)
    if flat.ndim != 2:
        raise ValueError(f"Flattened samples must be 2D (N,D). Got shape={flat.shape}")

    n, d = flat.shape
    logger.info(f"Flattened samples: N={n}, D={d}")

    for j in range(d):
        name = param_names[j] if j < len(param_names) else f"param_{j}"
        label = param_labels[j] if j < len(param_labels) else name

        v = flat[:, j]
        v = v[np.isfinite(v)]
        if v.size == 0:
            logger.warning(f"No finite posterior samples for {name}; skipping 1D posterior plot.")
            continue

        # Plot range: prefer prior bounds if available; else robust quantiles.
        if prior_lo is not None and prior_hi is not None and j < prior_lo.size:
            lo = float(prior_lo[j])
            hi = float(prior_hi[j])
            source = "prior bounds"
        else:
            qlo, qhi = np.quantile(v, [0.001, 0.999])
            lo, hi = float(qlo), float(qhi)
            source = "posterior 0.1%-99.9% quantiles"

        if not (math.isfinite(lo) and math.isfinite(hi)) or lo == hi:
            logger.warning(f"{name}: degenerate/non-finite plotting bounds (lo={lo}, hi={hi}); expanding.")
            lo = float(np.min(v))
            hi = float(np.max(v))
            if lo == hi:
                lo -= 1.0
                hi += 1.0

        # Axis scaling choice: only use log if values span many orders and are positive.
        use_log = should_use_log_axis(v, orders_threshold=orders_threshold, explicit_bounds=(lo, hi))
        logger.info(f"{name}: x-range from {source}: lo={lo:.6g}, hi={hi:.6g}, use_log={use_log}")

        fig, ax = plt.subplots(figsize=(7.0, 4.0))

        # Histogram bins: use log-spaced bins if log axis, else linear.
        if use_log:
            if lo <= 0.0:
                logger.warning(f"{name}: requested log bins but lo<=0 (lo={lo}); falling back to linear bins.")
                use_log = False
            else:
                bins: Any = np.logspace(np.log10(lo), np.log10(hi), 45)
        if not use_log:
            bins = 45

        ax.hist(v, bins=bins, density=True, alpha=0.75, label="posterior")

        # Overlay prior density if we know it
        if prior_lo is not None and prior_hi is not None and j < prior_lo.size:
            ptype = str(prior_types[j]).strip().lower() if j < len(prior_types) else "uniform"
            xx = np.linspace(lo, hi, 400)
            if use_log and lo > 0.0:
                xx = np.logspace(np.log10(lo), np.log10(hi), 400)

            if ptype == "uniform":
                pdf = np.ones_like(xx) / (hi - lo)
            elif ptype == "log10_uniform":
                # Uniform in log10(x) => p(x) ∝ 1 / x
                if lo <= 0.0:
                    pdf = np.full_like(xx, np.nan)
                else:
                    pdf = 1.0 / (xx * np.log(hi / lo))
            else:
                logger.warning(f"{name}: unknown prior type {ptype!r}; not overlaying prior.")
                pdf = np.full_like(xx, np.nan)

            ax.plot(xx, pdf, lw=2, label=f"prior ({ptype})")

        # Truth line if present
        if truth_vals is not None and j < truth_vals.size:
            truth = float(truth_vals[j])
            if math.isfinite(truth):
                ax.axvline(truth, lw=2, alpha=0.9, label="truth")

        ax.set_xlim(lo, hi)
        ax.set_xlabel(label)
        ax.set_ylabel("PDF")
        ax.set_title(f"Posterior (1D): {name}")
        if use_log:
            ax.set_xscale("log")

        ax.legend(loc="best", fontsize=9)
        safe_name = name.replace("/", "_").replace(" ", "_")
        save_fig(fig, f"posterior_1d_{safe_name}.png")


def plot_corner_with_text() -> None:
    """Custom corner plot that supports per-parameter log axes + summary text box."""
    logger.info("Plotting corner_posterior.png")
    flat = flatten_chain_draw(samples)  # (N, D)
    if flat.ndim != 2:
        raise ValueError(f"Flattened samples must be 2D (N,D). Got shape={flat.shape}")
    n, d = flat.shape
    logger.info(f"Corner plot input: N={n}, D={d}")

    # Remove any rows with NaNs/Infs.
    mask = np.isfinite(flat).all(axis=1)
    dropped = int(np.sum(~mask))
    if dropped:
        logger.warning(f"Dropping {dropped}/{n} non-finite posterior draws before corner plot.")
        flat = flat[mask]
        n = flat.shape[0]
    if n == 0:
        logger.error("No finite posterior draws; skipping corner plot.")
        return

    # Ranges: use prior bounds if available; else robust quantiles.
    ranges: List[Tuple[float, float]] = []
    for j in range(d):
        v = flat[:, j]
        if prior_lo is not None and prior_hi is not None and j < prior_lo.size:
            lo, hi = float(prior_lo[j]), float(prior_hi[j])
        else:
            qlo, qhi = np.quantile(v, [0.001, 0.999])
            lo, hi = float(qlo), float(qhi)
        if not math.isfinite(lo) or not math.isfinite(hi) or lo == hi:
            lo, hi = float(np.min(v)), float(np.max(v))
            if lo == hi:
                lo -= 1.0
                hi += 1.0
        ranges.append((lo, hi))

    # Decide axis scaling per parameter.
    use_log_axis: List[bool] = []
    for j in range(d):
        v = flat[:, j]
        use_log_axis.append(should_use_log_axis(v, orders_threshold=orders_threshold, explicit_bounds=ranges[j]))
        logger.info(f"corner axis {param_names[j] if j < len(param_names) else j}: range={ranges[j]}, log={use_log_axis[-1]}")
    # Prepare truth values if available.
    truths: Optional[List[float]] = None
    if truth_vals is not None and truth_vals.size >= d:
        truths = [float(x) for x in truth_vals[:d].tolist()]

    # Build summary text (truth vs recovered, errors).
    lines: List[str] = []
    for j in range(d):
        q16, q50, q84 = quantile_summary(flat[:, j])
        truth = None if truths is None else truths[j]
        name = param_names[j] if j < len(param_names) else f"param_{j}"
        lines.append(format_summary_line(name, truth, q16, q50, q84))
    summary_text = "\n".join(lines)

    # Corner layout: D x D grid, hist on diagonal, scatter below diagonal.
    fig = plt.figure(figsize=(2.2 * d + 1.5, 2.2 * d + 1.5))
    gs = fig.add_gridspec(d, d, wspace=0.05, hspace=0.05)

    for i in range(d):
        for j in range(d):
            ax = fig.add_subplot(gs[i, j])

            if i < j:
                ax.axis("off")
                continue

            x = flat[:, j]
            y = flat[:, i] if i != j else None

            # Axis ranges
            xlo, xhi = ranges[j]
            ax.set_xlim(xlo, xhi)
            if i != j:
                ylo, yhi = ranges[i]
                ax.set_ylim(ylo, yhi)

            # Log scales if needed (no log-transform of data; just axis scaling)
            if use_log_axis[j]:
                ax.set_xscale("log")
            if i != j and use_log_axis[i]:
                ax.set_yscale("log")

            if i == j:
                v = x[np.isfinite(x)]
                if v.size == 0:
                    ax.text(0.5, 0.5, "no finite", ha="center", va="center")
                else:
                    if use_log_axis[j] and xlo > 0.0:
                        bins = np.logspace(np.log10(xlo), np.log10(xhi), 40)
                    else:
                        bins = 40
                    ax.hist(v, bins=bins, density=True, alpha=0.85)

                if truths is not None and j < len(truths) and math.isfinite(truths[j]):
                    ax.axvline(truths[j], lw=2, alpha=0.9)

            else:
                assert y is not None
                m = np.isfinite(x) & np.isfinite(y)
                ax.plot(x[m], y[m], ".", ms=1.5, alpha=0.25)

                if truths is not None and j < len(truths) and i < len(truths):
                    if math.isfinite(truths[j]) and math.isfinite(truths[i]):
                        ax.axvline(truths[j], lw=1.5, alpha=0.9)
                        ax.axhline(truths[i], lw=1.5, alpha=0.9)

            # Ticks/labels
            if i == d - 1:
                ax.set_xlabel(param_labels[j] if j < len(param_labels) else str(j), fontsize=9)
            else:
                ax.set_xticklabels([])

            if j == 0 and i != 0:
                ax.set_ylabel(param_labels[i] if i < len(param_labels) else str(i), fontsize=9)
            else:
                ax.set_yticklabels([])

    # Summary text box
    fig.text(
        0.99,
        0.99,
        summary_text,
        ha="right",
        va="top",
        fontsize=8.5,
        family="monospace",
        bbox=dict(boxstyle="round", alpha=0.15),
    )
    fig.suptitle("Posterior corner + summary (truth vs recovered)", y=1.01)

    path = PLOTS_DIR / "corner_posterior.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_smc_diagnostics() -> None:
    logger.info("Plotting SMC diagnostics (if present)")
    if extra is None:
        logger.info("No mcmc_extra_fields.npz; skipping SMC diagnostics plots.")
        return

    if "smc_betas" not in extra.files:
        logger.info("smc_betas not present; skipping SMC diagnostics.")
        return

    betas = np.asarray(extra["smc_betas"]).reshape(-1)
    ess = np.asarray(extra["smc_ess"]).reshape(-1) if "smc_ess" in extra.files else None
    acc = np.asarray(extra["smc_acceptance_rate"]).reshape(-1) if "smc_acceptance_rate" in extra.files else None

    # Different BlackJAX versions may use different key names; accept both.
    logz = None
    if "smc_logZ" in extra.files:
        logz = np.asarray(extra["smc_logZ"]).reshape(-1)
    elif "smc_logZ_increment" in extra.files:
        inc = np.asarray(extra["smc_logZ_increment"]).reshape(-1)
        logz = np.cumsum(np.nan_to_num(inc, nan=0.0))

    n_particles = int(extra["smc_num_particles"]) if "smc_num_particles" in extra.files else int(cfg.get("smc_num_particles", 0) or 0)
    kernel = str(cfg.get("smc_mcmc_kernel", extra["smc_kernel"][()] if "smc_kernel" in extra.files else "unknown"))

    log_array_stats("smc_betas", betas)
    if ess is not None:
        log_array_stats("smc_ess", ess)
    if acc is not None:
        log_array_stats("smc_acceptance_rate", acc)
    if logz is not None:
        log_array_stats("smc_logZ", logz)

    if betas.size >= 2 and np.any(np.diff(betas) < -1e-12):
        logger.warning("smc_betas is not monotonic increasing (unexpected for tempered SMC).")

    if betas.size >= 1:
        logger.info(f"SMC final beta={betas[-1]:.6g} (should be ~1.0 for completed inference)")
        if betas[-1] < 0.999:
            logger.warning(
                "Final beta is < 1.0. This indicates adaptive tempering did not reach the posterior. "
                "In that case, posterior_samples may not represent the true posterior."
            )

    if ess is not None and ess.size > 0 and np.isfinite(ess).any() and n_particles > 0:
        ess_frac_min = float(np.nanmin(ess) / float(n_particles))
        logger.info(f"SMC ESS min fraction={ess_frac_min:.3f} (lower means worse weight degeneracy)")
        if ess_frac_min < 0.1:
            logger.warning(
                "Severe weight degeneracy detected (ESS/N < 0.1). If posteriors look wrong, increase particles, "
                "increase mutation steps, or reduce target_ess_frac."
            )

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    # Beta schedule
    ax = axs[0, 0]
    ax.plot(np.arange(betas.size), betas, marker="o", ms=4, lw=1.5)
    ax.set_xlabel("tempering step")
    ax.set_ylabel("beta")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Adaptive tempering schedule")

    # ESS
    ax = axs[0, 1]
    if ess is not None and ess.size > 0 and np.isfinite(ess).any():
        ax.plot(np.arange(ess.size), ess, marker="o", ms=4, lw=1.5)
        ax.set_xlabel("tempering step")
        ax.set_ylabel("ESS")
        title = "ESS after reweighting"
        if n_particles > 0:
            title += f" (N={n_particles})"
        ax.set_title(title)
        if n_particles > 0:
            ax2 = ax.twinx()
            ax2.plot(np.arange(ess.size), ess / float(n_particles), marker=".", ms=6, lw=1.0)
            ax2.set_ylabel("ESS / N")
            ax2.set_ylim(0.0, 1.05)
    else:
        ax.text(0.5, 0.5, "ESS not saved", ha="center", va="center")
        ax.axis("off")

    # Acceptance
    ax = axs[1, 0]
    if acc is not None and np.isfinite(acc).any():
        ax.plot(np.arange(acc.size), acc, marker="o", ms=4, lw=1.5)
        ax.set_xlabel("tempering step")
        ax.set_ylabel("mean acceptance")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"Mutation acceptance ({kernel})")
    else:
        ax.text(0.5, 0.5, "Acceptance not saved", ha="center", va="center")
        ax.axis("off")

    # log Z
    ax = axs[1, 1]
    if logz is not None and np.isfinite(logz).any():
        ax.plot(np.arange(logz.size), logz, marker="o", ms=4, lw=1.5)
        ax.set_xlabel("tempering step")
        ax.set_ylabel("log Z (cumulative)")
        ax.set_title("SMC log normalizer (diagnostic)")
    else:
        ax.text(0.5, 0.5, "logZ not saved", ha="center", va="center")
        ax.axis("off")

    save_fig(fig, "smc_diagnostics.png")

    # Final weights: plot weights on a log-x axis ONLY if they span many orders.
    if "smc_final_weights" in extra.files:
        w = np.asarray(extra["smc_final_weights"]).reshape(-1)
        log_array_stats("smc_final_weights", w)
        w = w[np.isfinite(w) & (w > 0)]
        if w.size > 0:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(w, bins=60, alpha=0.85)
            ax.set_xlabel("final weight")
            ax.set_ylabel("count")
            ax.set_title("Final SMC importance weights")
            if should_use_log_axis(w, orders_threshold=orders_threshold):
                ax.set_xscale("log")
            save_fig(fig, "smc_final_weights_hist.png")
        else:
            logger.info("smc_final_weights present but empty/non-finite; skipping weight histogram.")


def plot_maps() -> None:
    logger.info("Plotting maps.png (if present)")
    if maps is None:
        logger.info("No maps file; skipping maps.png")
        return

    required_keys = {"lon", "lat", "phi_truth", "T_truth", "I_truth", "phi_post", "T_post", "I_post"}
    missing = sorted(required_keys - set(maps.files))
    if missing:
        logger.warning(f"maps file missing required keys {missing}; skipping maps.png")
        return

    lon = np.asarray(maps["lon"])
    lat = np.asarray(maps["lat"])

    def _edges_1d(x: np.ndarray, *, is_lat: bool) -> np.ndarray:
        x = np.asarray(x).reshape(-1)
        if x.size < 2:
            return np.array([x[0] - 0.5, x[0] + 0.5])
        edges = np.zeros(x.size + 1)
        edges[1:-1] = 0.5 * (x[:-1] + x[1:])
        edges[0] = x[0] - 0.5 * (x[1] - x[0])
        edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
        if is_lat:
            edges[0] = -0.5 * np.pi
            edges[-1] = 0.5 * np.pi
        return edges

    def _pcolormesh(ax, lon_rad: np.ndarray, lat_rad: np.ndarray, z: np.ndarray, title: str) -> None:
        lon_edges = _edges_1d(lon_rad, is_lat=False)
        lat_edges = _edges_1d(lat_rad, is_lat=True)
        lon_e, lat_e = np.meshgrid(lon_edges, lat_edges)
        pcm = ax.pcolormesh(np.degrees(lon_e), np.degrees(lat_e), z, shading="auto")
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")
        ax.set_title(title)
        ax.get_figure().colorbar(pcm, ax=ax, shrink=0.85)

    def intensity_title(base: str) -> str:
        mode = str(cfg.get("emission_model", "bolometric")).strip().lower()
        if mode == "bolometric":
            return f"{base} (I ∝ T^4)"
        if mode == "planck":
            lam_m = cfg.get("planck_wavelength_m", None)
            if lam_m is None:
                return f"{base} (I ∝ B_λ[T])"
            try:
                lam_um = 1e6 * float(lam_m)
                return f"{base} (I ∝ B_λ[T], λ={lam_um:.3g} µm)"
            except Exception:
                return f"{base} (I ∝ B_λ[T])"
        return f"{base} (I; emission_model={mode})"

    fig, axs = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    _pcolormesh(axs[0, 0], lon, lat, np.asarray(maps["phi_truth"]), "Phi truth")
    _pcolormesh(axs[0, 1], lon, lat, np.asarray(maps["T_truth"]), "T truth [K]")
    _pcolormesh(axs[0, 2], lon, lat, np.asarray(maps["I_truth"]), intensity_title("I truth"))
    _pcolormesh(axs[1, 0], lon, lat, np.asarray(maps["phi_post"]), "Phi posterior median")
    _pcolormesh(axs[1, 1], lon, lat, np.asarray(maps["T_post"]), "T posterior median [K]")
    _pcolormesh(axs[1, 2], lon, lat, np.asarray(maps["I_post"]), intensity_title("I posterior median"))
    fig.suptitle("Terminal SWAMP maps and intensity proxy")
    path = PLOTS_DIR / "maps.png"
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_disk_renders() -> None:
    """Render visible disk images from saved Ylm coefficients (truth + posterior median).

    Requires jax + jaxoplanet/starry; otherwise skipped.
    """
    logger.info("Plotting disk renders (if possible)")
    if maps is None:
        logger.info("No maps file; skipping disk renders.")
        return

    if "y_truth" not in maps.files or "y_post" not in maps.files:
        logger.info("maps file missing y_truth/y_post; skipping disk renders.")
        return

    try:
        import jax
        import jax.numpy as jnp
        from jaxoplanet.starry.surface import Surface
        from jaxoplanet.starry.ylm import Ylm
    except Exception as e:
        logger.info(f"jax/jaxoplanet not importable; skipping disk renders. Error: {e}")
        return

    ydeg = int(cfg.get("ydeg", 10))
    inc = float(cfg.get("map_inc_rad", math.pi / 2))
    obl = float(cfg.get("map_obl_rad", 0.0))
    phase0 = float(cfg.get("phase_at_transit_rad", math.pi))
    time_transit = float(cfg.get("time_transit_days", 0.0))
    render_res = int(cfg.get("render_res", 250))
    render_phases = cfg.get("render_phases", [0.0, 0.25, 0.49, 0.51, 0.75])
    render_phases = [float(x) for x in render_phases]

    lm_list: List[Tuple[int, int]] = [(ell, m) for ell in range(ydeg + 1) for m in range(-ell, ell + 1)]

    def ylm_from_dense(y_dense: np.ndarray) -> Ylm:
        y = jnp.asarray(y_dense)
        data = {lm: y[i] for i, lm in enumerate(lm_list)}
        return Ylm(data)

    def make_surface(y_dense: np.ndarray) -> Surface:
        return Surface(
            y=ylm_from_dense(y_dense),
            u=(),
            inc=jnp.asarray(inc),
            obl=jnp.asarray(obl),
            period=jnp.asarray(orbital_period_days),
            phase=jnp.asarray(phase0),
            amplitude=jnp.asarray(1.0),
            normalize=False,
        )

    def safe_render(surface: Surface, phase: float, res: int) -> np.ndarray:
        try:
            sig = inspect.signature(surface.render)
            if "theta" in sig.parameters:
                img = surface.render(theta=jnp.asarray(phase), res=res)
            elif "phase" in sig.parameters:
                img = surface.render(phase=jnp.asarray(phase), res=res)
            else:
                img = surface.render(res=res)
        except Exception:
            img = surface.render(res=res)
        return np.asarray(img)

    def render_grid(y_dense: np.ndarray, label: str, filename: str) -> None:
        surface = make_surface(y_dense)
        fig, axs = plt.subplots(1, len(render_phases), figsize=(3.2 * len(render_phases), 3.0), constrained_layout=True)
        if len(render_phases) == 1:
            axs = [axs]
        for ax, ph in zip(axs, render_phases):
            t = time_transit + ph * orbital_period_days
            theta = 2.0 * math.pi * (t - time_transit) / orbital_period_days + phase0
            img = safe_render(surface, theta, render_res)
            ax.imshow(img, origin="lower")
            ax.set_title(f"{label}\nphase={ph:.2f}")
            ax.axis("off")
        path = PLOTS_DIR / filename
        fig.savefig(path)
        plt.close(fig)
        logger.info(f"Saved {path}")

    render_grid(np.asarray(maps["y_truth"]), "Truth", "disk_renders_truth.png")
    render_grid(np.asarray(maps["y_post"]), "Posterior median", "disk_renders_posterior.png")


# =============================================================================
# RUN (with per-step exception logging)
# =============================================================================


def _run_step(name: str, fn) -> Optional[str]:
    logger.info(f"--- {name} ---")
    t0 = time.perf_counter()
    try:
        fn()
    except Exception:
        logger.exception(f"FAILED step: {name}")
        return name
    dt = time.perf_counter() - t0
    logger.info(f"Finished {name} in {dt:.2f} s")
    return None


logger.info("Generating plots...")

failures: List[str] = []
for step_name, step_fn in [
    ("phase_curve", plot_phase_curve),
    ("phase_curve_residuals", plot_phase_curve_residuals),
    ("posterior_1d", plot_posterior_1d_and_overlay_priors),
    ("corner", plot_corner_with_text),
    ("smc_diagnostics", plot_smc_diagnostics),
    ("maps", plot_maps),
    ("disk_renders", plot_disk_renders),
]:
    failed = _run_step(step_name, step_fn)
    if failed is not None:
        failures.append(failed)

if maps is not None:
    maps.close()
if extra is not None:
    extra.close()

if failures:
    logger.error(f"Plotting completed with failures: {failures}")
    logger.error(f"See {log_path} for the full traceback(s).")
    raise SystemExit(1)

logger.info(f"DONE. Plots saved to {PLOTS_DIR.resolve()}")
logger.info(f"Log written to {log_path.resolve()}")
logger.info(f"=== plot_mala diagnostics end ({_utc_ts()}) ===")
