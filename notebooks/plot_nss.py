
#!/usr/bin/env python3
"""
plot_nss.py

Plot all results from a completed `nss.py` / `nss_fixed.py` run.

This script NEVER runs SWAMP and NEVER runs inference. It only reads saved outputs from
OUT_DIR and creates plots under OUT_DIR/plots.

This version mirrors the defensive / verbose style of plot_mala.py, but it targets the
Nested Slice Sampling (NSS) output schema written by the fixed runner:
- config.json
- observations.npz
- posterior_samples.npz
- mcmc_extra_fields.npz                  (NSS diagnostics; name kept for compatibility)
- posterior_predictive_quantiles.npz     (optional)
- maps_truth_and_posterior_summary.npz   (optional)

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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Force a non-interactive backend for headless / HPC environments.
import matplotlib

matplotlib.use("Agg")  # must be set before importing pyplot
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG
# =============================================================================

OUT_DIR = Path(os.environ.get("SWAMP_PLOT_OUT_DIR", "swamp_jaxoplanet_retrieval_outputs"))
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

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
logger = logging.getLogger("swamp_plot_nss")


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
    """Return up to the last `n_lines` lines of a text file (best-effort)."""
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
    logger.info(f"=== plot_nss diagnostics start ({_utc_ts()}) ===")
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
            "OUT_DIR does not exist. This plot script only reads outputs; run nss.py first, "
            "or set SWAMP_PLOT_OUT_DIR to the directory that contains config.json/observations.npz."
        )


def log_array_stats(name: str, x: Any, *, max_quantile_elems: int = 2_000_000) -> None:
    """Log shape/dtype and basic finite/min/max stats."""
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

    logger.info(f"{name}: finite={n_fin}/{n} ({100.0 * n_fin / max(n, 1):.2f}%), nonfinite={n_bad}")
    if n_fin == 0:
        return

    v = flat[finite]
    logger.info(
        f"{name}: min={float(np.min(v)):.6g}, max={float(np.max(v)):.6g}, "
        f"mean={float(np.mean(v)):.6g}, std={float(np.std(v)):.6g}"
    )

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
    _require_file(path, hint="nss.py should write this")
    try:
        obj = json.loads(path.read_text())
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON at {path}: {e}") from e
    if not isinstance(obj, dict):
        raise TypeError(f"Expected JSON object (dict) in {path}, got {type(obj)}")
    return obj


def load_npz_required(
    path: Path,
    *,
    required_keys: Sequence[str],
    allow_pickle: bool = False,
) -> np.lib.npyio.NpzFile:
    _require_file(path, hint="nss.py should write this")
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


def _scalar_value(x: Any) -> Any:
    arr = np.asarray(x)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(()).item()
    return arr


def _npz_scalar(npz: np.lib.npyio.NpzFile, key: str) -> Optional[Any]:
    if key not in npz.files:
        return None
    try:
        return _scalar_value(npz[key])
    except Exception as e:
        logger.warning(f"Could not extract scalar {key!r} from NPZ: {e}")
        return None


def _npz_float(npz: np.lib.npyio.NpzFile, key: str) -> Optional[float]:
    value = _npz_scalar(npz, key)
    if value is None:
        return None
    try:
        return float(value)
    except Exception as e:
        logger.warning(f"Could not convert NPZ key {key!r} to float: {e}")
        return None


def _npz_int(npz: np.lib.npyio.NpzFile, key: str) -> Optional[int]:
    value = _npz_scalar(npz, key)
    if value is None:
        return None
    try:
        return int(value)
    except Exception as e:
        logger.warning(f"Could not convert NPZ key {key!r} to int: {e}")
        return None


def _npz_str(npz: np.lib.npyio.NpzFile, key: str) -> Optional[str]:
    value = _npz_scalar(npz, key)
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return str(value)
    return str(value)


def _safe_float_array(x: Any, *, name: str) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(x, dtype=float)
    except Exception as e:
        logger.warning(f"Could not coerce {name} to float array: {e}")
        return None
    return arr


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
                "config.json out_dir does not match OUT_DIR used by plot_nss.py. "
                f"config out_dir={cfg_out}, plot OUT_DIR={out_res}. "
                "If you changed cfg.out_dir in nss.py, set SWAMP_PLOT_OUT_DIR accordingly."
            )
        else:
            logger.info("config.json out_dir matches plot OUT_DIR.")
    except Exception as e:
        logger.warning(f"Could not interpret config.json out_dir={cfg_out_dir!r} as a path: {e}")

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

extra_path = OUT_DIR / "mcmc_extra_fields.npz"
extra = load_npz_optional(extra_path)

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

        try:
            validate_1d_same_length("times_days", times_days, "ppc_p50", p50)
            validate_1d_same_length("times_days", times_days, "ppc_p05", p05)
            validate_1d_same_length("times_days", times_days, "ppc_p95", p95)
        except Exception:
            logger.exception("PPC arrays do not match observation times; ignoring PPC file.")
        else:
            if np.any(p05 > p50) or np.any(p50 > p95):
                logger.warning("PPC quantiles violate ordering (p05<=p50<=p95) at some times.")
            ppc_q = {"p05": p05, "p50": p50, "p95": p95}
            logger.info("PPC quantiles will be overlaid on phase curve plot.")

    q.close()

maps_path = OUT_DIR / "maps_truth_and_posterior_summary.npz"
maps = load_npz_optional(maps_path)
if maps is not None:
    for k in ("lon", "lat", "phi_truth", "T_truth", "I_truth", "phi_post", "T_post", "I_post"):
        if k not in maps.files:
            logger.warning(f"maps file missing key {k!r}; some plots may be skipped.")
    for k in ("phi_truth", "phi_post", "T_truth", "T_post", "I_truth", "I_post"):
        if k in maps.files:
            log_array_stats(f"maps.{k}", maps[k])

if extra is not None:
    method = _npz_str(extra, "inference_method")
    if method is not None:
        logger.info(f"Extra diagnostics inference_method={method!r}")
        if method.lower() != "nss":
            logger.warning(
                "mcmc_extra_fields.npz does not advertise inference_method='nss'. "
                "NSS diagnostics may be incomplete."
            )
    else:
        if "ns_logZ_mean" in extra.files:
            logger.info("Detected NSS diagnostics via ns_* keys.")
        else:
            logger.info("Could not infer diagnostics schema from mcmc_extra_fields.npz.")


# =============================================================================
# Helpers used by plotting
# =============================================================================


def flatten_chain_draw(x: np.ndarray) -> np.ndarray:
    """(chains, draws, ...) -> (chains*draws, ...)"""
    x = np.asarray(x)
    return x.reshape((-1,) + x.shape[2:])


def save_fig(fig: plt.Figure, filename: str) -> None:
    path = PLOTS_DIR / filename
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
    """Use a log axis only if the support is strictly positive and spans many orders."""
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

prior_types: List[str]
prior_lo: Optional[np.ndarray]
prior_hi: Optional[np.ndarray]
truth_vals: Optional[np.ndarray]

if cfg_param_names:
    if cfg_param_names != param_names:
        logger.warning(
            "Parameter name mismatch between config.json and posterior_samples.npz. "
            f"config.json names={cfg_param_names}, posterior names={param_names}. "
            "Proceeding using posterior_samples.npz names."
        )
        prior_types = ["uniform"] * len(param_names)
        prior_lo = None
        prior_hi = None
        truth_vals = None
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

flat_all = flatten_chain_draw(samples)
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
    logger.info("Plotting per-parameter 1D posteriors")
    flat = flatten_chain_draw(samples)
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

        use_log = should_use_log_axis(v, orders_threshold=orders_threshold, explicit_bounds=(lo, hi))
        logger.info(f"{name}: x-range from {source}: lo={lo:.6g}, hi={hi:.6g}, use_log={use_log}")

        fig, ax = plt.subplots(figsize=(7.0, 4.0))

        if use_log:
            if lo <= 0.0:
                logger.warning(f"{name}: requested log bins but lo<=0 (lo={lo}); falling back to linear bins.")
                use_log = False
            else:
                bins: Any = np.logspace(np.log10(lo), np.log10(hi), 45)
        if not use_log:
            bins = 45

        ax.hist(v, bins=bins, density=True, alpha=0.75, label="posterior")

        if prior_lo is not None and prior_hi is not None and j < prior_lo.size:
            ptype = str(prior_types[j]).strip().lower() if j < len(prior_types) else "uniform"
            xx = np.linspace(lo, hi, 400)
            if use_log and lo > 0.0:
                xx = np.logspace(np.log10(lo), np.log10(hi), 400)

            if ptype == "uniform":
                pdf = np.ones_like(xx) / (hi - lo)
            elif ptype == "log10_uniform":
                if lo <= 0.0:
                    pdf = np.full_like(xx, np.nan)
                else:
                    pdf = 1.0 / (xx * np.log(hi / lo))
            else:
                logger.warning(f"{name}: unknown prior type {ptype!r}; not overlaying prior.")
                pdf = np.full_like(xx, np.nan)

            ax.plot(xx, pdf, lw=2, label=f"prior ({ptype})")

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
    logger.info("Plotting corner_posterior.png")
    flat = flatten_chain_draw(samples)
    if flat.ndim != 2:
        raise ValueError(f"Flattened samples must be 2D (N,D). Got shape={flat.shape}")
    n, d = flat.shape
    logger.info(f"Corner plot input: N={n}, D={d}")

    mask = np.isfinite(flat).all(axis=1)
    dropped = int(np.sum(~mask))
    if dropped:
        logger.warning(f"Dropping {dropped}/{n} non-finite posterior draws before corner plot.")
        flat = flat[mask]
        n = flat.shape[0]
    if n == 0:
        logger.error("No finite posterior draws; skipping corner plot.")
        return

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

    use_log_axis: List[bool] = []
    for j in range(d):
        v = flat[:, j]
        use_log_axis.append(should_use_log_axis(v, orders_threshold=orders_threshold, explicit_bounds=ranges[j]))
        logger.info(f"corner axis {param_names[j] if j < len(param_names) else j}: range={ranges[j]}, log={use_log_axis[-1]}")

    truths: Optional[List[float]] = None
    if truth_vals is not None and truth_vals.size >= d:
        truths = [float(x) for x in truth_vals[:d].tolist()]

    lines: List[str] = []
    for j in range(d):
        q16, q50, q84 = quantile_summary(flat[:, j])
        truth = None if truths is None else truths[j]
        name = param_names[j] if j < len(param_names) else f"param_{j}"
        lines.append(format_summary_line(name, truth, q16, q50, q84))
    summary_text = "\n".join(lines)

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

            xlo, xhi = ranges[j]
            ax.set_xlim(xlo, xhi)
            if i != j:
                ylo, yhi = ranges[i]
                ax.set_ylim(ylo, yhi)

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

            if i == d - 1:
                ax.set_xlabel(param_labels[j] if j < len(param_labels) else str(j), fontsize=9)
            else:
                ax.set_xticklabels([])

            if j == 0 and i != 0:
                ax.set_ylabel(param_labels[i] if i < len(param_labels) else str(i), fontsize=9)
            else:
                ax.set_yticklabels([])

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


def plot_nss_diagnostics() -> None:
    logger.info("Plotting NSS diagnostics (if present)")
    if extra is None:
        logger.info("No mcmc_extra_fields.npz; skipping NSS diagnostics plots.")
        return

    required_any = {"ns_logZ_mean", "ns_num_live", "ns_ess", "ns_runtime_seconds"}
    if not required_any.intersection(extra.files):
        logger.info("No NSS-specific ns_* keys present; skipping NSS diagnostics.")
        return

    backend = _npz_str(extra, "ns_backend") or "unknown"
    bj_version = _npz_str(extra, "ns_blackjax_version") or "unknown"
    inner_req = _npz_str(extra, "ns_inner_kernel_requested") or "unknown"
    inner_used = _npz_str(extra, "ns_inner_kernel_used") or inner_req
    inner_step = _npz_float(extra, "ns_inner_step_size")
    inner_nleap = _npz_int(extra, "ns_inner_num_integration_steps")
    n_live = _npz_int(extra, "ns_num_live")
    n_delete = _npz_int(extra, "ns_num_delete")
    n_inner = _npz_int(extra, "ns_num_inner_steps")
    ns_steps = _npz_int(extra, "ns_steps")
    dead_points = _npz_int(extra, "ns_dead_points")
    logz_mean = _npz_float(extra, "ns_logZ_mean")
    logz_std = _npz_float(extra, "ns_logZ_std")
    ess_val = _npz_float(extra, "ns_ess")
    evals_val = _npz_int(extra, "ns_evals")
    runtime_val = _npz_float(extra, "ns_runtime_seconds")

    logger.info(
        "NSS summary: "
        f"backend={backend}, blackjax={bj_version}, n_live={n_live}, n_delete={n_delete}, "
        f"n_inner={n_inner}, inner_req={inner_req}, inner_used={inner_used}, "
        f"inner_step={inner_step}, inner_nleap={inner_nleap}, ns_steps={ns_steps}, dead_points={dead_points}, "
        f"logZ_mean={logz_mean}, logZ_std={logz_std}, ess={ess_val}, evals={evals_val}, runtime={runtime_val}"
    )

    if n_live is not None and n_live <= 0:
        logger.warning(f"ns_num_live={n_live} is invalid")
    if n_live is not None and n_delete is not None and (n_delete <= 0 or n_delete >= n_live):
        logger.warning(f"ns_num_delete={n_delete} is inconsistent with ns_num_live={n_live}")
    if n_live is not None and ess_val is not None:
        if ess_val <= 0:
            logger.warning(f"NSS ESS={ess_val:.6g} is non-positive")
        elif ess_val < 0.1 * float(n_live):
            logger.warning(
                f"NSS ESS={ess_val:.3f} is <10% of n_live={n_live}; posterior reweighting may be very noisy."
            )
    if runtime_val is not None and runtime_val <= 0:
        logger.warning(f"NSS runtime={runtime_val:.6g}s is non-positive")
    if logz_std is not None and logz_std > 5.0:
        logger.warning(
            f"NSS logZ bootstrap std={logz_std:.3f} is large; the evidence estimate may be very uncertain."
        )

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    # Summary text block
    ax = axs[0, 0]
    ax.axis("off")
    lines = [
        "Nested Sampling summary",
        f"backend: {backend}",
        f"blackjax: {bj_version}",
        f"inner kernel req: {inner_req}",
        f"inner kernel used: {inner_used}",
        f"inner step_size: {inner_step:.6g}" if inner_step is not None and math.isfinite(inner_step) else "inner step_size: n/a",
        (
            f"inner n_integration_steps: {inner_nleap}"
            if inner_nleap is not None and inner_nleap > 0
            else "inner n_integration_steps: n/a"
        ),
        f"n_live: {n_live if n_live is not None else 'n/a'}",
        f"num_delete: {n_delete if n_delete is not None else 'n/a'}",
        f"num_inner_steps: {n_inner if n_inner is not None else 'n/a'}",
        f"ns_steps: {ns_steps if ns_steps is not None else 'n/a'}",
        f"dead_points: {dead_points if dead_points is not None else 'n/a'}",
        f"evals: {evals_val if evals_val is not None else 'n/a'}",
        f"runtime [s]: {runtime_val:.3f}" if runtime_val is not None else "runtime [s]: n/a",
        f"log Z: {logz_mean:.6g} ± {logz_std:.3g}" if logz_mean is not None and logz_std is not None else f"log Z: {logz_mean}",
        f"ESS: {ess_val:.6g}" if ess_val is not None else "ESS: n/a",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        ha="left",
        va="top",
        family="monospace",
        fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.12),
    )

    # Settings chart
    ax = axs[0, 1]
    setting_names: List[str] = []
    setting_vals: List[float] = []
    for name, value in [
        ("n_live", n_live),
        ("n_delete", n_delete),
        ("n_inner", n_inner),
    ]:
        if value is not None and value > 0:
            setting_names.append(name)
            setting_vals.append(float(value))
    if setting_vals:
        ax.bar(setting_names, setting_vals)
        ax.set_ylabel("count")
        ax.set_title("NSS settings")
        if should_use_log_axis(np.asarray(setting_vals), orders_threshold=2.0):
            ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "settings not saved", ha="center", va="center")
        ax.axis("off")

    # Work chart
    ax = axs[1, 0]
    work_names: List[str] = []
    work_vals: List[float] = []
    for name, value in [
        ("steps", ns_steps),
        ("dead", dead_points),
        ("evals", evals_val),
    ]:
        if value is not None and value > 0:
            work_names.append(name)
            work_vals.append(float(value))
    if work_vals:
        ax.bar(work_names, work_vals)
        ax.set_ylabel("count")
        ax.set_title("NSS workload")
        if should_use_log_axis(np.asarray(work_vals), orders_threshold=2.0):
            ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "workload not saved", ha="center", va="center")
        ax.axis("off")

    # Evidence / ESS panel
    ax = axs[1, 1]
    if logz_mean is not None:
        yerr = logz_std if (logz_std is not None and math.isfinite(logz_std) and logz_std >= 0.0) else 0.0
        ax.errorbar([0.0], [logz_mean], yerr=[yerr], fmt="o", capsize=5)
        ax.set_xlim(-0.75, 0.75)
        ax.set_xticks([0.0])
        ax.set_xticklabels(["log Z"])
        ax.set_ylabel("evidence estimate")
        ax.set_title("Evidence bootstrap summary")
        note_lines: List[str] = []
        if ess_val is not None:
            note_lines.append(f"ESS={ess_val:.3g}")
            if n_live is not None and n_live > 0:
                note_lines.append(f"ESS / n_live={ess_val / float(n_live):.3g}")
        if runtime_val is not None:
            note_lines.append(f"runtime={runtime_val:.3g}s")
        if note_lines:
            ax.text(
                0.98,
                0.02,
                "\n".join(note_lines),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                family="monospace",
                fontsize=9,
                bbox=dict(boxstyle="round", alpha=0.12),
            )
    else:
        ax.text(0.5, 0.5, "log Z not saved", ha="center", va="center")
        ax.axis("off")

    save_fig(fig, "nss_diagnostics.png")


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

    def _pcolormesh(ax: plt.Axes, lon_rad: np.ndarray, lat_rad: np.ndarray, z: np.ndarray, title: str) -> None:
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
    """Render visible disk images from saved Ylm coefficients (truth + posterior median)."""
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
    render_phases = [float(x) for x in cfg.get("render_phases", [0.0, 0.25, 0.49, 0.51, 0.75])]

    lm_list: List[Tuple[int, int]] = [(ell, m) for ell in range(ydeg + 1) for m in range(-ell, ell + 1)]

    def ylm_from_dense(y_dense: np.ndarray) -> Any:
        y = jnp.asarray(y_dense)
        data = {lm: y[i] for i, lm in enumerate(lm_list)}
        return Ylm(data)

    def make_surface(y_dense: np.ndarray) -> Any:
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

    def safe_render(surface: Any, phase: float, res: int) -> np.ndarray:
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
# RUN
# =============================================================================


def _run_step(name: str, fn: Any) -> Optional[str]:
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
    ("nss_diagnostics", plot_nss_diagnostics),
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
logger.info(f"=== plot_nss diagnostics end ({_utc_ts()}) ===")
