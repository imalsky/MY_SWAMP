#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


def _load_export(path: Path) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    extra: Dict[str, str] = {"metadata.json": ""}
    ep = torch.export.load(path, extra_files=extra)
    meta: Dict[str, Any] = {}
    if extra.get("metadata.json"):
        meta = json.loads(extra["metadata.json"])
    return ep.module(), meta


def _load_first_test_shard(processed_dir: Path) -> Dict[str, np.ndarray]:
    shard_dir = processed_dir / "test"
    shards = sorted(shard_dir.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No test shards found under: {shard_dir}")
    with np.load(shards[0]) as f:
        return {
            "y_z": f["y_mat"].astype(np.float32),          # [N,T,S]
            "g_z": f["globals"].astype(np.float32),        # [N,G]
            "dt_norm": f["dt_norm_mat"].astype(np.float32) # [N,T-1]
        }


def _load_manifest(processed_dir: Path) -> Dict[str, Any]:
    p = processed_dir / "normalization.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing normalization.json: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _dt_norm_to_seconds(dt_norm: np.ndarray, manifest: Dict[str, Any]) -> np.ndarray:
    a = float(manifest["dt"]["log_min"])
    b = float(manifest["dt"]["log_max"])
    log_dt = dt_norm.astype(np.float64) * (b - a) + a
    return (10.0 ** log_dt).astype(np.float64)


def _denorm_species(y_z: np.ndarray, species_keys: List[str], manifest: Dict[str, Any]) -> np.ndarray:
    stats = manifest["per_key_stats"]
    mu = np.array([float(stats[k]["log_mean"]) for k in species_keys], dtype=np.float64)
    sd = np.array([float(stats[k]["log_std"]) for k in species_keys], dtype=np.float64)
    return (10.0 ** (y_z.astype(np.float64) * sd + mu)).astype(np.float64)


def _canonical_method(method: str) -> str:
    m = str(method).lower().strip().replace("_", "-")
    if m in ("", "none"):
        return "identity"
    if m in ("minmax", "min-max", "min_max"):
        return "min-max"
    if m in ("logminmax", "log-minmax", "log_min_max", "log-min-max"):
        return "log-min-max"
    if m in ("log10-standard", "log10_standard"):
        return "log-standard"
    return m


def _method_for(key: str, manifest: Dict[str, Any]) -> str:
    methods = manifest.get("methods") or manifest.get("normalization_methods") or {}
    default = manifest.get("default_method", "standard")
    return _canonical_method(methods.get(key, default))


def _denorm_globals(g_z: np.ndarray, gvars: List[str], manifest: Dict[str, Any]) -> np.ndarray:
    stats = manifest["per_key_stats"]
    out = np.zeros((len(gvars),), dtype=np.float64)

    for j, name in enumerate(gvars):
        m = _method_for(name, manifest)
        st = stats[name]
        z = float(g_z[j])

        if m == "identity":
            out[j] = z
        elif m == "standard":
            out[j] = z * float(st["std"]) + float(st["mean"])
        elif m == "min-max":
            out[j] = z * (float(st["max"]) - float(st["min"])) + float(st["min"])
        elif m == "log-min-max":
            out[j] = 10.0 ** (z * (float(st["log_max"]) - float(st["log_min"])) + float(st["log_min"]))
        else:
            raise ValueError(f"Unsupported global method '{m}' for key '{name}'.")

    return out


def _pick_device(s: str) -> str:
    s = (s or "").strip().lower()
    if s in ("", "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if s == "cuda" and not torch.cuda.is_available():
        return "cpu"
    if s == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return "cpu"
    return s


@torch.inference_mode()
def _rollout(
    model: torch.nn.Module,
    *,
    y0: np.ndarray,       # [S] physical
    g: np.ndarray,        # [G] physical
    dt_seq: np.ndarray,   # [n_steps] physical seconds
    device: str,
    dtype: torch.dtype,
) -> np.ndarray:
    y = torch.from_numpy(y0.astype(np.float32)).to(device=device, dtype=dtype).unsqueeze(0)  # [1,S]
    g_t = torch.from_numpy(g.astype(np.float32)).to(device=device, dtype=dtype).unsqueeze(0) # [1,G]

    ys: List[torch.Tensor] = []
    for dt in dt_seq:
        dt_t = torch.tensor([float(dt)], device=device, dtype=dtype)  # [1]
        y = model(y, dt_t, g_t)  # [1,S]
        ys.append(y[0])

    return torch.stack(ys, dim=0).cpu().numpy().astype(np.float64)


def _error_summary(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    eps = 1e-30
    yt = np.clip(y_true, eps, None)
    yp = np.clip(y_pred, eps, None)

    rel = np.abs(yp - yt) / (np.abs(yt) + 1e-12)
    mae = float(np.mean(np.abs(yp - yt)))
    log_mae = float(np.mean(np.abs(np.log10(yp) - np.log10(yt))))
    return (
        f"phys:  rel_err(mean)={rel.mean():.3e}, rel_err(max)={rel.max():.3e}, MAE={mae:.3e}\n"
        f"log10: mean |Δlog10|={log_mae:.3f} orders"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Autoregressive evaluation on processed test shard (exported PHYS model).")
    p.add_argument("--run-dir", type=Path, default=Path(__file__).resolve().parents[1] / "models" / "v1_done_1000_epochs")
    p.add_argument("--export", type=Path, default=None)
    p.add_argument("--processed-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "processed")
    p.add_argument("--sample-idx", type=int, default=0)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--dt-mode", choices=("per_step", "constant"), default="per_step")
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|mps")
    p.add_argument("--plot", action="store_true", help="If set, save a simple plot under <run_dir>/plots/.")
    p.add_argument("--plot-k", type=int, default=8, help="How many species to plot (largest max abundance).")
    args = p.parse_args()

    export_path = args.export
    if export_path is None:
        export_path = (args.run_dir / "export_cpu_dynB_1step_phys.pt2").resolve()
    export_path = export_path.expanduser().resolve()
    if not export_path.exists():
        raise FileNotFoundError(f"Export not found: {export_path}")

    processed_dir = args.processed_dir.expanduser().resolve()
    shard = _load_first_test_shard(processed_dir)
    manifest = _load_manifest(processed_dir)

    model, meta = _load_export(export_path)
    device = _pick_device(args.device)

    # Channel order must match between export metadata and the processed shards.
    sp_meta = meta.get("species_variables")
    gv_meta = meta.get("global_variables") or []
    sp = list(manifest["species_variables"])
    gv = list(manifest.get("global_variables") or manifest.get("meta", {}).get("global_variables") or [])

    if isinstance(sp_meta, list) and sp_meta and list(sp_meta) != sp:
        raise ValueError("species_variables mismatch between export metadata and normalization.json (order matters).")
    if isinstance(gv_meta, list) and list(gv_meta) != gv:
        raise ValueError("global_variables mismatch between export metadata and normalization.json (order matters).")

    # Move model.
    model = model.to(device)
    dtype = next(iter(model.buffers())).dtype if any(True for _ in model.buffers()) else torch.float32

    y_z = shard["y_z"][args.sample_idx]         # [T,S]
    g_z = shard["g_z"][args.sample_idx]         # [G]
    dt_norm = shard["dt_norm"][args.sample_idx] # [T-1]

    T = y_z.shape[0]
    if args.start_index < 0 or args.start_index >= T - 1:
        raise ValueError(f"--start-index out of range (got {args.start_index}, valid 0..{T-2}).")

    n_steps = int(min(args.n_steps, (T - 1) - args.start_index))
    if n_steps <= 0:
        raise ValueError("n_steps is zero after clipping to trajectory length.")

    y0_z = y_z[args.start_index]                             # [S]
    y_true_z = y_z[args.start_index + 1 : args.start_index + 1 + n_steps]  # [n_steps,S]

    y0 = _denorm_species(y0_z, sp, manifest)
    y_true = _denorm_species(y_true_z, sp, manifest)

    g = _denorm_globals(g_z, gv, manifest)

    dt_sec_all = _dt_norm_to_seconds(dt_norm, manifest)      # [T-1]
    if args.dt_mode == "constant":
        dt_seq = np.full((n_steps,), float(dt_sec_all[args.start_index]), dtype=np.float64)
    else:
        dt_seq = dt_sec_all[args.start_index : args.start_index + n_steps].astype(np.float64)

    y_pred = _rollout(model, y0=y0, g=g, dt_seq=dt_seq, device=device, dtype=dtype)

    print(f"export:   {export_path}")
    print(f"device:   {device}  dtype={str(dtype).replace('torch.', '')}")
    print(f"sample:   idx={args.sample_idx}  start={args.start_index}  steps={n_steps}  dt_mode={args.dt_mode}")
    print(_error_summary(y_true, y_pred))

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print("matplotlib not available; skipping plot")
            return

        t = np.cumsum(dt_seq)
        k = int(max(1, args.plot_k))
        order = np.argsort(y_true.max(axis=0))[::-1][:k]

        out_dir = (args.run_dir / "plots").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"eval_export_phys_{args.sample_idx}_start{args.start_index}_n{n_steps}_{args.dt_mode}.png"

        plt.figure(figsize=(8, 5))
        for j in order:
            plt.plot(t, y_true[:, j], linestyle="-", linewidth=2.0)
            plt.plot(t, y_pred[:, j], linestyle="None", marker="o", markersize=3.0, markerfacecolor="none")
        plt.yscale("log")
        plt.xlabel("time (s)")
        plt.ylabel("abundance")
        plt.title("Exported model rollout (truth line, pred markers)")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
