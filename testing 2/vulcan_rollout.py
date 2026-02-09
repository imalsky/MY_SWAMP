#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
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


def _load_vulcan(path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    with path.open("rb") as f:
        d = pickle.load(f)
    t = np.asarray(d["variable"]["t_time"], dtype=np.float64)                 # [T]
    Y = np.asarray(d["variable"]["y_time"], dtype=np.float64)                 # [T, layer, S]
    names = list(d["variable"]["species"])
    MR = Y[:, 0, :] / np.maximum(Y[:, 0, :].sum(axis=-1, keepdims=True), 1e-30)
    return t, MR.astype(np.float64), names


def _bases_from_species_vars(species_vars: List[str]) -> List[str]:
    out: List[str] = []
    for k in species_vars:
        out.append(k[:-10] if k.endswith("_evolution") else k)
    return out


def _guess_global_value(name: str, *, t_k: float, p_barye: float) -> float:
    n = name.strip().lower()
    if n.startswith("p") or "press" in n:
        return p_barye
    if n.startswith("t") or "temp" in n:
        return t_k
    return 0.0


@torch.inference_mode()
def _rollout(
    model: torch.nn.Module,
    *,
    y0: np.ndarray,     # [S] physical
    g: np.ndarray,      # [G] physical
    dt_sec: float,
    n_steps: int,
    device: str,
    dtype: torch.dtype,
) -> np.ndarray:
    y = torch.from_numpy(y0.astype(np.float32)).to(device=device, dtype=dtype).unsqueeze(0)  # [1,S]
    g_t = torch.from_numpy(g.astype(np.float32)).to(device=device, dtype=dtype).unsqueeze(0) # [1,G]
    dt = torch.tensor([float(dt_sec)], device=device, dtype=dtype)                            # [1]

    ys: List[torch.Tensor] = []
    for _ in range(int(n_steps)):
        y = model(y, dt, g_t)  # [1,S]
        ys.append(y[0])

    return torch.stack(ys, dim=0).cpu().numpy().astype(np.float64)


def main() -> None:
    p = argparse.ArgumentParser(description="VULCAN -> exported PHYS model rollout (simple overlay plot).")
    p.add_argument("--run-dir", type=Path, default=Path(__file__).resolve().parents[1] / "models" / "v1_done_1000_epochs")
    p.add_argument("--export", type=Path, default=None)
    p.add_argument("--vul-file", type=Path, required=True, help="Path to a *.vul pickle file.")
    p.add_argument("--t-k", type=float, required=True, help="Temperature in K (used to fill global variables).")
    p.add_argument("--p-barye", type=float, required=True, help="Pressure in barye (used to fill global variables).")
    p.add_argument("--start-t", type=float, default=1e-2, help="Anchor time in seconds (snapped to nearest VULCAN sample).")
    p.add_argument("--dt", type=float, default=1e-3, help="Constant dt (seconds) for ML rollout.")
    p.add_argument("--steps", type=int, default=1000, help="Number of ML steps.")
    p.add_argument("--species", type=str, default="H2O,CH4,CO,CO2,NH3,HCN,N2,C2H2", help="Comma-separated species to plot.")
    p.add_argument("--plot-every", type=int, default=20, help="Plot every Nth ML point (markers).")
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|mps")
    args = p.parse_args()

    export_path = args.export
    if export_path is None:
        export_path = (args.run_dir / "export_cpu_dynB_1step_phys.pt2").resolve()
    export_path = export_path.expanduser().resolve()
    if not export_path.exists():
        raise FileNotFoundError(f"Export not found: {export_path}")

    model, meta = _load_export(export_path)
    sp_vars = meta.get("species_variables")
    g_vars = meta.get("global_variables") or []
    if not isinstance(sp_vars, list) or not all(isinstance(x, str) for x in sp_vars) or not sp_vars:
        raise ValueError("Export is missing metadata.json with species_variables.")
    if not isinstance(g_vars, list) or not all(isinstance(x, str) for x in g_vars):
        raise ValueError("Export metadata.json has invalid global_variables.")

    device = _pick_device(args.device)
    model = model.to(device)
    dtype = next(iter(model.buffers())).dtype if any(True for _ in model.buffers()) else torch.float32

    t_all, MR_all, vul_species = _load_vulcan(args.vul_file)
    vul_idx = {n: i for i, n in enumerate(vul_species)}

    i0 = int(np.argmin(np.abs(t_all - float(args.start_t))))
    t0 = float(t_all[i0])

    bases = _bases_from_species_vars(list(sp_vars))
    S = len(bases)
    y0 = np.zeros((S,), dtype=np.float64)
    missing: List[str] = []
    for j, name in enumerate(bases):
        if name in vul_idx:
            y0[j] = float(MR_all[i0, vul_idx[name]])
        else:
            missing.append(name)
            y0[j] = 0.0

    y0 = np.clip(y0, 1e-30, None)
    y0 = y0 / max(float(y0.sum()), 1e-30)

    g = np.zeros((len(g_vars),), dtype=np.float64)
    for j, name in enumerate(g_vars):
        g[j] = _guess_global_value(name, t_k=float(args.t_k), p_barye=float(args.p_barye))

    y_pred = _rollout(
        model,
        y0=y0,
        g=g,
        dt_sec=float(args.dt),
        n_steps=int(args.steps),
        device=device,
        dtype=dtype,
    )
    y_pred = np.clip(y_pred, 1e-30, None)
    y_pred = y_pred / np.maximum(y_pred.sum(axis=1, keepdims=True), 1e-30)

    # Plot range (relative time window).
    t_rel_pred = float(args.dt) * np.arange(1, int(args.steps) + 1, dtype=np.float64)
    t_abs_end = t0 + float(t_rel_pred.max(initial=0.0))
    m = (t_all >= t0) & (t_all <= t_abs_end)
    t_rel_v = t_all[m] - t0
    MR_v = MR_all[m]

    species_plot = [s.strip() for s in args.species.split(",") if s.strip()]
    if not species_plot:
        raise ValueError("Empty --species list.")
    me = int(max(1, args.plot_every))
    idx_plot = np.arange(0, y_pred.shape[0], me, dtype=int)

    print(f"export:    {export_path}")
    print(f"vul_file:  {args.vul_file}")
    print(f"device:    {device} dtype={str(dtype).replace('torch.', '')}")
    print(f"anchor:    t0={t0:.3e}s (idx {i0})  dt={float(args.dt):.3e}s  steps={int(args.steps)}")
    if missing:
        print(f"warning:   {len(missing)} model species missing in VULCAN file (set to 0 in y0)")

    # Simple plot.
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping plot")
        return

    out_dir = (args.run_dir / "plots").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"vulcan_overlay_export_phys_t0_{t0:.3e}_dt_{float(args.dt):.0e}_N{int(args.steps)}.png"

    plt.figure(figsize=(9, 5))
    for sp in species_plot:
        if sp not in vul_idx:
            continue
        if sp not in bases:
            continue

        jv = vul_idx[sp]
        jm = bases.index(sp)

        if t_rel_v.size:
            plt.plot(t_rel_v, np.clip(MR_v[:, jv], 1e-30, None), linestyle="-", linewidth=2.0, label=f"{sp} (VULCAN)")
        plt.plot(
            t_rel_pred[idx_plot],
            np.clip(y_pred[idx_plot, jm], 1e-30, None),
            linestyle="None",
            marker="o",
            markersize=3.0,
            markerfacecolor="none",
            label=f"{sp} (ML)",
        )

    plt.yscale("log")
    plt.xlabel("time after anchor (s)")
    plt.ylabel("mixing ratio")
    plt.title("VULCAN vs exported ML rollout")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()
