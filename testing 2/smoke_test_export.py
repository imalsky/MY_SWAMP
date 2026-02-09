#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import torch


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.synchronize()  # type: ignore[attr-defined]
        except Exception:
            pass


def _load_export(path: Path) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    extra: Dict[str, str] = {"metadata.json": ""}
    ep = torch.export.load(path, extra_files=extra)
    meta: Dict[str, Any] = {}
    if extra.get("metadata.json"):
        meta = json.loads(extra["metadata.json"])
    return ep.module(), meta


def _dims_from_meta(meta: Dict[str, Any]) -> Tuple[int, int]:
    sp = meta.get("species_variables")
    gv = meta.get("global_variables")
    if not isinstance(sp, list) or not all(isinstance(x, str) for x in sp) or not sp:
        raise ValueError("Export is missing metadata.json with species_variables.")
    if gv is None:
        gv = []
    if not isinstance(gv, list) or not all(isinstance(x, str) for x in gv):
        raise ValueError("Export metadata.json has invalid global_variables.")
    return len(sp), len(gv)


def _sample_inputs(
    *,
    B: int,
    S: int,
    G: int,
    device: str,
    dtype: torch.dtype,
    meta: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y = torch.rand((B, S), device=device, dtype=dtype)
    y = y / y.sum(dim=1, keepdim=True).clamp_min(1e-30)

    dt_min = float(meta.get("dt_min_seconds", 1e-6))
    dt_max = float(meta.get("dt_max_seconds", 1.0))
    dt_min = max(dt_min, 1e-30)
    dt_max = max(dt_max, dt_min * 10.0)

    u = torch.rand((B,), device=device, dtype=torch.float32)
    dt = torch.pow(10.0, u * (math.log10(dt_max) - math.log10(dt_min)) + math.log10(dt_min)).to(dtype)

    g = torch.zeros((B, G), device=device, dtype=dtype) if G > 0 else torch.empty((B, 0), device=device, dtype=dtype)
    return y, dt, g


@torch.inference_mode()
def _run_one(model: torch.nn.Module, *, device: str, dtype: torch.dtype, meta: Dict[str, Any], B: int, S: int, G: int) -> None:
    y, dt, g = _sample_inputs(B=B, S=S, G=G, device=device, dtype=dtype, meta=meta)
    out1 = model(y, dt, g)
    out2 = model(y, dt, g)
    _sync(device)

    if out1.shape != (B, S):
        raise RuntimeError(f"{device}: wrong output shape {tuple(out1.shape)} (expected {(B, S)})")
    if not torch.isfinite(out1).all():
        raise RuntimeError(f"{device}: non-finite outputs detected")
    if not torch.allclose(out1, out2, rtol=1e-6, atol=1e-6):
        raise RuntimeError(f"{device}: non-deterministic outputs for identical inputs")

    print(
        f"{device}: OK  B={B}  dtype={str(dtype).replace('torch.', '')}  "
        f"out[min,max]=({out1.min().item():.3e},{out1.max().item():.3e})"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Smoke test for exported PHYSICAL 1-step model.")
    p.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Path to .pt2 export (default: <repo>/models/<run>/export_cpu_dynB_1step_phys.pt2 via --run-dir).",
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "v1_done_1000_epochs",
    )
    p.add_argument("--b1", type=int, default=1)
    p.add_argument("--b2", type=int, default=7)
    p.add_argument("--cpu-only", action="store_true")
    args = p.parse_args()

    export_path = args.export
    if export_path is None:
        export_path = (args.run_dir / "export_cpu_dynB_1step_phys.pt2").resolve()
    export_path = export_path.expanduser().resolve()
    if not export_path.exists():
        raise FileNotFoundError(f"Export not found: {export_path}")

    model, meta = _load_export(export_path)
    S, G = _dims_from_meta(meta)

    devices = ["cpu"]
    if (not args.cpu_only) and torch.cuda.is_available():
        devices.append("cuda")
    if (not args.cpu_only) and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")

    print(f"export: {export_path}")
    print(f"dims:   S={S} G={G}")
    print(f"test:   B in {{{args.b1}, {args.b2}}}  devices={devices}")
    print("")

    for dev in devices:
        m, meta = _load_export(export_path)
        try:
            m = m.to(dev)
        except Exception as e:
            if dev != "cpu":
                print(f"{dev}: cannot move export to {dev} ({type(e).__name__}: {e}); skipping")
                continue
        dtype = next(iter(m.buffers())).dtype if any(True for _ in m.buffers()) else torch.float32
        _run_one(m, device=dev, dtype=dtype, meta=meta, B=int(args.b1), S=S, G=G)
        _run_one(m, device=dev, dtype=dtype, meta=meta, B=int(args.b2), S=S, G=G)


if __name__ == "__main__":
    main()
