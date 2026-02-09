#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch


def _parse_batch_sizes(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("Empty --batch-sizes.")
    return out


def _available_devices(requested: Sequence[str]) -> List[str]:
    out: List[str] = []
    for d in requested:
        d = d.strip().lower()
        if not d:
            continue
        if d == "cpu":
            out.append("cpu")
        elif d == "cuda":
            if torch.cuda.is_available():
                out.append("cuda")
        elif d == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                out.append("mps")
        else:
            raise ValueError(f"Unknown device '{d}'. Use cpu,cuda,mps.")
    if not out:
        out = ["cpu"]
    return out


def _infer_module_dtype(m: torch.nn.Module) -> torch.dtype:
    for p in m.parameters():
        return p.dtype
    for b in m.buffers():
        return b.dtype
    return torch.float32


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
    # y_phys: positive, normalized rows (mixing-ratio-like).
    y = torch.rand((B, S), device=device, dtype=dtype)
    y = y / y.sum(dim=1, keepdim=True).clamp_min(1e-30)

    # dt_seconds: sample log-uniform from metadata range if available.
    dt_min = float(meta.get("dt_min_seconds", 1e-6))
    dt_max = float(meta.get("dt_max_seconds", 1.0))
    dt_min = max(dt_min, 1e-30)
    dt_max = max(dt_max, dt_min * 10.0)

    log_min = math.log10(dt_min)
    log_max = math.log10(dt_max)
    u = torch.rand((B,), device=device, dtype=torch.float32)
    dt = torch.pow(10.0, u * (log_max - log_min) + log_min).to(dtype)

    # g_phys: if you have meaningful globals, replace this with real values.
    g = torch.zeros((B, G), device=device, dtype=dtype) if G > 0 else torch.empty((B, 0), device=device, dtype=dtype)
    return y, dt, g


@torch.inference_mode()
def _bench(
    model: torch.nn.Module,
    *,
    device: str,
    dtype: torch.dtype,
    meta: Dict[str, Any],
    B: int,
    S: int,
    G: int,
    warmup: int,
    iters: int,
) -> float:
    y, dt, g = _sample_inputs(B=B, S=S, G=G, device=device, dtype=dtype, meta=meta)

    for _ in range(max(1, warmup)):
        _ = model(y, dt, g)
    _sync(device)

    t0 = time.perf_counter()
    for _ in range(max(1, iters)):
        _ = model(y, dt, g)
    _sync(device)
    t1 = time.perf_counter()

    return (t1 - t0) / float(max(1, iters))


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark exported PHYSICAL 1-step model vs batch size.")
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
        help="Run dir containing export_cpu_dynB_1step_phys.pt2 (used if --export not set).",
    )
    p.add_argument("--devices", type=str, default="cpu,cuda", help="Comma-separated devices to test: cpu,cuda,mps.")
    p.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32,64,128,256,512,1024,2048")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=200)
    args = p.parse_args()

    export_path = args.export
    if export_path is None:
        export_path = (args.run_dir / "export_cpu_dynB_1step_phys.pt2").resolve()
    export_path = export_path.expanduser().resolve()
    if not export_path.exists():
        raise FileNotFoundError(f"Export not found: {export_path}")

    devices = _available_devices(args.devices.split(","))
    batch_sizes = _parse_batch_sizes(args.batch_sizes)

    # Load once to read metadata and infer dims.
    model0, meta = _load_export(export_path)
    S, G = _dims_from_meta(meta)

    print(f"export:   {export_path}")
    print(f"dims:     S={S} G={G}")
    print(f"devices:  {', '.join(devices)}")
    print(f"batches:  {batch_sizes}")
    print("")

    rows: List[Tuple[str, int, float]] = []  # (device, B, us/sample)

    for dev in devices:
        model, meta = _load_export(export_path)
        try:
            model = model.to(dev)
        except Exception as e:
            if dev != "cpu":
                print(f"{dev}: cannot move export to {dev} ({type(e).__name__}: {e}); skipping")
                continue
        dtype = _infer_module_dtype(model)

        for B in batch_sizes:
            try:
                sec_per_call = _bench(
                    model,
                    device=dev,
                    dtype=dtype,
                    meta=meta,
                    B=int(B),
                    S=S,
                    G=G,
                    warmup=int(args.warmup),
                    iters=int(args.iters),
                )
            except Exception as e:
                print(f"{dev}: B={B} failed ({type(e).__name__}: {e}); skipping")
                continue

            rows.append((dev, int(B), (1e6 * sec_per_call) / float(B)))

    if not rows:
        print("No benchmark results.")
        return

    # Pretty print.
    print(f"{'device':<6}  {'B':>6}  {'us/sample':>12}")
    print("-" * 26)
    for dev, B, us in rows:
        print(f"{dev:<6}  {B:>6d}  {us:>12.3f}")


if __name__ == "__main__":
    main()
