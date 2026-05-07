"""main_function.py

Compatibility wrapper for the original SWAMPE `main_function.py`.

Goals, as a JAX rewrite of SWAMPE
------------
1) Preserve the *call signature* (positional args, keyword names, and defaults)
   from the original numpy SWAMPE `main_function.main(...)` so existing driver
   scripts can call this module unchanged.

2) Delegate the actual numerical work to the JAX rewrite's `model.run_model(...)`
   so the differentiable core remains in one place.
"""

from __future__ import annotations

import argparse
from typing import Optional

from .model import run_model


def main(
    M,
    dt,
    tmax,
    Phibar,
    omega,
    a,
    test,
    g=9.8,
    forcflag=1,
    taurad=86400,
    taudrag=86400,
    DPhieq=4 * (10**6),
    a1=0.05,
    plotflag=1,
    plotfreq=5,
    minlevel=6,
    maxlevel=7,
    diffflag=1,
    modalflag=1,
    alpha=0.01,
    contflag=0,
    saveflag=1,
    expflag=0,
    savefreq=150,
    k1=2 * 10 ** (-4),
    k2=4 * 10 ** (-4),
    pressure=100 * 250 * 9.8 / 10,
    R=3000,
    Cp=13000,
    sigmaSB=5.7 * 10 ** (-8),
    K6=1.24 * 10 ** 33,
    custompath=None,
    contTime=None,
    timeunits="hours",
    verbose=True,
    *,
    # JAX-only extensions (kept keyword-only to avoid disturbing legacy call sites)
    K6Phi: Optional[float] = None,
):
    """Run SWAMPE-JAX with the legacy numpy `main_function.main(...)` signature.

    Parameters
    ----------
    M : int
        Spectral truncation with ``N=M`` in the downstream solver.
    dt : float
        Timestep in seconds.
    tmax : int
        Number of model steps to execute.
    Phibar : float
        Reference geopotential in SI units.
    omega : float
        Planetary rotation rate in radians per second.
    a : float
        Planetary radius in meters.
    test : Optional[int]
        Legacy test selector. ``0`` is remapped to forced mode, and ``1`` or
        ``2`` select the idealized Williamson-style tests supported by the JAX
        port.
    g : float
        Surface gravity in meters per second squared.
    forcflag : int or bool
        Enables thermal forcing when truthy.
    taurad : float
        Radiative relaxation timescale in seconds.
    taudrag : float
        Drag timescale in seconds.
    DPhieq : float
        Day-night equilibrium geopotential contrast.
    a1 : float
        Tilt angle used by the analytic test initial conditions.
    plotflag : int or bool
        Enables legacy plotting side effects when truthy.
    plotfreq : int
        Plot cadence in timesteps.
    minlevel : Optional[float]
        Minimum contour level for geopotential plots.
    maxlevel : Optional[float]
        Maximum contour level for geopotential plots.
    diffflag : int or bool
        Enables hyperdiffusion when truthy.
    modalflag : int or bool
        Enables the modal/Robert-Asselin correction branch when truthy.
    alpha : float
        Robert-Asselin filter coefficient.
    contflag : int or bool
        Enables continuation from saved state when truthy.
    saveflag : int or bool
        Enables legacy data-saving side effects when truthy.
    expflag : int or bool
        Selects the explicit timestepper when truthy, otherwise modified Euler.
    savefreq : int
        Save cadence in timesteps.
    k1 : float
        Legacy compatibility parameter accepted but ignored by this maintained
        JAX wrapper.
    k2 : float
        Legacy compatibility parameter accepted but ignored by this maintained
        JAX wrapper.
    pressure : float
        Legacy compatibility parameter accepted but ignored by this maintained
        JAX wrapper.
    R : float
        Legacy compatibility parameter accepted but ignored by this maintained
        JAX wrapper.
    Cp : float
        Legacy compatibility parameter accepted but ignored by this maintained
        JAX wrapper.
    sigmaSB : float
        Legacy compatibility parameter accepted but ignored by this maintained
        JAX wrapper.
    K6 : float
        Sixth-order diffusion coefficient for vorticity and divergence.
    custompath : Optional[str]
        Optional directory used for continuation loads and legacy outputs.
    contTime : Optional[str]
        Timestamp token used when loading continuation files.
    timeunits : str
        Units used to interpret continuation timestamps.
    verbose : bool
        Enables verbose progress logging in the downstream wrapper.
    K6Phi : Optional[float]
        Optional sixth-order diffusion coefficient for geopotential.

    Returns
    -------
    dict
        Output dictionary returned by `model.run_model(...)`, preserving the
        legacy wrapper behavior.
    """
    # Normalize flags (legacy code used 0/1 ints).
    forcflag_b = bool(forcflag)
    plotflag_b = bool(plotflag)
    diffflag_b = bool(diffflag)
    modalflag_b = bool(modalflag)
    contflag_b = bool(contflag)
    saveflag_b = bool(saveflag)
    expflag_b = bool(expflag)
    # Accepted for legacy API compatibility with the original SWAMPE entrypoint.
    _ = (k1, k2, pressure, R, Cp, sigmaSB)

    # The legacy main_function.py accepted integer test codes; in the maintained
    # SWAMPE/model.py driver, forced mode is represented as test=None.
    # Many users passed test=0 to mean "forced"; keep that convention.
    if test == 0:
        test = None

    if test not in (None, 1, 2):
        raise NotImplementedError(
            f"test={test!r} is not supported by this JAX port. Supported: None, 1, 2. "
            "(The extra test modes referenced by the legacy numpy main_function are not "
            "implemented in the maintained SWAMPE/model.py driver shipped in the provided archive.)"
        )

    # Delegate to the JAX SWAMPE-compatible wrapper (which in turn calls the
    # differentiable core based on lax.scan).
    return run_model(
        M=int(M),
        dt=float(dt),
        tmax=int(tmax),
        Phibar=float(Phibar),
        omega=float(omega),
        a=float(a),
        test=test,
        g=float(g),
        forcflag=forcflag_b,
        taurad=float(taurad),
        taudrag=float(taudrag),
        DPhieq=float(DPhieq),
        a1=float(a1),
        plotflag=plotflag_b,
        plotfreq=int(plotfreq),
        minlevel=None if minlevel is None else float(minlevel),
        maxlevel=None if maxlevel is None else float(maxlevel),
        diffflag=diffflag_b,
        modalflag=modalflag_b,
        alpha=float(alpha),
        contflag=contflag_b,
        saveflag=saveflag_b,
        expflag=expflag_b,
        savefreq=int(savefreq),
        K6=float(K6),
        custompath=custompath,
        contTime=contTime,
        timeunits=str(timeunits),
        verbose=bool(verbose),
        K6Phi=K6Phi,
    )


def _parse_args() -> argparse.Namespace:
    """Parse CLI flags for the legacy-compatible SWAMPE wrapper.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments for :func:`cli_main`.
    """
    p = argparse.ArgumentParser(description="SWAMPE shallow-water model (JAX rewrite).")

    # Mirror common knobs; keep defaults aligned with SWAMPE/main_function.py.
    p.add_argument("--M", type=int, default=42, help="Triangular truncation (M=N). ")
    p.add_argument("--dt", type=float, default=600.0, help="Time step in seconds.")
    p.add_argument("--tmax", type=int, default=200, help="Number of time steps.")
    p.add_argument(
        "--test",
        type=int,
        default=0,
        help="0 => forced mode (maps to test=None); 1 or 2 for idealized tests.",
    )

    p.add_argument("--a", type=float, default=6.37122e6)
    p.add_argument("--omega", type=float, default=7.292e-5)
    # Match the default used by my_swamp.model.run_model (g=9.8) so behavior is
    # consistent across the CLI, the legacy main(), and the run_model wrapper.
    p.add_argument("--g", type=float, default=9.8)
    p.add_argument("--Phibar", type=float, default=3.0e5)

    p.add_argument("--taurad", type=float, default=86400.0)
    p.add_argument("--taudrag", type=float, default=86400.0)

    p.add_argument("--K6", type=float, default=1.24e33)
    p.add_argument("--DPhieq", type=float, default=4.0e6)
    p.add_argument("--a1", type=float, default=0.05)

    p.add_argument("--timeunits", type=str, default="hours", choices=["steps", "seconds", "minutes", "hours", "days"])
    p.add_argument("--verbose", action="store_true")

    # Flags with legacy-meaningful names.
    p.add_argument("--forc", action="store_true")
    p.add_argument("--no-forc", dest="forc", action="store_false")
    p.set_defaults(forc=True)

    p.add_argument("--diff", action="store_true")
    p.add_argument("--no-diff", dest="diff", action="store_false")
    p.set_defaults(diff=True)

    p.add_argument("--modal", action="store_true")
    p.add_argument("--no-modal", dest="modal", action="store_false")
    p.set_defaults(modal=True)

    p.add_argument("--explicit", action="store_true")
    p.add_argument("--implicit", dest="explicit", action="store_false")
    p.set_defaults(explicit=False)

    p.add_argument("--plot", action="store_true")
    p.add_argument("--no-plot", dest="plot", action="store_false")
    p.set_defaults(plot=False)

    p.add_argument("--plotfreq", type=int, default=5)
    p.add_argument("--save", action="store_true")
    p.add_argument("--no-save", dest="save", action="store_false")
    p.set_defaults(save=True)
    p.add_argument("--savefreq", type=int, default=150)

    # Continuation
    p.add_argument("--cont", action="store_true")
    p.add_argument("--no-cont", dest="cont", action="store_false")
    p.set_defaults(cont=False)
    p.add_argument("--custompath", type=str, default=None)
    p.add_argument("--contTime", type=str, default=None)

    # JAX-only extensions (kept optional)
    p.add_argument("--K6Phi", type=float, default=None)
    p.add_argument("--use-scipy-basis", action="store_true")

    return p.parse_args()


def cli_main() -> None:
    """Run the command-line entrypoint for the legacy wrapper module.

    This parses command-line arguments, normalizes the legacy ``test=0``
    convention to ``test=None``, and dispatches to :func:`main`.
    """
    args = _parse_args()
    test_val: Optional[int] = None if int(args.test) == 0 else int(args.test)

    main(
        int(args.M),
        float(args.dt),
        int(args.tmax),
        float(args.Phibar),
        float(args.omega),
        float(args.a),
        test_val,
        g=float(args.g),
        forcflag=bool(args.forc),
        taurad=float(args.taurad),
        taudrag=float(args.taudrag),
        DPhieq=float(args.DPhieq),
        a1=float(args.a1),
        plotflag=bool(args.plot),
        plotfreq=int(args.plotfreq),
        diffflag=bool(args.diff),
        modalflag=bool(args.modal),
        contflag=bool(args.cont),
        saveflag=bool(args.save),
        expflag=bool(args.explicit),
        savefreq=int(args.savefreq),
        K6=float(args.K6),
        custompath=args.custompath,
        contTime=args.contTime,
        timeunits=str(args.timeunits),
        verbose=bool(args.verbose),
        K6Phi=args.K6Phi,
    )


if __name__ == "__main__":
    cli_main()
