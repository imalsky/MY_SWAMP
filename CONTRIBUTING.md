# Contributing to MY_SWAMP

Thank you for considering a contribution.

## Reporting bugs / requesting features

- Please open a GitHub issue with:
  - a minimal reproducible example (ideally a short script),
  - your OS, Python version, and `jax`/`jaxlib` versions,
  - expected vs. observed behavior.

## Development setup

From the repository root:

```bash
python -m pip install -U pip
python -m pip install -e ".[dev]"
pytest -q
```

## Pull requests

- Keep changes focused and well-scoped.
- Add or update tests when fixing bugs or adding functionality.
- If you change numerical kernels or spectral conventions, update/extend the transform identity tests.
