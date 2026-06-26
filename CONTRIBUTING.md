# Contributing to neuroflower

Thank you for considering a contribution.

## Reporting bugs
Open a GitHub issue with: Python version, OS, minimal reproducer (≤ 20 lines
if possible), and the actual vs. expected behaviour.

## Proposing a change
For anything that touches the **locked parameters** (`p = 2`, mediator
weights, `k_hub`, RBF σ), please open an issue *first* to discuss. These
were chosen by a pre-registered diagnostic sweep (see methods paper §4);
changing them informally would invalidate the published validation.

For new features or bug fixes:
1. Fork, branch from `main`.
2. Add a test that fails on `main` and passes on your branch.
3. Open a pull request describing the change.

## Running tests locally
```
pip install -e ".[dev]"
pytest tests/ -v
```

## Code style
Pure NumPy. Avoid scipy/MNE/pyedflib as runtime dependencies — those would
break the deliberate dependency-light design (see methods paper §3).
