# DDX Funding-Rate Options Analysis

Quantitative analysis of DDX protocol option-style derivatives for hedging perpetual futures funding rates, compared against linear hedges (swaps/futures).

## Quick Start

```bash
pip install -e ".[dev]"
pytest                              # run unit tests
python scripts/build_dataset.py --input data/raw/bitmex_funding.csv
python scripts/run_descriptives.py --data data/processed/bitmex_xbtusd.parquet
python scripts/run_premium_grid.py --data data/processed/bitmex_xbtusd.parquet
python scripts/run_frontier.py     --data data/processed/bitmex_xbtusd.parquet
python scripts/run_event_studies.py --data data/processed/bitmex_xbtusd.parquet
```

## Products Analyzed

| # | Product | Description |
|---|---------|-------------|
| 1 | Vanilla Funding Floor | Full insurance against negative funding, preserves upside |
| 2 | Distress-Activated Floor | Persistence-gated floor — only pays after sustained bad regime |
| 3 | Aggregate Stop-Loss | Reserve-layer hedge — covers losses beyond a deductible |
| 4 | Soft-Duration Cover | Smoothed version of #2 to reduce cliff effects |
| B | Swap Benchmark | Linear hedge — lock funding at fixed rate |

See `docs/contract_specs.md` for full mathematical specifications.
See `docs/Master_Implementation_Plan.md` for the detailed multi-phase implementation plan.

## Implementation Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | MVP pipeline hardening (regularity gate, loss-only metrics, swap variants, config-driven scripts) | Pending |
| 2 | Synthetic sanity check (notebook 01) | Pending |
| 3 | BitMEX data integration + QA | Pending |
| 4 | Descriptive funding analytics (notebook 02) | Pending |
| 5 | Single-series MVP results at 30d (notebook 03) | Pending |
| 6 | Premium surfaces + parameter sweeps (notebook 04) | Pending |
| 7 | Full hedge-efficiency frontier 7d/30d/90d (notebook 05) | Pending |
| 8 | Stress episode event studies (notebook 06) | Pending |
| 9 | Regime calibration + stress pricing (notebook 07) | Pending |

## Repo Structure

```
configs/           Analysis parameters, contract grids, stress events
data/raw/          Raw BitMEX dumps (gitignored)
data/processed/    Canonical parquet (gitignored)
data/samples/      Tiny committed samples for tests
docs/              Contract specs, implementation reports, master plan, ChatGPT Pro reports
notebooks/         Exploration and presentation (no business logic)
reports/           Generated figures, tables, markdown reports
scripts/           Reproducible pipeline entry points
src/ddx/           Core library
  data/            Schema, preprocessing, I/O
  payoffs/         All 4 DDX product payoff functions
  pricing/         Premium decomposition (pure + risk load + capital charge)
  risk/            VaR, CVaR, episode statistics
  backtest/        Rolling-window engine, hedge strategy definitions
  models/          Synthetic generators, regime models
  viz/             Plotting utilities
  utils/           Config loading, helpers
tests/             Unit tests
```
