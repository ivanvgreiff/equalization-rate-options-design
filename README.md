# DDX Funding-Rate Options Analysis

Quantitative analysis of option-style derivatives for hedging perpetual futures funding-rate risk, compared against linear hedges (swaps/futures) offered by existing protocols in 2026.

**[Technical Specifications of all derivatives](https://hackmd.io/@Omv7qL5_Q-SAKHCU75SPHw/HJnvGo5_Wl)** — full mathematical definitions, parameters, calibration methodology, and design rationale for every instrument analyzed.

Synthetic dollar stablecoin protocols like Ethena hold delta-neutral short-perp positions that receive funding when positive and pay funding when negative. They manage negative-funding risk with reserve funds. DDX proposes that option-style derivatives are more capital-efficient than reserves or other existing on-chain rate-derivatives for hedging this tail risk.

This repo builds the empirical evidence: historical funding-rate data across four exchanges, calibrated product parameters, and a pricing/risk framework to evaluate each derivative's cost and tail-risk reduction.

## Core Products

| Product | Idea |
|---------|------|
| **Vanilla Funding Floor** | Full insurance against negative funding (benchmark — upper bound on cost) |
| **Distress-Activated Floor (DAF)** | Only pays after sustained distress (m consecutive bad intervals). Cheaper than the full floor. |
| **Aggregate Stop-Loss (ASL)** | "My reserve absorbs the first D of losses; insure me beyond that." Highest efficiency per premium dollar. |
| **Swap** (benchmark) | Linear hedge — locks funding at a fixed rate. Eliminates variance but sacrifices upside. |

## Key Empirical Findings

**Data:** Bybit BTCUSD inverse perpetual (primary), plus BitMEX, Deribit, and Binance for cross-venue validation. ~7 years of 8-hour funding-rate history.

- **81.6% of funding intervals are non-negative** on Bybit. Negative funding is the minority — but when it hits, it hits hard (excess kurtosis ~37, heavy tails).
- **Three exchanges share a discrete base rate at 0.0001 per 8h** (10.95% APR). Deribit is structurally different (no base rate, no cap, left-skewed). Parameters calibrated on base-rate venues do not transfer to Deribit.
- **Funding-rate persistence is heavier than geometric** — streaks of negative funding last longer than a memoryless model would predict. This is the phenomenon that makes persistence-gated products (DAF) economically meaningful.
- **ASL deductible D = q90(Λ)** per horizon positions the product as a reinsurance tail layer, activating in ~10% of rolling windows.
- **DAF with m=3 (24h sustained distress)** activates in ~25% of 30-day windows — frequent enough to provide real protection, rare enough to be meaningfully cheaper than the vanilla floor.

## Quick Start

```bash
pip install -e ".[dev]"
pytest
```

**Data pipeline** (requires internet):
```bash
python scripts/fetch_bybit.py
python scripts/fetch_bitmex.py
python scripts/fetch_deribit.py
python scripts/fetch_binance.py
python scripts/build_dataset.py --venue bybit
python scripts/build_dataset.py --venue bitmex
python scripts/build_dataset.py --venue deribit
python scripts/build_dataset.py --venue binance
```

**Notebooks** (open in Jupyter/VS Code):
| Notebook | Phase | What it does |
|----------|-------|-------------|
| `01_synthetic_sanity.ipynb` | 2 | Validates payoff functions on synthetic two-regime Markov data |
| `02_funding_descriptives.ipynb` | 4 | Full statistical characterization of funding rates across 4 venues |
| `03_calibration.ipynb` | 5 | Freezes baseline parameters for all products using empirical calibration |

## Repo Structure

```
configs/        Analysis parameters, contract grids, stress events
data/           Raw CSVs (gitignored), processed parquets (gitignored), committed samples
docs/           Technical specs, implementation reports, master plan
notebooks/      Analysis and presentation (no business logic)
reports/        Generated figures, tables, markdown QA reports
scripts/        Data fetching, processing, and analysis entry points
src/ddx/        Core library (payoffs, pricing, risk, backtest, calibration, viz, utils)
tests/          Unit tests (81 tests)
```
