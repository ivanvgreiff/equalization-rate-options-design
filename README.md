# DDX: Option-Style Derivatives for Perpetual Funding-Rate Risk

Quantitative design, calibration, and pricing of new option-style derivatives that hedge the tail risk of perpetual-swap funding for short-perp positions (e.g., Ethena-style reserve management). This work builds the empirical case that structured products — not just linear hedges — are the right tool for managing funding-rate exposure, and provides the pricing and risk framework to evaluate them.

**[Technical Specifications of All Derivatives](docs/Technical_Specifications.md)** — full mathematical definitions, parameters, calibration methodology, and design rationale.

---

## The Problem

Synthetic dollar stablecoin protocols hold delta-neutral short-perp positions that receive funding when positive and pay funding when negative. Negative funding drains reserves. The question is: **what is the most capital-efficient way to protect against this tail risk?**

Linear hedges (funding-rate swaps) eliminate variance but sacrifice upside — the holder gives up the ~81.6% of intervals where funding is positive. Option-style products can provide downside protection while preserving upside, but only if they're correctly structured, parameterized, and priced.

## The Products

| Product | Structure | Design Rationale |
|---------|-----------|-----------------|
| **Vanilla Floor** | Pays `max(0, -f_i - d)` per interval | Full insurance benchmark; upper bound on cost |
| **Distress-Activated Floor (DAF)** | Floor that only activates after `m` consecutive bad intervals | Filters short noise; pays only during sustained distress. 34–43% cheaper than the full floor. |
| **Aggregate Stop-Loss (ASL)** | Pays `max(0, Λ - D)` on total period loss | Reinsurance-style tail layer. Highest efficiency per premium dollar (sharpness 1.39 vs ~1.24 for Floor/DAF). |

All parameters are empirically calibrated from 7.3 years of Bybit BTCUSD 8-hour funding data, with cross-venue validation on BitMEX, Binance, and Deribit. See [Technical Specifications](docs/Technical_Specifications.md) for precise definitions.

## Key Findings

### Funding-rate empirical properties

- **81.6% of Bybit intervals are non-negative.** Negative funding is the minority — but tail losses are severe (excess kurtosis ~37).
- **Three exchanges share a discrete base rate at 0.0001/8h** (10.95% APR). Deribit is structurally different and requires separate calibration.
- **Funding-rate persistence is heavier than geometric.** Negative streaks last longer than a memoryless model predicts — the phenomenon that makes the DAF economically meaningful.

### Premium surfaces and pricing uncertainty

- **Risk load dominates loaded premiums at 4.4x the pure premium** (30d Floor d=0). On synthetic data, this ratio was 1–2x. Real-data tails are far heavier than any diffusion model would suggest.
- **Bootstrap CIs are 66–144% of the point estimate.** Loaded premium surfaces are not quotable in any tight sense. This is not a methodology artifact — block-size sensitivity analysis confirms CIs are stable across block sizes {30, 60, 90, 180, 270} in 16/18 product-horizon combinations.
- **The CVaR(1%) risk-load estimator drives 82% of the CI width.** The remaining 18% comes from pure premium and capital charge.
- **Nonstationarity is the dominant uncertainty driver.** Premiums computed on 2-year rolling windows vary by 194–302% (range/mean) across different eras. Crisis-containing periods produce premiums 10–30x higher than benign periods. The funding-rate process is not approximately stationary.
- **ASL q95 at 90d is formally unquotable** — the bootstrap CI lower bound is exactly 0% (9% of resamples produce zero payoff because only 6 of 88 30-day blocks contain enough loss to breach the q95 threshold).

### Product comparison (30d horizon, CVaR-loaded premium)

| Product | Loaded (% notional) | Sharpness | Activation | Savings vs Floor d=0 |
|---------|--------------------:|----------:|-----------:|---------------------:|
| Floor d=0 (benchmark) | 1.525 | 1.24 | 88.0% | — |
| Floor d=0.0001 | 1.311 | 1.23 | 66.3% | 14.0% |
| DAF m=3 | 0.864 | 1.21 | 24.5% | 43.3% |
| ASL q90 | 1.102 | **1.39** | 10.0% | 27.7% |

Sharpness = |CVaR improvement| / premium. ASL q90 delivers the most tail-risk reduction per dollar of premium. DAF m=3 is the cheapest product with meaningful protection.

## Approach

The work proceeds through a sequence of empirical analyses, each building on the previous:

1. **Pipeline validation** on synthetic two-regime Markov data (confirms payoff functions, premium decomposition, and hedge comparison framework work correctly).
2. **Descriptive funding analytics** across four exchanges (characterizes regimes, persistence, tails, cross-venue structure).
3. **Parameter calibration** using conditional loss quantiles, aggregate loss distributions, and activation frequency analysis (freezes all product parameters empirically).
4. **Premium surfaces** with full decomposition (pure + risk load + capital charge) and block-bootstrap uncertainty quantification.
5. **CI diagnostics** — block-size sensitivity, component-wise CI decomposition, and subsample dispersion (diagnoses that wide CIs are a data limitation + nonstationarity, not a methodology artifact).
6. **Alternative pricing functionals** (Wang distortion, Esscher transform) to test whether product rankings are robust to the choice of risk-loading principle.
7. **Regime-switching + EVT model** to produce smoother, regime-conditional premiums from a calibrated generative model.
8. **Uncertainty-aware hedge-efficiency frontiers** presenting all product comparisons as distributions with confidence bands.

## Notebooks

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | `01_synthetic_sanity` | Pipeline validation on synthetic data |
| 02 | `02_funding_descriptives` | Empirical characterization of funding rates (4 venues) |
| 03 | `03_calibration` | Parameter calibration and frozen baseline parameters |
| 04 | `04_premium_surfaces` | Premium sweeps, decomposition, bootstrap CIs, quote sheets |
| 04b | `04b_ci_diagnostics` | Block-size sensitivity, component CIs, subsample dispersion |
| 04c | `04c_pricing_functionals` | Wang/Esscher pricing comparison (in progress) |
| 05 | `05_model_validation` | Regime-EVT model fitting and validation (planned) |
| 06 | `06_hedge_frontiers` | Uncertainty-aware product comparison (planned) |

## Repository Structure

```
src/ddx/          Core library: payoffs, pricing, risk metrics, bootstrap, calibration, visualization
notebooks/        Analysis notebooks (the primary research output)
configs/          Product parameters, analysis settings, stress events
docs/             Technical specifications, implementation reports, master plan
reports/          Generated tables (CSV) and figures (PNG)
tests/            92 unit tests
```
