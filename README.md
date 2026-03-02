# DDX Equalization-Rate Options Design

Quantitative design and analysis of new option-style derivatives for hedging perpetual futures equalization-rates, compared against existing on-chain derivatives e.g. linear hedges (swaps/futures).

**[Technical Specifications of all Derivatives](https://hackmd.io/@Omv7qL5_Q-SAKHCU75SPHw/HJnvGo5_Wl)** — full mathematical definitions, parameters, calibration methodology, and design rationale for every instrument designed and analyzed.

Synthetic dollar stablecoin protocols like Ethena hold delta-neutral short-perp positions that receive funding when positive (set via the equalization-rates AKA "funding rates" by the respective exchange) and pay funding when negative. They manage negative-funding risk with reserve funds (though recently, companies like Pendle have publicly claimed to support Ethena's hedging needs via their interest rate swaps (Boros) protocol). 

The question is: **what is the most capital-efficient way to protect against this tail risk?**

Linear hedges (rate swaps) eliminate variance but sacrifice upside — the holder gives up (part of) the ~81.6% of intervals where funding is positive. Option-style products can provide downside protection while preserving upside, but only if they're correctly structured, parameterized, and priced.

This repo builds the empirical evidence: historical equalization-rate data across four exchanges, calibrated product parameters, and various pricing/risk frameworks to evaluate each derivative's cost and tail-risk reduction. 

## The Products

| Product | Structure | Design Rationale |
|---------|-----------|-----------------|
| **Vanilla Floor** | Pays `max(0, -f_i - d)` per interval | Full insurance benchmark; upper bound on cost |
| **Distress-Activated Floor (DAF)** | Floor that only activates after `m` consecutive bad intervals | Filters short noise; pays only during sustained distress. 34–43% cheaper than the full floor. |
| **Aggregate Stop-Loss (ASL)** | Pays `max(0, Λ - D)` on total period loss | Reinsurance-style tail layer. **Highest capital efficiency** (Eff_A = 2.66–2.70 vs 2.31–2.45 for Floor/DAF across all pricing methods). |

All parameters are empirically calibrated from 7.3 years of Bybit BTCUSD 8-hour funding data, with cross-venue validation on BitMEX, Binance, and Deribit. See the DDX Technical Specifications for precise definitions.

## Results

### What the rate data shows

Analysis of 7.3 years of 8-hour equalization-rate data across four exchanges (Bybit primary, BitMEX/Binance/Deribit for cross-venue validation) reveals three properties that shape the product design:

**Tail severity.** Only 18.4% of intervals are negative, but loss magnitude is extreme — excess kurtosis ~37, and fitting a GPD to stress-regime losses gives shape ξ = 0.124 > 0, a heavy Fréchet-type tail with no finite upper bound. Reserve-only strategies underestimate the worst outcomes.

**Persistence.** Losses don't arrive as isolated one-off intervals, they cluster into multi-interval episodes. This is what makes a persistence-gated product (DAF) viable as the gate filters noise without missing real distress.

**Nonstationarity.** The rate process switches between distinct calm and stress regimes. This shows up later as the dominant challenge for pricing, but it's visible already in the raw data.

All product parameters were calibrated directly from these empirical properties with conditional loss quantiles for the Floor deductible, aggregate-loss quantiles for the ASL attachment point, and activation-frequency analysis for the DAF streak threshold.

### What the products can cost

Loaded premiums (pure premium + CVaR-based risk load + capital charge) at a 30d horizon: Floor d=0.0001 at 1.31% of notional, DAF m=3 at 0.86%, ASL q90 at 1.10%. DAF is 34% cheaper than the floor. ASL has the highest capital efficiency (Eff_A = 2.66 — each unit of premium reduces required reserves by 2.66 units) because it concentrates payoff into the worst 10% of windows.

The risk load is 7–22× the pure premium across products. The loaded premium is overwhelmingly driven by the CVaR(1%) estimator — the average of the worst ~76 payoff windows out of ~7,600. This makes bootstrap confidence intervals wide: 66–144% of the point estimate. The premium surface is not quotable in a tight sense.

This requires confirmation: is the wide CI a methodology problem (wrong bootstrap block size, wrong pricing functional) or a data problem?

### Diagnosing the uncertainty

Three diagnostics were run to answer this. First, the bootstrap was re-run at five different block sizes (30 to 270 intervals). CI widths were stable in 16 of 18 product-horizon combinations, so the block size doesn't matter. Second, the CI was decomposed by premium component: the risk-load component accounting for 82% of the total CI width, confirms that the CVaR estimator is the bottleneck. Third, premiums were computed on rolling 2-year contiguous windows slid through the history, measuring how much the premium itself changes across eras. The answer: **10–30×**. Premiums on benign periods (2023–2025) are an order of magnitude lower than premiums on crisis-containing periods (2019–2021).

So wide CIs are driven by nonstationarity, not methodology. The rate process has distinct regimes, and 7.3 years contains only a handful of independent regime transitions.

### What holds up so far

To test whether the product rankings (as opposed to absolute premium levels) are robust, premiums were recomputed under 9 pricing methods: CVaR-loaded, Wang distortion (θ = 0.3, 0.5, 0.8), Esscher transform (θ = 0.5, 1.0, 2.0), and target-Sharpe. **ASL's sharpness advantage holds across all 9 methods.** The broad hierarchy (ASL/DAF cheaper and more efficient than Floor) is stable everywhere. The internal ordering of DAF m=2 vs ASL q90 shifts under CVaR loading — which penalizes ASL's concentrated tail payoffs more heavily — but the key conclusions are robust.

### Building a generative model

A 2-state Markov regime model with GPD tail augmentation was built to generate long histories respecting the empirical regime structure. This tightens Floor/DAF premium estimates ~2× — but fails for ASL and DAF activation rates because i.i.d. emissions within each state destroy the within-episode severity correlation that drives these path-dependent payoffs.

An **episode-based semi-Markov + EVT simulator** that resamples whole stress episodes (rather than individual intervals) fixes this structural failure, passing all 7 validation gates for DAF activation, ASL activation, and aggregate-loss quantiles. However, out-of-sample testing across eras reveals the model cannot predict across regimes (1/7 gates when training on early data and testing on late). This confirms that nonstationarity (not model specification) is the dominant uncertainty. The model is useful as a scenario generator, not a forward predictor.

### The hedge-efficiency frontier

The definitive comparison uses hedge-ratio frontiers over $h \in [0,1]$ with formal capital efficiency metrics, validated by a 52-month walk-forward protocol. Premium is treated as a deterministic expense separate from stochastic reserve risk — the reserve covers only the residual loss after the hedge payoff.

**The central finding:** at conventional cost-of-capital ($k = 10\%$), holding reserves costs only ~3 bps per 30d while option hedges cost 88–132 bps. Break-even $k^*$ is 450–530% annually. Options are not justified by capital savings alone — they are justified by tail-risk aversion: reducing worst-1% reserve draw from 3.7% to 0.5–1.8% of notional.

**Product ranking:** ASL q95 achieves the highest capital efficiency (Eff$_A$ = 2.70), stable across all pricing functionals. For protocols targeting CVaR $\leq$ 1.5%, only options can reach the required risk reduction (cheapest: ASL at ~85 bps). For targets $\geq$ 2.0%, a partial swap ($h \approx 0.3$) is cheapest because it diversifies without locking in stress-era rates.

| Target CVaR | Minimum-Cost Strategy | Hedge Ratio | Cost (bps / 30d) |
|---|---|---|---|
| ≤ 0.75% | Floor d=0.0001 | 0.92 | 121 |
| 1.0% | ASL q90 | 0.94 | 104 |
| 1.5% | ASL q95 | 0.92 | 85 |
| ≥ 2.0% | Swap (mean) | 0.40 | 2.6 |

**Robustness:** Absolute premium levels are uncertain (era bands show 3–10× variation), but the ranking — ASL most efficient, swaps cheapest for relaxed targets — holds across bootstrap, era, and model uncertainty layers.

## Notebooks

| # | Notebook | What it establishes |
|---|----------|---------------------|
| 01 | `01_synthetic_sanity` | Pipeline validation on synthetic data |
| 02 | `02_funding_descriptives` | Empirical regime structure, persistence, tails across 4 venues |
| 03 | `03_calibration` | Frozen product parameters from conditional loss and activation analysis |
| 04 | `04_premium_surfaces` | Premium decomposition, parameter sweeps, bootstrap CIs, quote sheets |
| 04b | `04b_ci_diagnostics` | Block-size sensitivity, component CIs, subsample dispersion |
| 04c | `04c_pricing_functionals` | Robustness of rankings across Wang/Esscher/CVaR/Sharpe pricing |
| 05 | `05_model_validation` | Regime-EVT model calibration, validation, model-based premiums |
| 05b | `05b_model_validation_clustered` | Episode-based semi-Markov model; passes all validation gates |
| 05c | `05c_model_risk_hardening` | Out-of-sample model-risk hardening; confirms nonstationarity dominates |
| 06 | `06_hedge_frontiers_v2` | Hedge-efficiency frontiers, capital metrics, walk-forward, decision rules |
