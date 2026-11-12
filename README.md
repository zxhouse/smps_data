# Stochastic Reliability, Survival, and Iterative Optimization Testbed for Self-Similar and Chaotic Queueing Networks

**Author:** Paweł Kuraś  
**Affiliation:** AGH University of Krakow, Faculty of Computer Science, Electronics and Telecommunications, Institute of Telecommunications and Cybersecurity, al. Adama Mickiewicza 30, 30-059 Krakow, Poland  
**Email:** `pkuras@agh.edu.pl`  

---

## Overview

This repository contains the programmable stochastic simulation testbed, exhaustive $16{,}384$-configuration ground-truth data tables, experimental runners for all seven paired Monte Carlo experiments ($\mathrm{E1}$–$\mathrm{E7}$), publication figure generators (`fig1`–`fig8`), and automated numerical verifier accompanying the research study:

> **Paweł Kuraś**, *"Stochastic Ordering, Right-Censored Survival Analysis, and Iterative Reliability Optimization in Self-Similar and Chaotic Queueing Networks"*.

Building upon the **Iterate–Test Core** convergence theorem ([Kuraś et al., 2026, `arXiv:2611.03034`](http://arxiv.org/abs/2611.03034)), this framework models sequential reliability optimization of multi-stage edge-to-core queueing networks under right-censored Weibull–Cox survival observations across Poisson, heavy-tailed Pareto ON/OFF self-similar ($H = 0.82$), and generalized logistic chaotic burst arrivals.

---

## Repository Structure

```text
.
├── src/
│   ├── network_sim.py              # Core two-stage queueing & AQM shock-filtering simulator
│   ├── smps_testbed.py             # Weibull-Cox survival model, Abry-Veitch wavelet Hurst,
│   │                               # Theiler phase surrogates, 0-1 chaos test, & 16,384-state enrichment
│   ├── run_smps_experiments.py     # Complete runner for all 7 paired Monte Carlo experiments (E1-E7)
│   ├── render_smps_schematics.py   # Vector hardware & experimental pipeline schematics (Figs 1-3)
│   ├── render_smps_data_figures.py # Empirical multi-panel publication plots (Figs 4-8)
│   └── verify_smps_numbers.py      # Automated benchmark & numerical consistency verifier (39/39 checks)
├── data/
│   ├── smps_results.json           # Complete JSON experimental outputs for E1-E7
│   ├── gt_smps_poisson_orig.npy    # Exhaustive 16,384 x 10 ground-truth table (Poisson)
│   ├── gt_smps_self_similar_orig.npy # Exhaustive 16,384 x 10 ground-truth table (Pareto ON/OFF)
│   ├── gt_smps_chaos_orig.npy      # Exhaustive 16,384 x 10 ground-truth table (Logistic Chaos)
│   ├── gt_smps_chaos_zero_floor.npy# Exhaustive 16,384 x 10 ground-truth table (Zero-Floor AQM)
│   ├── dynamics.json               # Precomputed overload/feedback bifurcation sweep data
│   ├── round2_osc.json             # Sustained overload oscillation indicator data
│   ├── separation_v2.json          # Exhaustive 16,384-state separation violation statistics
│   └── summary_v2.json             # Exact simulator 100-replication paired allocation runs
├── Figs/
│   ├── fig1_network_architecture.pdf/.png
│   ├── fig2_iterate_test_pipeline.pdf/.png
│   ├── fig3_experimental_protocol.pdf/.png
│   ├── fig4_dynamics_hurst.pdf/.png
│   ├── fig5_landscape_survival.pdf/.png
│   ├── fig6_censoring_allocation.pdf/.png
│   ├── fig7_benchmarks_ocba.pdf/.png
│   └── fig8_persistent_pareto.pdf/.png
├── requirements.txt                # Python dependencies
└── LICENSE                         # MIT License
```

---

## Installation & Quick Start

1. **Install Python dependencies** (Python 3.9+):
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify all 39 quantitative benchmark results in `data/smps_results.json`:**
   ```bash
   python3 src/verify_smps_numbers.py
   ```

3. **Regenerate all 8 figures (`Figs/fig1_*.pdf` through `Figs/fig8_*.pdf`):**
   ```bash
   python3 src/render_smps_schematics.py
   python3 src/render_smps_data_figures.py
   ```

4. **Re-run all 7 experimental suites ($\mathrm{E1}$–$\mathrm{E7}$):**
   ```bash
   python3 src/run_smps_experiments.py
   ```

---

## Ground-Truth Table Columns (`data/gt_smps_*.npy`)

Each `.npy` file in `data/` has shape `(16384, 10)` corresponding to all $|\mathcal{P}| = 2^{14} = 16{,}384$ binary system configurations $p \in \{0,1\}^{14}$:
- `col 0`: `U_perf(p)` — Smooth performance utility
- `col 1`: `U_pen(p)` — SLA-penalized reliability utility
- `col 2`: `lambda_max(p)` — Mean short-window Rosenstein divergence indicator
- `col 3`: `q_p` — Empirical probability of SLA penalty activation ($\hat{\lambda}_{\max} > 0.05$)
- `col 4`: `sd_perf(p)` — Single-test standard deviation of smooth utility
- `col 5`: `sd_pen(p)` — Total single-test standard deviation including Bernoulli jump
- `col 6`: `R_p(tau_cens)` — Weibull–Cox right-censored survival probability at $\tau_{\mathrm{cens}} = 250$
- `col 7`: `MTTF(p)` — Mean Time To System Failure
- `col 8`: `C(p)` — Normalized hardware/energy provisioning cost in $[0, 1]$
- `col 9`: `U_smooth(p)` — Smooth integrated survival-performance score

---

## Acknowledgments

This work was supported by the Polish Ministry of Science and Higher Education with the subvention funds of the Faculty of Computer Science, Electronics and Telecommunications of AGH University.
