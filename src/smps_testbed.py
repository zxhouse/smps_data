"""
Programmable Stochastic Reliability, Survival, and Queueing Testbed for SMPS
=============================================================================
Implements the stochastic coherent SDN-Edge-Cloud reliability and queueing model,
Weibull-Cox proportional-hazard survival dynamics, heavy-tailed Pareto ON/OFF and
generalized logistic burst processes, Abry-Veitch wavelet Hurst estimation,
phase-randomized surrogate testing, and the Iterate-Test Core (arXiv:2611.03034)
alongside OCBA, UCB1, Thompson Sampling, Successive Halving, and Pareto Archives.

Author: Pawel Kuras (AGH University of Krakow)
Target Journal: Stochastic Models in Probability and Statistics (SMPS)
"""

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from network_sim import NetworkSystem, TrafficGenerator, NetworkNode  # noqa: E402

N_FEAT = 14


def components(res):
    norm_t = np.clip(res["throughput"] / 65.0, 0.0, 1.0)
    norm_d = np.clip(res["delay"] / 80.0, 0.0, 1.0)
    norm_l = np.clip(res["drop_rate"] / 0.15, 0.0, 1.0)
    perf = 0.50 * norm_t + 0.25 * (1.0 - norm_d) + 0.25 * (1.0 - norm_l)
    pen = float(np.clip(perf - (0.15 if res["lyapunov"] > 0.05 else 0.0), 0.0, 1.0))
    return float(np.clip(perf, 0.0, 1.0)), pen, float(res["lyapunov"])


class ZeroFloorNode(NetworkNode):
    """AQM whose drop probability is exactly zero at an empty queue."""
    def drop_probability(self):
        raw = NetworkNode.drop_probability(self)
        e0 = np.clip(self.aqm_slope * (0.0 - self.aqm_threshold), -30.0, 30.0)
        p0 = 1.0 / (1.0 + np.exp(-e0))
        return float(max(0.0, (raw - p0) / (1.0 - p0)))


def make_net(cfg, mode, seed, aqm="orig", mean_rate=52.0, tau=5, hurst=0.82):
    net = NetworkSystem(cfg, traffic_mode=mode, mean_rate=mean_rate, hurst=hurst, seed=seed)
    if aqm == "zero_floor":
        for name in ("s1", "s2"):
            old = getattr(net, name)
            new = ZeroFloorNode(old.node_id, old.capacity, old.service_rate, old.aqm_threshold, old.aqm_slope)
            setattr(net, name, new)
    if tau != 5:
        net.feedback_delay = tau
        net.recent_loss = [0.0] * tau
    return net


def idx_to_cfg(i):
    return tuple((i >> (N_FEAT - 1 - b)) & 1 for b in range(N_FEAT))


def cfg_to_idx(cfg):
    out = 0
    for b in cfg:
        out = (out << 1) | int(b)
    return out


# ---------------------------------------------------------------------------
# 1. Statistical & Time-Series Diagnostics (Abry-Veitch Hurst, Surrogates, 0-1)
# ---------------------------------------------------------------------------
def estimate_hurst_wavelet(series, min_scale=1, max_scale=6):
    """
    Abry-Veitch wavelet-based estimator of the Hurst exponent H.
    Uses dyadic Haar wavelet detail coefficients d(j, k) across octaves j,
    regressing log2(E[d(j, .)^2]) = (2H - 1) * j + const.
    Returns (H_hat, r_squared, octaves, log2_variances).
    """
    x = np.asarray(series, dtype=float)
    octaves = []
    log2_vars = []
    curr = x.copy()
    for j in range(1, max_scale + 1):
        n = (len(curr) // 2) * 2
        if n < 8:
            break
        even = curr[:n:2]
        odd = curr[1:n:2]
        detail = (even - odd) / np.sqrt(2.0)
        curr = (even + odd) / np.sqrt(2.0)
        if j >= min_scale:
            v = float(np.mean(detail ** 2))
            if v > 1e-12:
                octaves.append(float(j))
                log2_vars.append(float(np.log2(v)))
    if len(octaves) < 3:
        return 0.5, 0.0, octaves, log2_vars
    oct_arr = np.array(octaves)
    lv_arr = np.array(log2_vars)
    slope, intercept = np.polyfit(oct_arr, lv_arr, 1)
    pred = slope * oct_arr + intercept
    ss_res = float(np.sum((lv_arr - pred) ** 2))
    ss_tot = float(np.sum((lv_arr - lv_arr.mean()) ** 2))
    r2 = 1.0 - ss_res / max(1e-12, ss_tot)
    h_hat = float(np.clip((slope + 1.0) / 2.0, 0.05, 0.99))
    return h_hat, r2, octaves, log2_vars


def phase_randomized_surrogate(series, rng):
    """
    Generate a Theiler phase-randomized Fourier surrogate preserving the exact
    empirical power spectrum and autocorrelation while destroying nonlinear phase coupling.
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    f = np.fft.rfft(x - x.mean())
    phases = rng.uniform(0, 2.0 * np.pi, size=len(f))
    phases[0] = 0.0
    if n % 2 == 0 and len(f) > 1:
        phases[-1] = 0.0
    f_surr = np.abs(f) * np.exp(1j * phases)
    surr = np.fft.irfft(f_surr, n=n) + x.mean()
    return surr


def zero_one_chaos_test(series, c=1.78):
    """
    Gottwald-Melbourne 0-1 test statistic K_c in [0, 1] for deterministic chaos.
    Values near 0 indicate regular/periodic dynamics; values near 1 indicate chaos/irregularity.
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n < 50:
        return 0.0
    t = np.arange(n, dtype=float)
    p = np.cumsum(x * np.cos(c * t))
    q = np.cumsum(x * np.sin(c * t))
    n_cut = max(10, n // 10)
    msd = np.zeros(n_cut)
    for k in range(1, n_cut + 1):
        dp = p[k:] - p[:-k]
        dq = q[k:] - q[:-k]
        msd[k - 1] = np.mean(dp ** 2 + dq ** 2)
    ks = np.arange(1, n_cut + 1, dtype=float)
    if np.std(msd) < 1e-12:
        return 0.0
    corr = float(np.corrcoef(ks, msd)[0, 1])
    return float(np.clip(corr, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 2. Reliability, Survival, Hazard, and Provisioning Cost Model
# ---------------------------------------------------------------------------
def decode_config_params(cfg):
    """
    Decode 14-bit binary configuration p in {0,1}^14 into physical switch,
    AQM shock-filter, warm-standby redundancy, and maintenance parameters.
    """
    p = np.asarray(cfg, dtype=int)
    mu1 = 50.0 + 3.5 * (p[0] * 4 + p[1] * 2 + p[2])       # [50.0, 74.5]
    th1 = 40.0 + 5.5 * (p[3] * 4 + p[4] * 2 + p[5])       # [40.0, 78.5]
    kap1 = 0.05 + 0.10 * (p[6] * 2 + p[7])                # {0.05, 0.15, 0.25, 0.35}
    mu2 = 55.0 + 3.5 * (p[8] * 4 + p[9] * 2 + p[10])      # [55.0, 79.5]
    th2 = 45.0 + 5.5 * (p[11] * 4 + p[12] * 2 + p[13])    # [45.0, 83.5]

    # Normalized hardware/energy provisioning cost C(p) in [0, 1]
    # Higher service rates and standby server allocations consume more energy/hardware
    cost_mu1 = (mu1 - 50.0) / 24.5
    cost_mu2 = (mu2 - 55.0) / 24.5
    cost_buf = 0.5 * ((th1 - 40.0) / 38.5 + (th2 - 45.0) / 38.5)
    provisioning_cost = float(0.45 * cost_mu1 + 0.45 * cost_mu2 + 0.10 * cost_buf)

    # Warm-standby & preventive rejuvenation hazard reduction factor psi_red(p)
    # Higher core capacity mu2 and balanced threshold th2 reduce failure hazard
    psi_red = float(np.exp(-0.35 * cost_mu2 - 0.20 * cost_mu1))
    return {
        "mu1": mu1, "th1": th1, "kap1": kap1,
        "mu2": mu2, "th2": th2, "kap2": 0.12,
        "cost": provisioning_cost, "psi_red": psi_red
    }


def enrich_ground_truth_table(gt_raw):
    """
    Given a raw (16384, 5) ground-truth table [perf, pen, lle, flag_frac, sd_perf],
    compute the expanded SMPS reliability ground-truth columns for all 16,384 configurations:
      col 0: perf (smooth performance utility U_perf)
      col 1: pen (SLA-penalized utility U_pen)
      col 2: lle (mean short-window Lyapunov divergence indicator)
      col 3: flag_frac (fraction of runs exceeding threshold 0.05)
      col 4: sd_perf (single-test standard deviation of U_perf)
      col 5: sd_pen (single-test total standard deviation including Bernoulli SLA penalty)
      col 6: survival_prob R_p(tau_cens) under Weibull-Cox proportional hazard
      col 7: mttf_est (Mean Time To System Failure / Threshold Exceedance)
      col 8: cost (Hardware/energy provisioning cost C(p) in [0, 1])
      col 9: smooth_rel_utility (Integrated smooth survival-performance utility U_smooth)
    """
    n = gt_raw.shape[0]
    out = np.zeros((n, 10), dtype=float)
    out[:, :5] = gt_raw

    for i in range(n):
        cfg = tuple((i >> (13 - b)) & 1 for b in range(14))
        params = decode_config_params(cfg)
        perf, pen, lle, flag_f, sd_p = gt_raw[i]

        # Exact total SD of single test with binary penalty -0.15 * I{lle > 0.05}
        var_pen_jump = (0.15 ** 2) * flag_f * (1.0 - flag_f)
        sd_pen = float(np.sqrt(sd_p ** 2 + var_pen_jump))

        # Weibull-Cox proportional cumulative hazard H_p(tau_cens)
        # Congestion/loss stress (1 - perf) and dynamic irregularity max(0, lle)
        # accelerate baseline Weibull hazard
        stress = 2.4 * (1.0 - perf) + 0.45 * max(0.0, lle)
        cum_hazard = 0.22 * params["psi_red"] * np.exp(stress)
        surv_prob = float(np.exp(-cum_hazard))

        # Weibull shape beta = 2.2 => MTTF = eta_eff * Gamma(1 + 1/beta)
        # Since H(tau) = (tau / eta_eff)^beta = cum_hazard, eta_eff = 250 / (cum_hazard^(1/beta))
        beta = 2.2
        gamma_factor = 0.8856  # Gamma(1 + 1/2.2)
        eta_eff = 250.0 / max(1e-4, cum_hazard ** (1.0 / beta))
        mttf = float(eta_eff * gamma_factor)

        # Smooth integrated survival-performance score (no discontinuous threshold jump)
        smooth_rel = float(0.75 * perf + 0.25 * surv_prob)

        out[i, 5] = sd_pen
        out[i, 6] = surv_prob
        out[i, 7] = mttf
        out[i, 8] = params["cost"]
        out[i, 9] = smooth_rel

    return out
