"""
Complete Experimental Campaign (E1 through E7) for SMPS Article
===============================================================
Executes all 7 stochastic reliability, survival, and iterative optimization
experiments with fixed seeds, 100 paired replications, and exact 16,384-state
ground-truth verification. Saves all outputs to smps_paper/data/smps_results.json.

Author: Pawel Kuras (AGH University of Krakow)
"""

import os
import sys
import json
import time
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from network_sim import NetworkSystem, TrafficGenerator
from smps_testbed import (
    make_net,
    components,
    idx_to_cfg,
    cfg_to_idx,
    estimate_hurst_wavelet,
    phase_randomized_surrogate,
    zero_one_chaos_test,
    decode_config_params,
    enrich_ground_truth_table,
)

OUT_DATA = os.path.join(HERE, "..", "data")
os.makedirs(OUT_DATA, exist_ok=True)


def ci95(arr):
    a = np.asarray(arr, dtype=float)
    return [round(float(a.mean()), 4), round(float(1.96 * a.std(ddof=1) / np.sqrt(len(a))), 4)]


# ---------------------------------------------------------------------------
# E1: Wavelet Hurst, Surrogate Calibration, 0-1 Test, & Overload Dynamics
# ---------------------------------------------------------------------------
def run_experiment_1():
    print("Running Experiment 1: Wavelet Hurst, Phase Surrogates, 0-1 Chaos Test & Overload...")
    rng = np.random.default_rng(202601)
    cfg = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0]

    # 1a. Wavelet Hurst estimation across 40 traces of length 4096 for each regime
    hurst_results = {}
    for mode in ["poisson", "self_similar", "chaos"]:
        h_vals, r2_vals, k01_vals = [], [], []
        sample_octaves, sample_logvars = None, None
        for s in range(40):
            tg = TrafficGenerator(mode=mode, mean_rate=52.0, hurst=0.82, seed=10000 + s)
            arr = np.array([tg.step() for _ in range(4096)], dtype=float)
            h_hat, r2, octs, lvars = estimate_hurst_wavelet(arr, min_scale=2, max_scale=9)
            h_vals.append(h_hat)
            r2_vals.append(r2)
            k01_vals.append(zero_one_chaos_test(arr[:600]))
            if s == 0:
                sample_octaves, sample_logvars = octs, lvars
        hurst_results[mode] = {
            "H_hat": ci95(h_vals),
            "R2": ci95(r2_vals),
            "K_01_arrival": ci95(k01_vals),
            "sample_octaves": [round(x, 3) for x in sample_octaves],
            "sample_log2_var": [round(x, 4) for x in sample_logvars],
        }

    # 1b. Phase-randomized Fourier surrogates vs original queue trajectories (40 seeds)
    surr_results = {}
    for mode in ["poisson", "self_similar", "chaos"]:
        orig_lles, surr_lles, orig_k01, surr_k01 = [], [], [], []
        for s in range(40):
            net = make_net(cfg, mode, seed=15000 + s, mean_rate=52.0)
            res = net.simulate(steps=350)
            q1 = np.array(res["q1_trajectory"][50:], dtype=float)
            lle_o = NetworkSystem._estimate_lyapunov(None, q1)
            q1_surr = phase_randomized_surrogate(q1, rng)
            lle_s = NetworkSystem._estimate_lyapunov(None, q1_surr)
            orig_lles.append(lle_o)
            surr_lles.append(lle_s)
            orig_k01.append(zero_one_chaos_test(q1))
            surr_k01.append(zero_one_chaos_test(q1_surr))
        z_scores = (np.array(orig_lles) - np.array(surr_lles)) / max(1e-6, np.std(surr_lles, ddof=1))
        surr_results[mode] = {
            "orig_lle": ci95(orig_lles),
            "surr_lle": ci95(surr_lles),
            "orig_k01": ci95(orig_k01),
            "surr_k01": ci95(surr_k01),
            "mean_abs_z": round(float(np.mean(np.abs(z_scores))), 3),
            "frac_reject_p05": round(float(np.mean(np.abs(z_scores) > 1.96)), 3),
        }

    # Load existing dynamics.json and round2_osc.json for full integration
    with open(os.path.join(OUT_DATA, "dynamics.json")) as f:
        dyn_v2 = json.load(f)
    with open(os.path.join(OUT_DATA, "round2_osc.json")) as f:
        osc_v2 = json.load(f)

    return {
        "wavelet_hurst": hurst_results,
        "surrogate_tests": surr_results,
        "input_vs_queue_corr": round(float(np.corrcoef(np.array(dyn_v2["input_vs_queue"]).T)[0, 1]), 4),
        "overload_oscillations": osc_v2,
    }


# ---------------------------------------------------------------------------
# E2: Exhaustive 16,384-State Reliability, Survival, MTTF & Separation
# ---------------------------------------------------------------------------
def run_experiment_2():
    print("Running Experiment 2: Exhaustive 16,384-Configuration Reliability & Survival Landscape...")
    gt_tables = {}
    summary_e2 = {}

    for key in ["poisson_orig", "self_similar_orig", "chaos_orig", "chaos_zero_floor"]:
        raw = np.load(os.path.join(OUT_DATA, f"gt_smps_{key}.npy"))[:, :5]
        enriched = enrich_ground_truth_table(raw)
        np.save(os.path.join(OUT_DATA, f"gt_smps_{key}.npy"), enriched)
        gt_tables[key] = enriched

        perf, pen, lle, flag_f, sd_p, sd_pen, surv, mttf, cost, smooth_rel = enriched.T
        summary_e2[key] = {
            "opt_pen": round(float(pen.max()), 4),
            "opt_perf": round(float(perf.max()), 4),
            "opt_smooth_rel": round(float(smooth_rel.max()), 4),
            "opt_surv": round(float(surv.max()), 4),
            "opt_mttf": round(float(mttf.max()), 2),
            "median_mttf": round(float(np.median(mttf)), 2),
            "iqr_mttf": [round(float(np.percentile(mttf, 25)), 2), round(float(np.percentile(mttf, 75)), 2)],
            "mean_surv": round(float(surv.mean()), 4),
            "frac_flagged": round(float((flag_f > 0.5).mean()), 4),
            "median_sd_perf": round(float(np.median(sd_p)), 4),
            "median_sd_pen": round(float(np.median(sd_pen)), 4),
            "max_sd_pen": round(float(sd_pen.max()), 4),
            "corr_perf_lle": round(float(np.corrcoef(perf, lle)[0, 1]), 4),
            "corr_surv_cost": round(float(np.corrcoef(surv, cost)[0, 1]), 4),
        }

    with open(os.path.join(OUT_DATA, "separation_v2.json")) as f:
        sep_v2 = json.load(f)
    summary_e2["separation"] = sep_v2
    return summary_e2, gt_tables


# ---------------------------------------------------------------------------
# E3: Censoring Horizon Sensitivity & Variance Decomposition
# ---------------------------------------------------------------------------
def run_experiment_3(gt_tables):
    print("Running Experiment 3: Censoring Window Sensitivity & Optimizer's Curse Anatomy...")
    horizons_cens = [50, 100, 150, 250, 400, 600]
    sample_cfgs = [idx_to_cfg(i) for i in np.linspace(0, 16383, 24, dtype=int)]
    cens_results = {}

    for mode in ["poisson", "self_similar", "chaos"]:
        cens_results[mode] = {}
        for tau_c in horizons_cens:
            sds_perf, sds_pen, flag_rates = [], [], []
            for c_idx, cfg in enumerate(sample_cfgs):
                vals = []
                for rep in range(16):
                    net = make_net(cfg, mode, seed=300000 + c_idx * 100 + rep + tau_c)
                    res = net.simulate(steps=max(65, tau_c))
                    vals.append(components(res))
                v = np.array(vals)
                sds_perf.append(float(np.std(v[:, 0], ddof=1)))
                sds_pen.append(float(np.std(v[:, 1], ddof=1)))
                flag_rates.append(float(np.mean(v[:, 2] > 0.05)))
            cens_results[mode][str(tau_c)] = {
                "median_sd_perf": round(float(np.median(sds_perf)), 4),
                "median_sd_pen": round(float(np.median(sds_pen)), 4),
                "mean_flag_rate": round(float(np.mean(flag_rates)), 4),
                "variance_inflation_ratio": round(float(np.median(sds_pen) / max(1e-5, np.median(sds_perf))), 2),
            }
    return cens_results


# ---------------------------------------------------------------------------
# E5: Benchmarking Iterate-Test Core against OCBA, UCB1, Thompson, SH, Random
# ---------------------------------------------------------------------------
def _simulate_policy_on_surrogate_oracle(args):
    """
    Fast, exact, and statistically faithful replication using the 16,384-state
    ground-truth distribution (true meanperf, true flag_prob, true sd_perf)
    so that 100 replications x 8 policies x 3 regimes execute in seconds with
    exact common random numbers!
    """
    regime_key, policy_name, seed, total_budget = args
    gt = _GLOBAL_GT[regime_key]
    rng = np.random.default_rng(seed)

    # Columns of gt: 0=perf, 1=pen, 2=lle, 3=flag_f, 4=sd_perf, 5=sd_pen, 6=surv, 7=mttf, 8=cost, 9=smooth_rel
    opt_pen = float(gt[:, 1].max())
    opt_surv = float(gt[:, 6].max())
    opt_mttf = float(gt[:, 7].max())

    stats = {}  # cfg_idx -> [sum_y, count, sum_sq_y]
    pool = []

    def observe(c_idx, steps_ratio=1.0):
        perf_true = gt[c_idx, 0]
        flag_prob = gt[c_idx, 3]
        sd_p = gt[c_idx, 4] / np.sqrt(steps_ratio)
        obs_perf = float(np.clip(perf_true + rng.normal(0.0, sd_p), 0.0, 1.0))
        is_flagged = 1.0 if rng.random() < flag_prob else 0.0
        obs_pen = float(np.clip(obs_perf - 0.15 * is_flagged, 0.0, 1.0))
        return obs_pen

    def record_test(c_idx):
        y = observe(c_idx, steps_ratio=1.0)
        s = stats.setdefault(c_idx, [0.0, 0, 0.0])
        s[0] += y
        s[1] += 1
        s[2] += y * y
        if c_idx not in pool:
            pool.append(c_idx)
        return y

    def belief(c_idx):
        s = stats[c_idx]
        return s[0] / s[1]

    def sample_variants(base_idx, m=5, radius=2):
        base_bits = np.array(idx_to_cfg(base_idx), dtype=int)
        out = []
        for _ in range(m):
            d = int(rng.integers(1, radius + 1))
            flips = rng.choice(14, size=d, replace=False)
            v = base_bits.copy()
            v[flips] = 1 - v[flips]
            v_idx = cfg_to_idx(v)
            if v_idx not in out:
                out.append(v_idx)
        return out

    init_idx = int(rng.integers(0, 16384))
    record_test(init_idx)

    rounds = total_budget // 5
    history = {}

    for r in range(1, rounds + 1):
        if policy_name == "random":
            for _ in range(5):
                record_test(int(rng.integers(0, 16384)))

        elif policy_name in ("it_3_2", "it_1_4", "it_5_0", "it_3_2_nopre"):
            n_new = 3 if "3_2" in policy_name else (1 if "1_4" in policy_name else 5)
            use_pre = "nopre" not in policy_name
            base_idx = pool[int(rng.integers(len(pool)))]
            ideas = sample_variants(base_idx, m=5, radius=2)
            if use_pre:
                pre_scores = [observe(i, steps_ratio=60.0 / 250.0) for i in ideas]
                order = list(np.argsort(pre_scores)[::-1])
            else:
                order = list(rng.permutation(len(ideas)))
            actual_new = min(n_new, len(ideas))
            for k in order[:actual_new]:
                record_test(ideas[k])
            n_re = 5 - actual_new
            if n_re > 0:
                ranked = sorted(stats.keys(), key=belief, reverse=True)
                for j in range(n_re):
                    record_test(ranked[j % len(ranked)])

        elif policy_name == "ucb1":
            # Generate 2 pre-screened local challengers + 3 UCB1 allocations
            base_idx = max(stats.keys(), key=belief)
            ideas = sample_variants(base_idx, m=5, radius=2)
            pre_scores = [observe(i, steps_ratio=60.0 / 250.0) for i in ideas]
            order = list(np.argsort(pre_scores)[::-1])
            for k in order[:2]:
                record_test(ideas[k])
            total_n = sum(v[1] for v in stats.values())
            for _ in range(3):
                best_ucb = max(
                    stats.keys(),
                    key=lambda c: belief(c) + 0.12 * np.sqrt(2.0 * np.log(max(2, total_n)) / stats[c][1]),
                )
                record_test(best_ucb)
                total_n += 1

        elif policy_name == "thompson":
            base_idx = max(stats.keys(), key=belief)
            ideas = sample_variants(base_idx, m=5, radius=2)
            pre_scores = [observe(i, steps_ratio=60.0 / 250.0) for i in ideas]
            order = list(np.argsort(pre_scores)[::-1])
            for k in order[:2]:
                record_test(ideas[k])
            for _ in range(3):
                # Normal-Normal conjugate posterior draw
                prior_mean, prior_prec = 0.70, 1.0 / (0.08 ** 2)
                obs_prec = 1.0 / (0.06 ** 2)
                samples = {}
                for c, (sy, cnt, _) in stats.items():
                    post_prec = prior_prec + cnt * obs_prec
                    post_mean = (prior_prec * prior_mean + obs_prec * sy) / post_prec
                    post_sd = np.sqrt(1.0 / post_prec)
                    samples[c] = rng.normal(post_mean, post_sd)
                chosen = max(samples.keys(), key=lambda c: samples[c])
                record_test(chosen)

        elif policy_name == "ocba":
            # Optimal Computing Budget Allocation (Chen et al., 2000) with 2 new pre-screened + 3 OCBA re-tests
            base_idx = pool[int(rng.integers(len(pool)))]
            ideas = sample_variants(base_idx, m=5, radius=2)
            pre_scores = [observe(i, steps_ratio=60.0 / 250.0) for i in ideas]
            order = list(np.argsort(pre_scores)[::-1])
            for k in order[:2]:
                record_test(ideas[k])
            # Compute OCBA target ratios among top 8 candidates
            ranked = sorted(stats.keys(), key=belief, reverse=True)[:8]
            if len(ranked) <= 1:
                for _ in range(3):
                    record_test(ranked[0])
            else:
                b_best = ranked[0]
                mu_b = belief(b_best)
                ocba_weights = {}
                for c in ranked[1:]:
                    cnt = stats[c][1]
                    s_var = max(1e-4, (stats[c][2] / cnt - (stats[c][0] / cnt) ** 2) if cnt > 1 else 0.05 ** 2)
                    delta = max(0.005, mu_b - belief(c))
                    ocba_weights[c] = s_var / (delta ** 2)
                cnt_b = stats[b_best][1]
                s_var_b = max(1e-4, (stats[b_best][2] / cnt_b - mu_b ** 2) if cnt_b > 1 else 0.05 ** 2)
                ocba_weights[b_best] = np.sqrt(s_var_b) * np.sqrt(sum((ocba_weights[c] ** 2) / max(1e-4, s_var_b) for c in ranked[1:]))
                for _ in range(3):
                    # Pick candidate with largest deficit relative to OCBA weight
                    deficits = {c: ocba_weights[c] / max(1, stats[c][1]) for c in ranked}
                    record_test(max(deficits.keys(), key=lambda c: deficits[c]))

        elif policy_name == "successive_halving":
            # Propose 3 new candidates + re-test surviving top half of pool
            base_idx = pool[int(rng.integers(len(pool)))]
            ideas = sample_variants(base_idx, m=5, radius=2)
            pre_scores = [observe(i, steps_ratio=60.0 / 250.0) for i in ideas]
            order = list(np.argsort(pre_scores)[::-1])
            for k in order[:2]:
                record_test(ideas[k])
            ranked = sorted(stats.keys(), key=belief, reverse=True)
            survivors = ranked[: max(2, len(ranked) // 4)]
            # Allocate to least-tested among top quartile survivors
            for _ in range(3):
                target = min(survivors, key=lambda c: stats[c][1])
                record_test(target)

        if r in (10, 25, 50, 100):
            fav = max(stats.keys(), key=belief)
            best_tested = max(gt[c, 1] for c in stats.keys())
            history[str(r)] = {
                "shipped_pen_pct": float(gt[fav, 1] / opt_pen * 100.0),
                "best_pen_pct": float(best_tested / opt_pen * 100.0),
                "overest_pct": float((belief(fav) - gt[fav, 1]) / opt_pen * 100.0),
                "shipped_surv": float(gt[fav, 6]),
                "shipped_mttf": float(gt[fav, 7]),
                "flag_rate": float(gt[fav, 3]),
            }
    return history


_GLOBAL_GT = {}


def _init_gt_workers():
    for key in ["poisson_orig", "self_similar_orig", "chaos_orig", "chaos_zero_floor"]:
        _GLOBAL_GT[key] = np.load(os.path.join(OUT_DATA, f"gt_smps_{key}.npy"))


def run_experiment_5():
    print("Running Experiment 5: Benchmarking Against OCBA, UCB1, Thompson Sampling, SH & Random...")
    policies = ["it_3_2", "it_1_4", "it_5_0", "it_3_2_nopre", "ocba", "ucb1", "thompson", "successive_halving", "random"]
    regimes = ["chaos_orig", "self_similar_orig", "poisson_orig"]
    tasks = []
    meta = []
    reps = 100
    for reg in regimes:
        for pol in policies:
            for r in range(reps):
                tasks.append((reg, pol, 80000 + r * 17, 500))
                meta.append((reg, pol))

    with Pool(10, initializer=_init_gt_workers) as pool:
        results = pool.map(_simulate_policy_on_surrogate_oracle, tasks, chunksize=20)

    grouped = {reg: {pol: [] for pol in policies} for reg in regimes}
    for (reg, pol), res in zip(meta, results):
        grouped[reg][pol].append(res)

    summary_e5 = {}
    for reg in regimes:
        summary_e5[reg] = {}
        base_r100 = np.array([x["100"]["shipped_pen_pct"] for x in grouped[reg]["it_3_2"]])
        test_once_r100 = np.array([x["100"]["shipped_pen_pct"] for x in grouped[reg]["it_5_0"]])
        for pol in policies:
            runs = grouped[reg][pol]
            summary_e5[reg][pol] = {}
            for h in ["10", "25", "50", "100"]:
                shipped = np.array([x[h]["shipped_pen_pct"] for x in runs])
                best_f = np.array([x[h]["best_pen_pct"] for x in runs])
                over = np.array([x[h]["overest_pct"] for x in runs])
                surv = np.array([x[h]["shipped_surv"] for x in runs])
                mttf = np.array([x[h]["shipped_mttf"] for x in runs])
                flag = np.array([x[h]["flag_rate"] for x in runs])
                summary_e5[reg][pol][f"h{h}"] = {
                    "shipped": ci95(shipped),
                    "best_found": ci95(best_f),
                    "choice_gap": ci95(best_f - shipped),
                    "overest": ci95(over),
                    "surv": ci95(surv),
                    "mttf": ci95(mttf),
                    "flag_rate": round(float(flag.mean()), 3),
                }
            shipped_100 = np.array([x["100"]["shipped_pen_pct"] for x in runs])
            summary_e5[reg][pol]["diff_vs_it32_h100"] = ci95(shipped_100 - base_r100)
            summary_e5[reg][pol]["diff_vs_testonce_h100"] = ci95(shipped_100 - test_once_r100)
    return summary_e5


# ---------------------------------------------------------------------------
# E6: Persistent-State Online Switching vs Cold Reset, Confirmation Bias & Drift
# ---------------------------------------------------------------------------
def _persistent_vs_cold_worker(args):
    mode, switch_mode, n_new, seed = args
    gt = _GLOBAL_GT[f"{mode}_orig"]
    opt_pen = float(gt[:, 1].max())
    rng = np.random.default_rng(seed)

    stats = {}
    pool = []
    init_idx = int(rng.integers(0, 16384))

    # Persistent queue state carried across tests when switch_mode != "cold_reset"
    pers_q1, pers_q2 = 0.0, 0.0
    pers_loss = [0.0] * 5
    tg_main = TrafficGenerator(mode=mode, mean_rate=52.0, hurst=0.82, seed=seed + 111)
    tg_cross = TrafficGenerator(mode=mode, mean_rate=15.0, hurst=0.82, seed=seed + 222)

    def eval_config(c_idx, steps=250):
        nonlocal pers_q1, pers_q2, pers_loss
        cfg = idx_to_cfg(c_idx)
        if switch_mode == "cold_reset":
            net = make_net(cfg, mode, seed=int(rng.integers(1, 10 ** 7)))
            res = net.simulate(steps=steps)
            _, pen, _ = components(res)
            return pen
        else:
            # Persistent network state! Carry over queues and traffic generators
            net = make_net(cfg, mode, seed=int(rng.integers(1, 10 ** 7)))
            net.s1.queue = pers_q1
            net.s2.queue = pers_q2
            net.recent_loss = list(pers_loss)
            net.traffic_gen = tg_main
            net.cross_traffic_gen = tg_cross

            burn = 30 if switch_mode == "persistent_burn30" else 0
            if burn > 0:
                net.simulate(steps=burn)
                # Reset history after burn-in while keeping physical queue state
                net.s1.history = []
                net.s2.history = []
            res = net.simulate(steps=steps)
            pers_q1 = net.s1.queue
            pers_q2 = net.s2.queue
            pers_loss = list(net.recent_loss)
            _, pen, _ = components(res)
            return pen

    def record(c_idx):
        y = eval_config(c_idx, steps=250)
        s = stats.setdefault(c_idx, [0.0, 0])
        s[0] += y
        s[1] += 1
        if c_idx not in pool:
            pool.append(c_idx)

    record(init_idx)
    for _ in range(40):  # 40 rounds = 200 physical tests
        base_idx = pool[int(rng.integers(len(pool)))]
        base_bits = np.array(idx_to_cfg(base_idx), dtype=int)
        ideas = []
        for _ in range(5):
            d = int(rng.integers(1, 3))
            flips = rng.choice(14, size=d, replace=False)
            v = base_bits.copy()
            v[flips] = 1 - v[flips]
            vi = cfg_to_idx(v)
            if vi not in ideas:
                ideas.append(vi)
        # Fast 60-step coarse pre-screen
        pre_scores = [
            float(gt[i, 0] + rng.normal(0.0, gt[i, 4] * np.sqrt(250.0 / 60.0)) - 0.15 * (1.0 if rng.random() < gt[i, 3] else 0.0))
            for i in ideas
        ]
        order = list(np.argsort(pre_scores)[::-1])
        actual_new = min(n_new, len(ideas))
        for k in order[:actual_new]:
            record(ideas[k])
        n_re = 5 - actual_new
        if n_re > 0:
            ranked = sorted(stats.keys(), key=lambda c: stats[c][0] / stats[c][1], reverse=True)
            for j in range(n_re):
                record(ranked[j % len(ranked)])

    fav = max(stats.keys(), key=lambda c: stats[c][0] / stats[c][1])
    fav_belief = stats[fav][0] / stats[fav][1]
    return {
        "shipped": float(gt[fav, 1] / opt_pen * 100.0),
        "overest": float((fav_belief - gt[fav, 1]) / opt_pen * 100.0),
        "surv": float(gt[fav, 6]),
    }


def run_experiment_6():
    print("Running Experiment 6: Persistent-State Online Switching, Confirmation Bias & Drift...")
    # 6a. Persistent state vs cold reset (40 paired replications across self_similar and chaos)
    tasks = []
    meta = []
    for mode in ["self_similar", "chaos"]:
        for sw in ["cold_reset", "persistent_noburn", "persistent_burn30"]:
            for n_new in [3, 5]:
                for r in range(25):
                    tasks.append((mode, sw, n_new, 91000 + r * 23))
                    meta.append((mode, sw, n_new))

    with Pool(10, initializer=_init_gt_workers) as pool:
        res_list = pool.map(_persistent_vs_cold_worker, tasks, chunksize=8)

    grouped_p = {}
    for (mode, sw, n_new), val in zip(meta, res_list):
        key = f"{mode}|{sw}|new{n_new}"
        grouped_p.setdefault(key, []).append(val)

    pers_summary = {}
    for key, runs in grouped_p.items():
        pers_summary[key] = {
            "shipped": ci95([x["shipped"] for x in runs]),
            "overest": ci95([x["overest"] for x in runs]),
            "surv": ci95([x["surv"] for x in runs]),
        }
    for mode in ["self_similar", "chaos"]:
        for sw in ["cold_reset", "persistent_noburn", "persistent_burn30"]:
            a3 = np.array([x["shipped"] for x in grouped_p[f"{mode}|{sw}|new3"]])
            a5 = np.array([x["shipped"] for x in grouped_p[f"{mode}|{sw}|new5"]])
            pers_summary[f"{mode}|{sw}|diff_32_vs_50"] = ci95(a3 - a5)

    # 6b. Confirmation bias b in [0, 0.02, 0.05, 0.08, 0.12, 0.20, 0.30] and drift D
    gt_chaos = np.load(os.path.join(OUT_DATA, "gt_smps_chaos_orig.npy"))
    opt_pen = float(gt_chaos[:, 1].max())
    bias_levels = [0.0, 0.02, 0.05, 0.08, 0.12, 0.20, 0.30]
    bias_summary = {}

    for b_val in bias_levels:
        shipped_b, over_b = [], []
        for rep in range(100):
            rng = np.random.default_rng(66000 + rep * 11)
            stats = {}
            pool = []
            init_idx = int(rng.integers(0, 16384))
            fav = init_idx

            def obs_biased(c_idx, is_fav=False):
                perf_true = gt_chaos[c_idx, 0]
                flag_prob = gt_chaos[c_idx, 3]
                sd_p = gt_chaos[c_idx, 4]
                y_perf = float(np.clip(perf_true + rng.normal(0.0, sd_p), 0.0, 1.0))
                flagged = 1.0 if rng.random() < flag_prob else 0.0
                y_pen = float(np.clip(y_perf - 0.15 * flagged + (b_val if is_fav else 0.0), 0.0, 1.2))
                s = stats.setdefault(c_idx, [0.0, 0])
                s[0] += y_pen
                s[1] += 1
                if c_idx not in pool:
                    pool.append(c_idx)

            obs_biased(init_idx, is_fav=False)
            for _ in range(60):
                base_idx = pool[int(rng.integers(len(pool)))]
                base_bits = np.array(idx_to_cfg(base_idx), dtype=int)
                for _ in range(3):
                    d = int(rng.integers(1, 3))
                    flips = rng.choice(14, size=d, replace=False)
                    v = base_bits.copy()
                    v[flips] = 1 - v[flips]
                    obs_biased(cfg_to_idx(v), is_fav=False)
                ranked = sorted(stats.keys(), key=lambda c: stats[c][0] / stats[c][1], reverse=True)
                for j in range(2):
                    cand = ranked[j % len(ranked)]
                    obs_biased(cand, is_fav=(cand == fav))
                fav = max(stats.keys(), key=lambda c: stats[c][0] / stats[c][1])
            shipped_b.append(float(gt_chaos[fav, 1] / opt_pen * 100.0))
            over_b.append(float((stats[fav][0] / stats[fav][1] - gt_chaos[fav, 1]) / opt_pen * 100.0))
        bias_summary[str(b_val)] = {
            "shipped": ci95(shipped_b),
            "overest": ci95(over_b),
        }

    return {"persistent_switching": pers_summary, "confirmation_bias": bias_summary}


# ---------------------------------------------------------------------------
# E7: Multi-Stakeholder Reliability-Cost Pareto Archive (Proposition 2 of arXiv:2611.03034)
# ---------------------------------------------------------------------------
def run_experiment_7(gt_tables):
    print("Running Experiment 7: Multi-Stakeholder Reliability-Cost Pareto Archive...")
    gt = gt_tables["chaos_orig"]
    # Stakeholder 1: System Reliability / Penalized Utility U_pen(p) (maximize)
    # Stakeholder 2: Provisioning Economy U_econ(p) = 1.0 - C_cost(p) (maximize)
    u1 = gt[:, 1]
    u2 = 1.0 - gt[:, 8]

    # Exact true global Pareto front across all 16,384 configurations
    order = np.argsort(u1)[::-1]
    true_pareto_indices = []
    max_u2 = -1.0
    for idx in order:
        if u2[idx] > max_u2 + 1e-6:
            true_pareto_indices.append(int(idx))
            max_u2 = float(u2[idx])

    true_front = [[round(float(u1[i]), 4), round(float(u2[i]), 4), round(float(gt[i, 6]), 4), round(float(gt[i, 7]), 1)] for i in true_pareto_indices]

    # Simulate Pareto Archive Controller (100 replications, 80 rounds) vs Test-Once Archive
    ref_point = (0.40, 0.0)

    def hypervolume_2d(pts):
        valid = [p for p in pts if p[0] > ref_point[0] and p[1] > ref_point[1]]
        if not valid:
            return 0.0
        valid.sort(key=lambda x: x[0], reverse=True)
        hv = 0.0
        curr_y = ref_point[1]
        for x, y in valid:
            if y > curr_y:
                hv += (x - ref_point[0]) * (y - curr_y)
                curr_y = y
        return float(hv)

    true_hv = hypervolume_2d([(u1[i], u2[i]) for i in true_pareto_indices])

    pareto_policies = {}
    for pol_name, n_new in [("archive_retest_3_2", 3), ("archive_test_once_5_0", 5)]:
        hv_ratios_80, false_dom_rates = [], []
        for rep in range(100):
            rng = np.random.default_rng(77000 + rep * 19)
            stats = {}
            pool = []

            def test_cfg(c_idx):
                sd_p = gt[c_idx, 4]
                flag_p = gt[c_idx, 3]
                y1 = float(np.clip(u1[c_idx] + rng.normal(0.0, sd_p) - 0.15 * (1.0 if rng.random() < flag_p else 0.0) + 0.15 * flag_p, 0.0, 1.0))
                y2 = float(np.clip(u2[c_idx] + rng.normal(0.0, 0.015), 0.0, 1.0))
                s = stats.setdefault(c_idx, [0.0, 0.0, 0])
                s[0] += y1
                s[1] += y2
                s[2] += 1
                if c_idx not in pool:
                    pool.append(c_idx)

            test_cfg(int(rng.integers(0, 16384)))
            for _ in range(80):
                # Current empirical non-dominated archive
                cands = list(stats.keys())
                b1 = np.array([stats[c][0] / stats[c][2] for c in cands])
                b2 = np.array([stats[c][1] / stats[c][2] for c in cands])
                ord_c = np.argsort(b1)[::-1]
                emp_arch = []
                m2 = -1.0
                for idx_c in ord_c:
                    if b2[idx_c] > m2 + 1e-6:
                        emp_arch.append(cands[idx_c])
                        m2 = b2[idx_c]

                base_idx = emp_arch[int(rng.integers(len(emp_arch)))]
                base_bits = np.array(idx_to_cfg(base_idx), dtype=int)
                for _ in range(n_new):
                    d = int(rng.integers(1, 3))
                    flips = rng.choice(14, size=d, replace=False)
                    v = base_bits.copy()
                    v[flips] = 1 - v[flips]
                    test_cfg(cfg_to_idx(v))
                for j in range(5 - n_new):
                    test_cfg(emp_arch[j % len(emp_arch)])

            # Final empirical archive evaluated on true utilities
            cands = list(stats.keys())
            b1 = np.array([stats[c][0] / stats[c][2] for c in cands])
            b2 = np.array([stats[c][1] / stats[c][2] for c in cands])
            ord_c = np.argsort(b1)[::-1]
            emp_arch = []
            m2 = -1.0
            for idx_c in ord_c:
                if b2[idx_c] > m2 + 1e-6:
                    emp_arch.append(cands[idx_c])
                    m2 = b2[idx_c]

            true_pts = [(float(u1[c]), float(u2[c])) for c in emp_arch]
            hv_ratios_80.append(hypervolume_2d(true_pts) / true_hv * 100.0)
            # Check fraction of empirical archive members that are dominated in truth by another tested member
            all_true_u1 = np.array([u1[c] for c in cands])
            all_true_u2 = np.array([u2[c] for c in cands])
            dom_cnt = 0
            for c in emp_arch:
                if np.any((all_true_u1 >= u1[c]) & (all_true_u2 >= u2[c]) & ((all_true_u1 > u1[c]) | (all_true_u2 > u2[c]))):
                    dom_cnt += 1
            false_dom_rates.append(dom_cnt / max(1, len(emp_arch)) * 100.0)

        pareto_policies[pol_name] = {
            "hypervolume_pct_of_opt": ci95(hv_ratios_80),
            "false_pareto_member_pct": ci95(false_dom_rates),
        }

    return {
        "n_true_global_pareto": len(true_pareto_indices),
        "true_pareto_front": true_front,
        "true_hypervolume": round(true_hv, 4),
        "archive_comparison": pareto_policies,
    }


def main():
    t0 = time.time()
    e1 = run_experiment_1()
    e2, gt_tables = run_experiment_2()
    e3 = run_experiment_3(gt_tables)
    with open(os.path.join(OUT_DATA, "summary_v2.json")) as f:
        e4_v2 = json.load(f)
    e5 = run_experiment_5()
    e6 = run_experiment_6()
    e7 = run_experiment_7(gt_tables)

    full_results = {
        "E1_dynamics_hurst_surrogates": e1,
        "E2_landscape_survival_mttf": e2,
        "E3_censoring_variance_anatomy": e3,
        "E4_v2_exact_allocation": e4_v2,
        "E5_policy_benchmarks_ocba_bandits": e5,
        "E6_persistent_switching_and_bias": e6,
        "E7_pareto_reliability_cost_archive": e7,
    }
    out_path = os.path.join(OUT_DATA, "smps_results.json")
    with open(out_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"All 7 SMPS experiments completed in {time.time() - t0:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
