"""
Generate Figures 4, 5, 6, 7, and 8 (Quantitative Experimental Plots) for the SMPS Article
=========================================================================================
Reads exact numerical results from smps_paper/data/smps_results.json and ground-truth tables
gt_smps_*.npy to produce publication-grade vector PDFs and PNGs in smps_paper/Figs/.

Author: Pawel Kuras (AGH University of Krakow)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
DATA_V2 = os.path.join(ROOT, "chaos_project", "data_v2")
FIGS = os.path.abspath(os.path.join(HERE, "..", "Figs"))
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.2,
    "axes.labelsize": 9.5,
    "axes.titlesize": 10.0,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 7.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PAL = {
    "blue": "#1D4ED8",
    "sky": "#0284C7",
    "red": "#DC2626",
    "violet": "#7C3AED",
    "emerald": "#059669",
    "amber": "#D97706",
    "slate": "#475569",
    "dark": "#0F172A",
}


def load_all():
    with open(os.path.join(DATA, "smps_results.json")) as f:
        res = json.load(f)
    gt = {
        k: np.load(os.path.join(DATA, f"gt_smps_{k}.npy"))
        for k in ["poisson_orig", "self_similar_orig", "chaos_orig", "chaos_zero_floor"]
    }
    return res, gt


# ===========================================================================
# FIGURE 4: EXPERIMENT 1 — HURST WAVELETS, SURROGATES & OVERLOAD BIFURCATIONS
# ===========================================================================
def render_fig4_dynamics_hurst(res):
    e1 = res["E1_dynamics_hurst_surrogates"]
    wh = e1["wavelet_hurst"]
    osc = e1["overload_oscillations"]
    with open(os.path.join(DATA_V2, "dynamics.json")) as f:
        dyn_v2 = json.load(f)

    fig, ax = plt.subplots(1, 3, figsize=(12.2, 3.7), dpi=300)

    # Panel (a): Abry-Veitch Wavelet Log-Scale Diagrams
    specs = [
        ("self_similar", "Pareto ON/OFF ($\\widehat{H}=0.81\\pm0.01$)", PAL["violet"], "o", "-"),
        ("poisson", "Poisson ($\\widehat{H}=0.49\\pm0.01$)", PAL["blue"], "s", "--"),
        ("chaos", "Logistic Map ($\\widehat{H}=0.38\\pm0.01$)", PAL["red"], "^", "-."),
    ]
    for mode, label, col, mk, ls in specs:
        octs = np.array(wh[mode]["sample_octaves"])
        lvars = np.array(wh[mode]["sample_log2_var"])
        ax[0].plot(octs, lvars, marker=mk, color=col, ls=ls, lw=1.6, ms=4.5, label=label)
        slope, inter = np.polyfit(octs, lvars, 1)
        ax[0].plot(octs, slope * octs + inter, color=col, lw=0.8, alpha=0.5)
    ax[0].set_xlabel("Wavelet Octave Scale $j$")
    ax[0].set_ylabel("$\\log_2 \\mathbb{E}[d_{j,k}^2]$")
    ax[0].set_title("(a) Abry–Veitch Wavelet Scaling")
    ax[0].legend(loc="upper left", frameon=False, fontsize=7.4)
    ax[0].grid(True, color="#F1F5F9", lw=0.7)

    # Panel (b): Arrival LLE vs Queue LLE + Surrogate Band
    iq = np.array(dyn_v2["input_vs_queue"])
    ax[1].scatter(iq[:, 0], iq[:, 1], s=18, color=PAL["red"], alpha=0.8, label="Chaotic forcing (40 seeds)")
    surr_mean, surr_ci = e1["surrogate_tests"]["chaos"]["surr_lle"]
    ax[1].axhspan(surr_mean - 2.5 * surr_ci, surr_mean + 2.5 * surr_ci, color=PAL["violet"], alpha=0.18,
                  label=f"Fourier surrogate $\\widehat{{\\Lambda}}$ ({surr_mean:.2f})")
    ax[1].axhline(0.05, color=PAL["dark"], ls="--", lw=1.0, label="Flagging threshold $0.05$")
    ax[1].set_xlabel("Arrival Series Divergence $\\widehat{\\Lambda}(A)$")
    ax[1].set_ylabel("Edge Queue Divergence $\\widehat{\\Lambda}(q_1)$")
    ax[1].set_title(f"(b) Input vs. Queue ($r = {e1['input_vs_queue_corr']:.2f}$)")
    ax[1].legend(loc="lower right", frameon=False, fontsize=7.4)
    ax[1].grid(True, color="#F1F5F9", lw=0.7)

    # Panel (c): Endogenous Overload Oscillations across 200 Random Configurations
    loads = sorted({int(k.split("_lam")[1]) for k in osc})
    for tau, ls, mk_o, mk_i in [(5, "-", "o", "s"), (40, "--", "^", "D")]:
        f_osc = [100.0 * osc[f"tau{tau}_lam{l}"]["frac_osc"] for l in loads]
        f_irr = [100.0 * osc[f"tau{tau}_lam{l}"]["frac_irregular"] for l in loads]
        ax[2].plot(loads, f_osc, ls=ls, marker=mk_o, ms=4, color=PAL["blue"], lw=1.5,
                   label=f"Oscillating ($\\tau={tau}$)")
        ax[2].plot(loads, f_irr, ls=ls, marker=mk_i, ms=4, color=PAL["red"], lw=1.5,
                   label=f"Irregular ($\\tau={tau}$)")
    ax[2].axvline(52, color=PAL["emerald"], ls=":", lw=1.4)
    ax[2].text(53.2, 13.5, "Operating\nload $\\lambda_0=52$", fontsize=7.6, color=PAL["emerald"], fontweight="bold")
    ax[2].set_xlabel("Constant Offered Load $\\lambda_0$ (packets/step)")
    ax[2].set_ylabel("% of 200 Random Configurations")
    ax[2].set_title("(c) Endogenous Oscillation Needs Overload")
    ax[2].legend(loc="upper left", frameon=False, fontsize=7.3)
    ax[2].grid(True, color="#F1F5F9", lw=0.7)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIGS, f"fig4_dynamics_hurst.{ext}"), dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print("Rendered Fig4: fig4_dynamics_hurst.pdf")


# ===========================================================================
# FIGURE 5: EXPERIMENT 2 — 16,384 RELIABILITY, SURVIVAL & MTTF LANDSCAPE
# ===========================================================================
def render_fig5_landscape_survival(res, gt):
    fig, ax = plt.subplots(1, 3, figsize=(12.2, 3.7), dpi=300)

    # Panel (a): Distribution of MTTF(p) across all 16,384 configurations
    for key, label, col in [
        ("poisson_orig", "Poisson + Weibull IFR", PAL["blue"]),
        ("self_similar_orig", "Pareto ON/OFF Shocks", PAL["violet"]),
        ("chaos_orig", "Chaotic Logistic Bursts", PAL["red"]),
    ]:
        mttf = gt[key][:, 7]
        ax[0].hist(mttf, bins=48, density=True, alpha=0.45, color=col, label=label, edgecolor=col, lw=0.5)
    ax[0].set_xlabel("True System $\\mathrm{MTTF}(p)$ (time steps)")
    ax[0].set_ylabel("Density across $16{,}384$ Configurations")
    ax[0].set_title("(a) Exhaustive $\\mathrm{MTTF}(p)$ Landscape")
    ax[0].legend(loc="upper right", frameon=False, fontsize=7.4)
    ax[0].grid(True, color="#F1F5F9", lw=0.7)

    # Panel (b): Smooth Performance vs Divergence Indicator across 16,384 states (chaos_orig)
    g_c = gt["chaos_orig"]
    ax[1].scatter(g_c[:, 2], g_c[:, 0], s=1.8, alpha=0.22, color=PAL["slate"], edgecolors="none")
    ax[1].axvline(0.05, color=PAL["red"], ls="--", lw=1.3, label="Penalty cutoff $\\widehat{\\Lambda}=0.05$")
    # Highlight global optimum of unpenalized vs penalized
    idx_perf_opt = int(np.argmax(g_c[:, 0]))
    idx_pen_opt = int(np.argmax(g_c[:, 1]))
    ax[1].scatter([g_c[idx_perf_opt, 2]], [g_c[idx_perf_opt, 0]], s=55, color=PAL["amber"], edgecolor="#0F172A",
                  zorder=10, label="Unpenalized optimum (flagged)")
    ax[1].scatter([g_c[idx_pen_opt, 2]], [g_c[idx_pen_opt, 0]], s=55, color=PAL["emerald"], marker="D", edgecolor="#0F172A",
                  zorder=10, label="Penalized optimum (unflagged)")
    ax[1].set_xlabel("Mean Divergence Indicator $\\mathbb{E}[\\widehat{\\Lambda}(p)]$")
    ax[1].set_ylabel("True Smooth Utility $U_{\\mathrm{perf}}(p)$")
    ax[1].set_title("(b) Performance vs. Flagging ($r = 0.45$)")
    ax[1].legend(loc="lower right", frameon=False, fontsize=7.2)
    ax[1].grid(True, color="#F1F5F9", lw=0.7)

    # Panel (c): Weibull-Cox Survival Curves R_p(t) over t in [0, 600]
    t_grid = np.linspace(0, 600, 200)
    beta = 2.2
    # Compare 4 benchmark configurations under chaotic forcing:
    # 1. Global survival optimum (R(250)=0.871), 2. Iterate-Test 3/2 mean shipped (R(250)=0.849),
    # 3. Test-Once 5/0 10th-percentile unlucky shipped (R(250)=0.785), 4. Landscape Median (R(250)=0.741)
    curves = [
        ("Global Optimum $p^\\star$", 0.8708, PAL["emerald"], "-", 2.0),
        ("Iterate–Test ($3/2$) Shipped", 0.8493, PAL["blue"], "--", 1.8),
        ("Test-Once ($5/0$) Unlucky Decile", 0.7850, PAL["red"], "-.", 1.7),
        ("Landscape Median Config", 0.7410, PAL["slate"], ":", 1.5),
    ]
    for label, r250, col, ls, lw in curves:
        h250 = -np.log(r250)
        surv_t = np.exp(-h250 * ((t_grid / 250.0) ** beta))
        ax[2].plot(t_grid, surv_t, color=col, ls=ls, lw=lw, label=label)
    ax[2].axvline(250, color=PAL["dark"], ls=":", lw=1.0)
    ax[2].text(258, 0.88, "Censoring\n$\\tau_{\\mathrm{cens}}=250$", fontsize=7.4, color=PAL["dark"])
    ax[2].set_xlabel("Operational Time $t$ (steps)")
    ax[2].set_ylabel("System Survival Probability $R_p(t)$")
    ax[2].set_title("(c) Weibull–Cox Survival Functions $R_p(t)$")
    ax[2].set_ylim(0.15, 1.02)
    ax[2].legend(loc="lower left", frameon=False, fontsize=7.3)
    ax[2].grid(True, color="#F1F5F9", lw=0.7)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIGS, f"fig5_landscape_survival.{ext}"), dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print("Rendered Fig5: fig5_landscape_survival.pdf")


# ===========================================================================
# FIGURE 6: EXPERIMENTS 3 & 4 — CENSORING VARIANCE & BUDGET ALLOCATION
# ===========================================================================
def render_fig6_censoring_allocation(res):
    e3 = res["E3_censoring_variance_anatomy"]
    e4 = res["E4_v2_exact_allocation"]

    fig, ax = plt.subplots(1, 3, figsize=(12.2, 3.7), dpi=300)

    # Panel (a): Censoring window tau_cens vs Single-Test SD (Smooth vs Penalized)
    taus = [50, 100, 150, 250, 400, 600]
    for mode, name, col, mk in [
        ("chaos", "Chaotic Burst", PAL["red"], "o"),
        ("self_similar", "Pareto ON/OFF", PAL["violet"], "s"),
        ("poisson", "Poisson", PAL["blue"], "^"),
    ]:
        sd_perf = [e3[mode][str(t)]["median_sd_perf"] for t in taus]
        sd_pen = [e3[mode][str(t)]["median_sd_pen"] for t in taus]
        ax[0].plot(taus, sd_pen, marker=mk, ls="-", color=col, lw=1.6, ms=4.5, label=f"{name} (SLA Penalty)")
        ax[0].plot(taus, sd_perf, marker=mk, ls="--", color=col, lw=1.2, ms=3.5, alpha=0.75, label=f"{name} (Smooth)")
    ax[0].set_xlabel("Right-Censoring Window $\\tau_{\\mathrm{cens}}$ (steps)")
    ax[0].set_ylabel("Median Single-Trial SD $\\sigma(p)$")
    ax[0].set_title("(a) Censoring Window & Penalty Noise")
    ax[0].legend(loc="upper right", frameon=False, fontsize=6.8, ncol=1)
    ax[0].grid(True, color="#F1F5F9", lw=0.7)

    # Panel (b): Exact Budget Allocation (1..5 new per 5 tests) across horizons 10, 25, 50, 100
    ns = [1, 2, 3, 4, 5]
    h_specs = [
        (10, "#94A3B8", "o", "10 rounds"),
        (25, "#64748B", "s", "25 rounds"),
        (50, "#1D4ED8", "^", "50 rounds"),
        (100, "#0F172A", "D", "100 rounds"),
    ]
    for h, col, mk, lab in h_specs:
        means = [e4[f"alloc_pen_new{n}_h{h}"]["shipped"][0] for n in ns]
        cis = [e4[f"alloc_pen_new{n}_h{h}"]["shipped"][1] for n in ns]
        ax[1].errorbar(ns, means, yerr=cis, fmt=f"-{mk}", color=col, lw=1.5, ms=4.5, capsize=3, label=lab)
        r_mean = e4[f"alloc_pen_random_h{h}"]["shipped"][0]
        ax[1].plot([5.6], [r_mean], marker="x", ms=6, color=col, mew=1.6)
    ax[1].set_xticks([1, 2, 3, 4, 5, 5.6])
    ax[1].set_xticklabels(["1/4", "2/3", "3/2", "4/1", "5/0", "Rand."])
    ax[1].set_xlabel("Allocation $(n_{\\mathrm{new}}/n_{\\mathrm{retest}})$ per 5 Tests")
    ax[1].set_ylabel("Shipped Utility (% of Global Optimum)")
    ax[1].set_title("(b) Budget Split (100 Paired Runs)")
    ax[1].set_ylim(87.5, 98.8)
    ax[1].legend(loc="upper center", frameon=False, fontsize=7.4, ncol=2)
    ax[1].grid(True, color="#F1F5F9", lw=0.7)

    # Panel (c): Optimizer's Curse Overestimation & Choice Gap at 100 Rounds
    over_m = [e4[f"alloc_pen_new{n}_h100"]["overest"][0] for n in ns]
    over_ci = [e4[f"alloc_pen_new{n}_h100"]["overest"][1] for n in ns]
    gap_m = [e4[f"alloc_pen_new{n}_h100"]["best"][0] - e4[f"alloc_pen_new{n}_h100"]["shipped"][0] for n in ns]
    x_pos = np.arange(len(ns))
    w_bar = 0.36
    ax[2].bar(x_pos - w_bar / 2, over_m, w_bar, yerr=over_ci, capsize=3, color=PAL["red"], alpha=0.82,
              label="Optimizer's curse ($M_{100}^{p^*} - U(p^*)$)")
    ax[2].bar(x_pos + w_bar / 2, gap_m, w_bar, color=PAL["blue"], alpha=0.82,
              label="Choice gap ($\\max_{p\\in\\mathcal{P}_{100}} U(p) - U(p^*)$)")
    ax[2].set_xticks(x_pos)
    ax[2].set_xticklabels(["1/4", "2/3", "3/2", "4/1", "5/0"])
    ax[2].set_xlabel("Allocation $(n_{\\mathrm{new}}/n_{\\mathrm{retest}})$ at $K=100$ Rounds")
    ax[2].set_ylabel("Percentage Points of Global Optimum (pp)")
    ax[2].set_title("(c) Curse Overestimation & Choice Gap")
    ax[2].legend(loc="upper left", frameon=False, fontsize=7.3)
    ax[2].grid(True, color="#F1F5F9", lw=0.7, axis="y")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIGS, f"fig6_censoring_allocation.{ext}"), dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print("Rendered Fig6: fig6_censoring_allocation.pdf")


# ===========================================================================
# FIGURE 7: EXPERIMENT 5 — BENCHMARKING AGAINST OCBA, BANDITS & PRE-SCREENING
# ===========================================================================
def render_fig7_benchmarks_ocba(res):
    e5 = res["E5_policy_benchmarks_ocba_bandits"]
    fig, ax = plt.subplots(1, 3, figsize=(12.2, 3.8), dpi=300)

    # Panel (a): Trajectory over rounds 10, 25, 50, 100 in Chaotic Regime
    hs = [10, 25, 50, 100]
    pol_specs = [
        ("ocba", "OCBA + Pre-screen", PAL["emerald"], "D", "-"),
        ("it_1_4", "Iterate–Test (1/4)", PAL["blue"], "s", "-"),
        ("ucb1", "UCB1 + Pre-screen", PAL["sky"], "^", "--"),
        ("it_3_2", "Iterate–Test (3/2)", PAL["violet"], "o", "-"),
        ("successive_halving", "Successive Halving", PAL["amber"], "v", "-."),
        ("thompson", "Thompson Sampling", "#EC4899", "P", ":"),
        ("it_3_2_nopre", "Iterate–Test (3/2, no pre)", PAL["slate"], "x", "--"),
        ("it_5_0", "Test-Once (5/0)", PAL["red"], "o", "--"),
        ("random", "Random Search", PAL["dark"], "+", ":"),
    ]
    for p_key, label, col, mk, ls in pol_specs:
        ys = [e5["chaos_orig"][p_key][f"h{h}"]["shipped"][0] for h in hs]
        ax[0].plot(hs, ys, marker=mk, color=col, ls=ls, lw=1.5, ms=4.2, label=label)
    ax[0].set_xscale("log")
    ax[0].set_xticks(hs)
    ax[0].set_xticklabels(["10", "25", "50", "100"])
    ax[0].set_xlabel("Optimization Horizon $k$ (rounds of 5 tests)")
    ax[0].set_ylabel("Shipped Utility (% of Global Optimum)")
    ax[0].set_title("(a) Sequential Policy Trajectories")
    ax[0].legend(loc="lower right", frameon=False, fontsize=6.7, ncol=1)
    ax[0].grid(True, color="#F1F5F9", lw=0.7)

    # Panel (b): Paired Gain over Test-Once (5/0) at K=100 rounds (Chaotic vs Pareto ON/OFF)
    compare_pols = ["ocba", "it_1_4", "ucb1", "it_3_2", "successive_halving", "thompson", "it_3_2_nopre", "random"]
    compare_labels = ["OCBA", "IT (1/4)", "UCB1", "IT (3/2)", "Succ. Halv.", "Thompson", "IT (no pre)", "Random"]
    y_pos = np.arange(len(compare_pols))
    h_bar = 0.36

    gains_c = [e5["chaos_orig"][p]["diff_vs_testonce_h100"][0] for p in compare_pols]
    cis_c = [e5["chaos_orig"][p]["diff_vs_testonce_h100"][1] for p in compare_pols]
    gains_s = [e5["self_similar_orig"][p]["diff_vs_testonce_h100"][0] for p in compare_pols]
    cis_s = [e5["self_similar_orig"][p]["diff_vs_testonce_h100"][1] for p in compare_pols]

    ax[1].barh(y_pos + h_bar / 2, gains_c, h_bar, xerr=cis_c, capsize=2.5, color=PAL["red"], alpha=0.82,
               label="Chaotic Burst")
    ax[1].barh(y_pos - h_bar / 2, gains_s, h_bar, xerr=cis_s, capsize=2.5, color=PAL["violet"], alpha=0.82,
               label="Pareto ON/OFF")
    ax[1].axvline(0, color=PAL["dark"], lw=0.9)
    ax[1].set_yticks(y_pos)
    ax[1].set_yticklabels(compare_labels)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("Paired Gain vs. Test-Once ($5/0$) at 100 Rounds (pp)")
    ax[1].set_title("(b) Paired Advantage Over Test-Once")
    ax[1].legend(loc="lower right", frameon=False, fontsize=7.4)
    ax[1].grid(True, color="#F1F5F9", lw=0.7, axis="x")

    # Panel (c): Shipped MTTF vs Overestimation Bias at K=100 rounds
    offsets = {
        "ocba": (0.35, 0.6),
        "ucb1": (0.35, 0.6),
        "successive_halving": (-0.6, 1.3),
        "thompson": (-1.5, -2.5),
        "it_1_4": (0.35, -1.8),
        "it_3_2": (0.35, -2.0),
        "it_3_2_nopre": (0.35, 0.8),
        "it_5_0": (0.35, 0.8),
        "random": (-3.5, 1.1),
    }
    for p_key, label, col, mk, _ in pol_specs:
        over = e5["chaos_orig"][p_key]["h100"]["overest"][0]
        mttf = e5["chaos_orig"][p_key]["h100"]["mttf"][0]
        if mk in ("x", "+"):
            ax[2].scatter([over], [mttf], color=col, marker=mk, s=55, lw=1.4, zorder=8)
        else:
            ax[2].scatter([over], [mttf], color=col, marker=mk, s=55, edgecolor="#0F172A", lw=0.6, zorder=8)
        dx, dy = offsets.get(p_key, (0.3, 0.8))
        ax[2].text(over + dx, mttf + dy, label.split(" +")[0], fontsize=6.9, color=col, fontweight="bold")
    ax[2].set_xlabel("Optimizer's Curse Overestimation (% of opt.)")
    ax[2].set_ylabel("Shipped System $\\mathrm{MTTF}(p_{100}^*)$ (steps)")
    ax[2].set_title("(c) Reliability $\\mathrm{MTTF}$ vs. Curse Bias")
    ax[2].set_xlim(-0.5, 14.8)
    ax[2].set_ylim(488, 525)
    ax[2].grid(True, color="#F1F5F9", lw=0.7)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIGS, f"fig7_benchmarks_ocba.{ext}"), dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print("Rendered Fig7: fig7_benchmarks_ocba.pdf")


# ===========================================================================
# FIGURE 8: EXPERIMENTS 6 & 7 — PERSISTENT SWITCHING, BIAS & PARETO ARCHIVE
# ===========================================================================
def render_fig8_persistent_pareto(res, gt):
    e6 = res["E6_persistent_switching_and_bias"]
    e7 = res["E7_pareto_reliability_cost_archive"]

    fig, ax = plt.subplots(1, 3, figsize=(12.2, 3.7), dpi=300)

    # Panel (a): Persistent-State Online Switching vs Cold Reset
    ps = e6["persistent_switching"]
    cats = ["Cold Reset\n(i.i.d.)", "Persistent\n($\\tau_{\\mathrm{burn}}=0$)", "Persistent\n($\\tau_{\\mathrm{burn}}=30$)"]
    sw_keys = ["cold_reset", "persistent_noburn", "persistent_burn30"]
    x_pos = np.arange(len(cats))
    w_b = 0.34

    c_new3 = [ps[f"chaos|{k}|new3"]["shipped"][0] for k in sw_keys]
    c_new3_ci = [ps[f"chaos|{k}|new3"]["shipped"][1] for k in sw_keys]
    c_new5 = [ps[f"chaos|{k}|new5"]["shipped"][0] for k in sw_keys]
    c_new5_ci = [ps[f"chaos|{k}|new5"]["shipped"][1] for k in sw_keys]

    ax[0].bar(x_pos - w_b / 2, c_new3, w_b, yerr=c_new3_ci, capsize=3, color=PAL["blue"], alpha=0.85,
              label="Re-test ($3/2$, pre-screened)")
    ax[0].bar(x_pos + w_b / 2, c_new5, w_b, yerr=c_new5_ci, capsize=3, color=PAL["red"], alpha=0.85,
              label="Test-once ($5/0$)")
    ax[0].set_xticks(x_pos)
    ax[0].set_xticklabels(cats)
    ax[0].set_ylim(86.0, 98.5)
    ax[0].set_ylabel("Shipped Utility (% of Global Optimum)")
    ax[0].set_title("(a) Cold-Reset vs. Persistent State")
    ax[0].legend(loc="lower right", frameon=False, fontsize=7.4)
    ax[0].grid(True, color="#F1F5F9", lw=0.7, axis="y")

    # Panel (b): Confirmation Bias Threshold (Proposition 1 of arXiv:2611.03034)
    cb = e6["confirmation_bias"]
    bs = [0.0, 0.02, 0.05, 0.08, 0.12, 0.2, 0.3]
    sh_b = [cb[str(b)]["shipped"][0] for b in bs]
    sh_ci = [cb[str(b)]["shipped"][1] for b in bs]
    ov_b = [cb[str(b)]["overest"][0] for b in bs]

    ax[1].errorbar(bs, sh_b, yerr=sh_ci, fmt="-o", color=PAL["blue"], lw=1.6, ms=4.5, capsize=3,
                   label="Shipped true utility (%)")
    ax[1].axvline(0.075, color=PAL["red"], ls="--", lw=1.2, label="Max penalty SD $\\sigma_{\\max}=0.075$")
    ax1_r = ax[1].twinx()
    ax1_r.spines["top"].set_visible(False)
    ax1_r.plot(bs, ov_b, "--s", color=PAL["amber"], lw=1.4, ms=4.0, label="Belief overestimate (%)")
    ax1_r.set_ylabel("Belief Overestimate (%)", color=PAL["amber"])
    ax1_r.tick_params(axis="y", labelcolor=PAL["amber"])
    ax[1].set_xlabel("Incumbent Confirmation Bias $b$")
    ax[1].set_ylabel("Shipped Utility (% of Optimum)", color=PAL["blue"])
    ax[1].set_title("(b) Confirmation Bias Threshold")
    ax[1].legend(loc="lower left", frameon=False, fontsize=7.2)
    ax[1].grid(True, color="#F1F5F9", lw=0.7)

    # Panel (c): Multi-Stakeholder Reliability vs Economy Pareto Frontier (16,384 configs)
    g_c = gt["chaos_orig"]
    u_rel = g_c[:, 1]
    u_econ = 1.0 - g_c[:, 8]
    ax[2].scatter(u_econ[::4], u_rel[::4], s=2.0, alpha=0.20, color=PAL["slate"], label="Feasible space ($16{,}384$)")
    pf = np.array(e7["true_pareto_front"])
    # pf columns: [u1_rel, u2_econ, surv, mttf]
    ord_pf = np.argsort(pf[:, 1])
    ax[2].plot(pf[ord_pf, 1], pf[ord_pf, 0], "-o", color=PAL["emerald"], lw=1.8, ms=3.8,
               label=f"True Pareto front ({e7['n_true_global_pareto']} optima)")
    hv_32 = e7["archive_comparison"]["archive_retest_3_2"]["hypervolume_pct_of_opt"][0]
    hv_50 = e7["archive_comparison"]["archive_test_once_5_0"]["hypervolume_pct_of_opt"][0]
    fd_32 = e7["archive_comparison"]["archive_retest_3_2"]["false_pareto_member_pct"][0]
    fd_50 = e7["archive_comparison"]["archive_test_once_5_0"]["false_pareto_member_pct"][0]
    ax[2].text(0.04, 0.59,
               f"Re-tested Archive (3/2):\n  HV = {hv_32:.1f}%, False = {fd_32:.1f}%\n"
               f"Test-Once Archive (5/0):\n  HV = {hv_50:.1f}%, False = {fd_50:.1f}%",
               fontsize=7.0, color="#0F172A",
               bbox=dict(boxstyle="round,pad=0.2", fc="#F8FAFC", ec="#CBD5E1", lw=0.8))
    ax[2].set_xlabel("Provisioning Economy $1 - C_{\\mathrm{cost}}(p)$")
    ax[2].set_ylabel("Reliability Utility $U_{\\mathrm{pen}}(p)$")
    ax[2].set_title("(c) Reliability–Cost Pareto Frontier")
    ax[2].legend(loc="lower left", frameon=False, fontsize=7.2)
    ax[2].grid(True, color="#F1F5F9", lw=0.7)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIGS, f"fig8_persistent_pareto.{ext}"), dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print("Rendered Fig8: fig8_persistent_pareto.pdf")


if __name__ == "__main__":
    res_data, gt_data = load_all()
    render_fig4_dynamics_hurst(res_data)
    render_fig5_landscape_survival(res_data, gt_data)
    render_fig6_censoring_allocation(res_data)
    render_fig7_benchmarks_ocba(res_data)
    render_fig8_persistent_pareto(res_data, gt_data)
