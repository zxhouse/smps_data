#!/usr/bin/env python3
"""
Automated numerical benchmark and consistency verifier for data/smps_results.json.
Verifies all 39 quantitative benchmark results across Experiments E1-E7 (and also
checks the LaTeX manuscript sections if present locally).
"""
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data", "smps_results.json")

EXPECTED = {
    "E1 Pareto Hurst A": "0.807",
    "E1 Poisson Hurst A": "0.494",
    "E1 Chaos Hurst A": "0.384",
    "E1 Chaos K01 A": "0.962",
    "E2 Poisson opt U_perf": "0.9994",
    "E2 Poisson opt U_pen": "0.9994",
    "E2 Pareto opt U_pen": "0.9830",
    "E2 Chaos opt U_perf": "0.9740",
    "E2 Chaos opt U_pen": "0.9269",
    "E2 Chaos median MTTF": "395.6",
    "E2 Chaos opt MTTF": "544.1",
    "E2 Chaos sep violations": "871",
    "E3 Chaos tau=250 sd_perf": "0.0060",
    "E3 Chaos tau=250 sd_pen": "0.0239",
    "E3 Chaos tau=250 ratio": "4.00",
    "E3 Pareto tau=250 ratio": "1.81",
    "E4 3/2 shipped h100": "95.6",
    "E4 5/0 shipped h100": "91.6",
    "E4 3/2 overest h100": "2.2",
    "E4 5/0 overest h100": "9.0",
    "E5 OCBA shipped": "98.20",
    "E5 IT 1/4 shipped": "97.83",
    "E5 UCB1 shipped": "97.36",
    "E5 IT 3/2 shipped": "97.19",
    "E5 Succ Halving shipped": "96.90",
    "E5 Thompson shipped": "95.59",
    "E5 IT 3/2 no pre shipped": "95.44",
    "E5 Test-Once 5/0 shipped": "92.39",
    "E6 Chaos cold 3/2": "94.55",
    "E6 Chaos cold 5/0": "91.57",
    "E6 Chaos noburn 3/2": "94.01",
    "E6 Chaos noburn 5/0": "90.35",
    "E6 Chaos burn30 3/2": "95.15",
    "E6 Chaos burn30 5/0": "90.62",
    "E7 True Pareto count": "47",
    "E7 3/2 HV pct": "89.14",
    "E7 5/0 HV pct": "86.06",
    "E7 3/2 False pct": "56.19",
    "E7 5/0 False pct": "74.45",
}


def main():
    with open(DATA, "r") as f:
        res = json.load(f)

    tex_text = ""
    for fn in ["SMPS_manuscript.tex", "sec1_intro.tex", "sec2_model.tex",
               "sec3_theory.tex", "sec4_experiments.tex", "sec5_discussion.tex"]:
        p = os.path.join(BASE, fn)
        if os.path.exists(p):
            with open(p, "r") as f:
                tex_text += "\n" + f.read()

    has_tex = len(tex_text) > 0
    checks = []

    def check_val(label, val_str):
        exp = EXPECTED[label]
        ok_json = (val_str == exp)
        ok_tex = (val_str in tex_text) if has_tex else True
        found = ok_json and ok_tex
        checks.append((label, val_str, found))
        if not found:
            print(f"[FAIL] {label}: got '{val_str}' (expected '{exp}')")
        else:
            print(f"[PASS] {label}: '{val_str}' verified")

    # E1 checks
    e1 = res["E1_dynamics_hurst_surrogates"]
    check_val("E1 Pareto Hurst A", f"{e1['wavelet_hurst']['self_similar']['H_hat'][0]+1e-9:.3f}")
    check_val("E1 Poisson Hurst A", f"{e1['wavelet_hurst']['poisson']['H_hat'][0]+1e-9:.3f}")
    check_val("E1 Chaos Hurst A", f"{e1['wavelet_hurst']['chaos']['H_hat'][0]+1e-9:.3f}")
    check_val("E1 Chaos K01 A", f"{e1['wavelet_hurst']['chaos']['K_01_arrival'][0]+1e-9:.3f}")

    # E2 checks
    e2 = res["E2_landscape_survival_mttf"]
    check_val("E2 Poisson opt U_perf", f"{e2['poisson_orig']['opt_perf']:.4f}")
    check_val("E2 Poisson opt U_pen", f"{e2['poisson_orig']['opt_pen']:.4f}")
    check_val("E2 Pareto opt U_pen", f"{e2['self_similar_orig']['opt_pen']:.4f}")
    check_val("E2 Chaos opt U_perf", f"{e2['chaos_orig']['opt_perf']:.4f}")
    check_val("E2 Chaos opt U_pen", f"{e2['chaos_orig']['opt_pen']:.4f}")
    check_val("E2 Chaos median MTTF", f"{e2['chaos_orig']['median_mttf']:.1f}")
    check_val("E2 Chaos opt MTTF", f"{e2['chaos_orig']['opt_mttf']:.1f}")
    check_val("E2 Chaos sep violations", str(e2["separation"]["chaos_orig_pen"]["n_sep_violations"]))

    # E3 checks
    e3 = res["E3_censoring_variance_anatomy"]
    check_val("E3 Chaos tau=250 sd_perf", f"{e3['chaos']['250']['median_sd_perf']:.4f}")
    check_val("E3 Chaos tau=250 sd_pen", f"{e3['chaos']['250']['median_sd_pen']:.4f}")
    check_val("E3 Chaos tau=250 ratio", f"{e3['chaos']['250']['variance_inflation_ratio']:.2f}")
    check_val("E3 Pareto tau=250 ratio", f"{e3['self_similar']['250']['variance_inflation_ratio']:.2f}")

    # E4 checks
    e4 = res["E4_v2_exact_allocation"]
    check_val("E4 3/2 shipped h100", f"{e4['alloc_pen_new3_h100']['shipped'][0]:.1f}")
    check_val("E4 5/0 shipped h100", f"{e4['alloc_pen_new5_h100']['shipped'][0]:.1f}")
    check_val("E4 3/2 overest h100", f"{e4['alloc_pen_new3_h100']['overest'][0]:.1f}")
    check_val("E4 5/0 overest h100", f"{e4['alloc_pen_new5_h100']['overest'][0]:.1f}")

    # E5 checks
    e5 = res["E5_policy_benchmarks_ocba_bandits"]["chaos_orig"]
    check_val("E5 OCBA shipped", f"{e5['ocba']['h100']['shipped'][0]:.2f}")
    check_val("E5 IT 1/4 shipped", f"{e5['it_1_4']['h100']['shipped'][0]:.2f}")
    check_val("E5 UCB1 shipped", f"{e5['ucb1']['h100']['shipped'][0]:.2f}")
    check_val("E5 IT 3/2 shipped", f"{e5['it_3_2']['h100']['shipped'][0]:.2f}")
    check_val("E5 Succ Halving shipped", f"{e5['successive_halving']['h100']['shipped'][0]:.2f}")
    check_val("E5 Thompson shipped", f"{e5['thompson']['h100']['shipped'][0]:.2f}")
    check_val("E5 IT 3/2 no pre shipped", f"{e5['it_3_2_nopre']['h100']['shipped'][0]:.2f}")
    check_val("E5 Test-Once 5/0 shipped", f"{e5['it_5_0']['h100']['shipped'][0]:.2f}")

    # E6 checks
    e6 = res["E6_persistent_switching_and_bias"]
    ps = e6["persistent_switching"]
    check_val("E6 Chaos cold 3/2", f"{ps['chaos|cold_reset|new3']['shipped'][0]:.2f}")
    check_val("E6 Chaos cold 5/0", f"{ps['chaos|cold_reset|new5']['shipped'][0]:.2f}")
    check_val("E6 Chaos noburn 3/2", f"{ps['chaos|persistent_noburn|new3']['shipped'][0]:.2f}")
    check_val("E6 Chaos noburn 5/0", f"{ps['chaos|persistent_noburn|new5']['shipped'][0]:.2f}")
    check_val("E6 Chaos burn30 3/2", f"{ps['chaos|persistent_burn30|new3']['shipped'][0]:.2f}")
    check_val("E6 Chaos burn30 5/0", f"{ps['chaos|persistent_burn30|new5']['shipped'][0]:.2f}")

    # E7 checks
    e7 = res["E7_pareto_reliability_cost_archive"]
    check_val("E7 True Pareto count", str(e7["n_true_global_pareto"]))
    check_val("E7 3/2 HV pct", f"{e7['archive_comparison']['archive_retest_3_2']['hypervolume_pct_of_opt'][0]:.2f}")
    check_val("E7 5/0 HV pct", f"{e7['archive_comparison']['archive_test_once_5_0']['hypervolume_pct_of_opt'][0]:.2f}")
    check_val("E7 3/2 False pct", f"{e7['archive_comparison']['archive_retest_3_2']['false_pareto_member_pct'][0]:.2f}")
    check_val("E7 5/0 False pct", f"{e7['archive_comparison']['archive_test_once_5_0']['false_pareto_member_pct'][0]:.2f}")

    n_pass = sum(1 for _, _, ok in checks if ok)
    print(f"\nSummary: {n_pass}/{len(checks)} quantitative checks passed.")
    if n_pass != len(checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
