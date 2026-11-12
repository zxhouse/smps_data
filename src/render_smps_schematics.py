"""
Render High-Precision Vector Schematics (Figures 1, 2, and 3) for the SMPS Article
==================================================================================
1. Fig1 (fig1_network_architecture.pdf): Ideational Network & Reliability Hardware Schematic
   with realistic switches, client workstations/IoT gateways, redundant server racks, and SDN plane.
2. Fig2 (fig2_iterate_test_pipeline.pdf): Two-Level Iterate-Test Core & Right-Censored Life-Testing
   Workflow Schematic (arXiv:2611.03034 applied to Stochastic Reliability Engineering).
3. Fig3 (fig3_experimental_protocol.pdf): End-to-End 7-Experiment Validation & Ground-Truth
   Evaluation Protocol Schematic.

All arrows use explicit non-overlapping orthogonal or smooth Bezier paths with verified margins.
Author: Pawel Kuras (AGH University of Krakow)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch, Polygon
from matplotlib.path import Path

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.abspath(os.path.join(HERE, "..", "Figs"))
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.5,
    "axes.labelsize": 10,
    "axes.titlesize": 10.5,
    "figure.autolayout": False,
})


def _add_card(ax, x, y, w, h, fc="#FFFFFF", ec="#334155", lw=1.4, radius=0.10, shadow=True, z=3):
    """Draw a rounded card centered at (x, y) with width w and height h."""
    if shadow:
        ax.add_patch(FancyBboxPatch(
            (x - w / 2 + 0.05, y - h / 2 - 0.05), w, h,
            boxstyle=f"round,pad=0.02,rounding_size={radius}",
            facecolor="#0F172A", edgecolor="none", alpha=0.07, zorder=z - 1
        ))
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z
    )
    ax.add_patch(patch)
    return patch


def _ortho_arrow(ax, pts, col="#1E293B", lw=1.8, ls="-", style="-|>", ms=13, z=15):
    """Draw an orthogonal polyline arrow through a list of (x, y) vertices."""
    codes = [Path.MOVETO] + [Path.LINETO] * (len(pts) - 1)
    patch = FancyArrowPatch(
        path=Path(pts, codes), arrowstyle=style, mutation_scale=ms,
        color=col, linewidth=lw, linestyle=ls, shrinkA=0, shrinkB=0, zorder=z
    )
    ax.add_patch(patch)
    return patch


def _straight_arrow(ax, p1, p2, col="#1E293B", lw=1.8, ls="-", style="-|>", ms=13, rad=0.0, z=15):
    """Draw a direct or gently curved arrow from p1 to p2."""
    patch = FancyArrowPatch(
        p1, p2, arrowstyle=style, mutation_scale=ms,
        color=col, linewidth=lw, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2, zorder=z
    )
    ax.add_patch(patch)
    return patch


# ===========================================================================
# FIGURE 1: IDEATIONAL HARDWARE & STOCHASTIC RELIABILITY NETWORK ARCHITECTURE
# ===========================================================================
def render_fig1_network_architecture():
    fig, ax = plt.subplots(figsize=(13.6, 8.4), dpi=300)
    ax.set_xlim(0.0, 16.0)
    ax.set_ylim(0.0, 10.2)
    ax.axis("off")

    # Subtle engineering blueprint grid
    for gx in np.linspace(0.5, 15.5, 31):
        ax.axvline(gx, color="#F1F5F9", lw=0.5, zorder=0)
    for gy in np.linspace(0.5, 9.7, 20):
        ax.axhline(gy, color="#F1F5F9", lw=0.5, zorder=0)

    # --- Helper 1: Ideational Workstation / Industrial IoT Traffic Generator ---
    def draw_workstation_node(x, y, title, subtitle, formula, wave_type="poisson", accent="#0284C7"):
        w, h = 2.75, 2.05
        _add_card(ax, x, y, w, h, fc="#FFFFFF", ec=accent, lw=1.6, z=3)
        # Header banner
        ax.add_patch(FancyBboxPatch(
            (x - w / 2 + 0.03, y + h / 2 - 0.36), w - 0.06, 0.33,
            boxstyle="round,pad=0.01,rounding_size=0.06",
            facecolor=accent, edgecolor="none", zorder=4
        ))
        ax.text(x, y + h / 2 - 0.195, title, ha="center", va="center",
                fontsize=8.6, fontweight="bold", color="#FFFFFF", zorder=5)

        # Monitor frame (left side of card)
        mx, my, mw, mh = x - 1.15, y - 0.28, 1.25, 0.78
        ax.add_patch(FancyBboxPatch(
            (mx, my), mw, mh, boxstyle="round,pad=0.01,rounding_size=0.04",
            facecolor="#1E293B", edgecolor="#0F172A", lw=1.0, zorder=4
        ))
        ax.add_patch(Rectangle((mx + 0.06, my + 0.07), mw - 0.12, mh - 0.14,
                               facecolor="#F0F9FF", edgecolor=accent, lw=0.7, zorder=5))
        # Waveform inside screen
        tx = np.linspace(mx + 0.10, mx + mw - 0.10, 28)
        if wave_type == "chaos":
            ty = my + 0.39 + 0.22 * np.sin(3.7 * np.linspace(0, 4 * np.pi, 28)) * np.cos(1.9 * np.linspace(0, 4 * np.pi, 28))
        elif wave_type == "pareto":
            spikes = np.array([0.0, 0.05, 0.28, 0.25, -0.1, -0.08, 0.30, 0.28, 0.22, -0.12, -0.1, 0.0, 0.26, -0.05] * 2)
            ty = my + 0.32 + spikes[:28]
        else:
            ty = my + 0.39 + 0.14 * np.sin(np.linspace(0, 3.5 * np.pi, 28))
        ax.plot(tx, ty, color=accent, lw=1.25, zorder=6)
        # Monitor stand
        ax.add_patch(Rectangle((mx + mw / 2 - 0.08, my - 0.16), 0.16, 0.16, facecolor="#64748B", zorder=4))
        ax.add_patch(Rectangle((mx + mw / 2 - 0.24, my - 0.20), 0.48, 0.05, facecolor="#475569", zorder=4))

        # Workstation Tower / Gateway Chassis (right side of monitor)
        tx0, ty0, tw0, th0 = x + 0.32, y - 0.44, 0.72, 0.94
        ax.add_patch(FancyBboxPatch(
            (tx0, ty0), tw0, th0, boxstyle="round,pad=0.01,rounding_size=0.04",
            facecolor="#334155", edgecolor="#0F172A", lw=1.0, zorder=4
        ))
        for b_i in range(3):
            ax.add_patch(Rectangle((tx0 + 0.08, ty0 + 0.62 - b_i * 0.16), tw0 - 0.16, 0.09,
                                   facecolor="#0F172A", edgecolor="#64748B", lw=0.4, zorder=5))
        ax.add_patch(Circle((tx0 + 0.20, ty0 + 0.16), 0.045, facecolor="#22C55E", zorder=6))
        ax.add_patch(Circle((tx0 + 0.36, ty0 + 0.16), 0.045, facecolor="#38BDF8", zorder=6))

        # Labels below icon
        ax.text(x, y - 0.64, subtitle, ha="center", va="center",
                fontsize=7.9, fontweight="bold", color="#0F172A", zorder=6)
        ax.text(x, y - 0.86, formula, ha="center", va="center",
                fontsize=7.4, color="#475569", zorder=6)

    # --- Helper 2: Ideational Programmable Rack Switch with AQM & Hazard Bar ---
    def draw_switch_node(x, y, title, role, params_txt, fill_pct, is_core=False):
        w, h = 3.25, 2.25
        ec = "#1D4ED8" if is_core else "#0369A1"
        _add_card(ax, x, y, w, h, fc="#F8FAFC", ec=ec, lw=1.7, z=3)

        # Header
        ax.add_patch(FancyBboxPatch(
            (x - w / 2 + 0.03, y + h / 2 - 0.36), w - 0.06, 0.33,
            boxstyle="round,pad=0.01,rounding_size=0.06",
            facecolor=ec, edgecolor="none", zorder=4
        ))
        ax.text(x, y + h / 2 - 0.195, f"{title}: {role}", ha="center", va="center",
                fontsize=7.9, fontweight="bold", color="#FFFFFF", zorder=5)

        # 1U Rack Switch Chassis
        cx, cy, cw, ch = x - w / 2 + 0.18, y + 0.12, w - 0.36, 0.50
        ax.add_patch(FancyBboxPatch(
            (cx, cy), cw, ch, boxstyle="round,pad=0.01,rounding_size=0.04",
            facecolor="#1E293B", edgecolor="#0F172A", lw=1.1, zorder=4
        ))
        # RJ45 / SFP+ Ports & Status LEDs
        for p_i in range(8):
            px = cx + 0.14 + p_i * 0.27
            ax.add_patch(Rectangle((px, cy + 0.08), 0.17, 0.20,
                                   facecolor="#0F172A", edgecolor="#94A3B8", lw=0.5, zorder=5))
            led_col = "#22C55E" if p_i % 3 != 0 else "#F59E0B"
            ax.add_patch(Circle((px + 0.085, cy + 0.37), 0.032, facecolor=led_col, zorder=6))
        ax.text(cx + cw - 0.24, cy + 0.24, "⇄", ha="center", va="center",
                fontsize=11, fontweight="bold", color="#38BDF8", zorder=6)

        # Parameter badge
        ax.text(x, y - 0.12, params_txt, ha="center", va="center",
                fontsize=7.6, fontweight="bold", color="#1E293B", zorder=6)

        # Queue & Proportional-Hazard Stress Gauge Bar
        bx, by, bw, bh = x - w / 2 + 0.20, y - 0.56, w - 0.40, 0.26
        ax.add_patch(FancyBboxPatch(
            (bx, by), bw, bh, boxstyle="round,pad=0.01,rounding_size=0.03",
            facecolor="#F1F5F9", edgecolor="#64748B", lw=0.8, zorder=4
        ))
        fw = bw * (fill_pct / 100.0)
        bar_col = "#FCA5A5" if fill_pct >= 65 else ("#FDE047" if fill_pct >= 45 else "#6EE7B7")
        ax.add_patch(FancyBboxPatch(
            (bx, by), fw, bh, boxstyle="round,pad=0.01,rounding_size=0.03",
            facecolor=bar_col, edgecolor="none", zorder=5
        ))
        ax.text(x, by + bh / 2, f"Queue $q_i(t)$ & Hazard Stress: {fill_pct}%",
                ha="center", va="center", fontsize=7.4, fontweight="bold", color="#0F172A", zorder=6)

        # Bottom reliability formula note
        sub_note = "Zero-Floor / Sigmoid AQM $\\tilde{P}_1(q_1)$" if not is_core else "Core Rejuvenation & Standby Gate $\\theta_2(p)$"
        ax.text(x, y - 0.84, sub_note, ha="center", va="center",
                fontsize=7.3, fontstyle="italic", color="#475569", zorder=6)

    # --- Helper 3: Ideational Redundant Server Rack Cluster (k-out-of-n) ---
    def draw_server_rack(x, y, title, role, mttf_txt, is_standby=False):
        w, h = 2.70, 2.25
        ec = "#059669" if not is_standby else "#7C3AED"
        _add_card(ax, x, y, w, h, fc="#FFFFFF", ec=ec, lw=1.6, z=3)

        ax.add_patch(FancyBboxPatch(
            (x - w / 2 + 0.03, y + h / 2 - 0.36), w - 0.06, 0.33,
            boxstyle="round,pad=0.01,rounding_size=0.06",
            facecolor=ec, edgecolor="none", zorder=4
        ))
        ax.text(x, y + h / 2 - 0.195, title, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="#FFFFFF", zorder=5)

        # Server Rack Cabinet with 3 Blade Units
        rx, ry, rw, rh = x - 0.95, y - 0.38, 1.90, 1.02
        ax.add_patch(FancyBboxPatch(
            (rx, ry), rw, rh, boxstyle="round,pad=0.01,rounding_size=0.04",
            facecolor="#0F172A", edgecolor="#334155", lw=1.1, zorder=4
        ))
        for b_i in range(3):
            by_i = ry + 0.08 + b_i * 0.31
            ax.add_patch(Rectangle((rx + 0.08, by_i), rw - 0.16, 0.24,
                                   facecolor="#1E293B", edgecolor="#475569", lw=0.7, zorder=5))
            # Drive bay grilles
            for g_i in range(4):
                ax.add_patch(Rectangle((rx + 0.14 + g_i * 0.22, by_i + 0.04), 0.16, 0.16,
                                       facecolor="#334155", edgecolor="none", zorder=6))
            # Blade LEDs
            ax.add_patch(Circle((rx + rw - 0.38, by_i + 0.12), 0.035, facecolor="#22C55E", zorder=6))
            ax.add_patch(Circle((rx + rw - 0.22, by_i + 0.12), 0.035,
                                facecolor="#38BDF8" if not is_standby else "#C084FC", zorder=6))

        ax.text(x, y - 0.60, role, ha="center", va="center",
                fontsize=7.8, fontweight="bold", color="#0F172A", zorder=6)
        ax.text(x, y - 0.84, mttf_txt, ha="center", va="center",
                fontsize=7.3, color="#475569", zorder=6)

    # 1. Place Top SDN & Reliability Iterate-Test Orchestration Plane
    sdn_x, sdn_y, sdn_w, sdn_h = 8.1, 9.15, 11.4, 1.35
    _add_card(ax, sdn_x, sdn_y, sdn_w, sdn_h, fc="#FAF5FF", ec="#7C3AED", lw=1.8, z=4)
    ax.add_patch(FancyBboxPatch(
        (sdn_x - sdn_w / 2 + 0.04, sdn_y + sdn_h / 2 - 0.38), sdn_w - 0.08, 0.34,
        boxstyle="round,pad=0.01,rounding_size=0.06",
        facecolor="#6D28D9", edgecolor="none", zorder=5
    ))
    ax.text(sdn_x, sdn_y + sdn_h / 2 - 0.21,
            "SDN Programmable Reliability & Iterate–Test Control Plane (Kuraś et al., 2026, arXiv:2611.03034)",
            ha="center", va="center", fontsize=9.2, fontweight="bold", color="#FFFFFF", zorder=6)
    ax.text(sdn_x, sdn_y - 0.02,
            "14-Bit Configuration Register $p \\in \\{0,1\\}^{14}$ ($|2^F| = 16{,}384$ States)  $\\cdot$  "
            "Pre-Screening ($\\tau_{\\mathrm{pre}}=60$)  $\\cdot$  Right-Censored Life-Tests ($\\tau_{\\mathrm{cens}}=250$)",
            ha="center", va="center", fontsize=8.0, fontweight="bold", color="#4C1D95", zorder=6)
    ax.text(sdn_x, sdn_y - 0.36,
            "Maintains Empirical Survival & Utility Beliefs $M_k^p$, Selects Shipped Configuration $p_k^* = \\arg\\max_{p \\in \\mathcal{P}_k} M_k^p$, "
            "and Allocates Budget $(n_{\\mathrm{new}}, n_{\\mathrm{retest}})$",
            ha="center", va="center", fontsize=7.6, color="#5B21B6", zorder=6)

    # 2. Place Left Column: 3 Ingress Traffic & Shock Generators (x = 1.65)
    draw_workstation_node(1.65, 6.85, "Source $H_1$: Chaotic Burst",
                          "Generalized Logistic Map", "$x_{t+1}=r x_t(1-x_t),\\; r=3.89$",
                          wave_type="chaos", accent="#DC2626")
    draw_workstation_node(1.65, 4.35, "Source $H_2$: Heavy-Tailed",
                          "32 Pareto ON/OFF Sources", "$\\alpha=1.36,\\; H=0.82,\\; \\widehat{H}=0.81$",
                          wave_type="pareto", accent="#7C3AED")
    draw_workstation_node(1.65, 1.85, "Source $H_3$: Cross-Traffic",
                          "Poisson + Weibull Ageing", "$\\lambda_{\\mathrm{cross}}=15,\\; \\beta=2.2$ (IFR)",
                          wave_type="poisson", accent="#0284C7")

    # 3. Place Middle-Left Column: Edge Switches S1 & S3 (x = 6.0)
    draw_switch_node(6.0, 5.75, "Edge Switch $S_1$", "Ingress Shock Filter",
                     "Bits $p_{0..7}:\\; \\mu_1 \\in [50,74.5],\\; \\theta_1,\\; \\kappa_1$",
                     fill_pct=68, is_core=False)
    draw_switch_node(6.0, 2.05, "Edge Switch $S_3$", "Cross-Traffic Multiplexer",
                     "Buffer $B_3=100,\\; \\text{Feeds Core Spine } S_2$",
                     fill_pct=34, is_core=False)

    # 4. Place Middle-Right Column: Core Spine Switch S2 (x = 10.35)
    draw_switch_node(10.35, 4.25, "Core Switch $S_2$", "Spine & Rejuvenation",
                     "Bits $p_{8..13}:\\; \\mu_2 \\in [55,79.5],\\; \\theta_2 \\in [45,83.5]$",
                     fill_pct=54, is_core=True)

    # 5. Place Right Column: Coherent k-out-of-n Redundant Server Cluster (x = 14.35)
    draw_server_rack(14.35, 6.15, "Server Cluster $D_1$", "Active Processing Blades",
                     "Weibull–Cox Survival $\\widehat{S}_p(\\tau_{\\mathrm{cens}})$", is_standby=False)
    draw_server_rack(14.35, 2.45, "Standby Pool $D_2$", "Warm-Standby & Telemetry",
                     "Tracks $\\mathrm{MTTF}(p),\\; \\widehat{T},\\; \\widehat{D},\\; \\widehat{L},\\; \\widehat{\\Lambda}$", is_standby=True)

    # 6. Data-Plane Arrows (Cleanly routed without any text collisions)
    # H1 -> S1
    _ortho_arrow(ax, [(3.03, 6.85), (3.75, 6.85), (3.75, 6.15), (4.42, 6.15)], col="#DC2626", lw=2.0)
    # H2 -> S1
    _ortho_arrow(ax, [(3.03, 4.35), (3.75, 4.35), (3.75, 5.35), (4.42, 5.35)], col="#7C3AED", lw=2.0)
    ax.text(3.75, 7.12, "Paced Load $\\lambda(t)$\n($\\lambda_0 = 52$ pkt/step)",
            ha="center", va="bottom", fontsize=7.4, fontweight="bold", color="#1E293B",
            bbox=dict(boxstyle="round,pad=0.14", fc="#FFFFFF", ec="#CBD5E1", lw=0.8), zorder=18)

    # H3 -> S3
    _straight_arrow(ax, (3.03, 1.85), (4.42, 1.85), col="#0284C7", lw=2.0)
    ax.text(3.72, 2.15, "Cross-Traffic\n$X(t)$", ha="center", va="bottom", fontsize=7.3,
            fontweight="bold", color="#0369A1", zorder=18)

    # S1 -> S2
    _ortho_arrow(ax, [(7.58, 5.75), (8.18, 5.75), (8.18, 4.65), (8.77, 4.65)], col="#1D4ED8", lw=2.4)
    ax.text(8.18, 6.02, "Departures $D_1(t)$\n$\\min(q_1, \\mu_1(p))$",
            ha="center", va="bottom", fontsize=7.4, fontweight="bold", color="#1E3A8A",
            bbox=dict(boxstyle="round,pad=0.14", fc="#EFF6FF", ec="#93C5FD", lw=0.8), zorder=18)

    # S3 -> S2
    _ortho_arrow(ax, [(7.58, 2.05), (8.18, 2.05), (8.18, 3.85), (8.77, 3.85)], col="#0284C7", lw=2.1)

    # S2 -> D1 and S2 -> D2
    _ortho_arrow(ax, [(11.93, 4.65), (12.48, 4.65), (12.48, 6.15), (13.00, 6.15)], col="#059669", lw=2.4)
    _ortho_arrow(ax, [(11.93, 3.85), (12.48, 3.85), (12.48, 2.45), (13.00, 2.45)], col="#7C3AED", lw=2.1)
    ax.text(12.48, 6.45, "Survived Flow\n$D_2(t)$", ha="center", va="bottom", fontsize=7.4,
            fontweight="bold", color="#065F46",
            bbox=dict(boxstyle="round,pad=0.14", fc="#ECFDF5", ec="#6EE7B7", lw=0.8), zorder=18)

    # 7. SDN Control Plane Downlinks (Purple dashed orthogonal lines)
    # Downlink to S1
    _straight_arrow(ax, (6.0, 8.47), (6.0, 6.88), col="#7C3AED", lw=1.7, ls="--")
    ax.text(6.0, 7.68, "Set $p_{0..7}$\n($\\mu_1,\\theta_1,\\kappa_1$)", ha="center", va="center",
            fontsize=7.3, fontweight="bold", color="#6D28D9",
            bbox=dict(boxstyle="round,pad=0.14", fc="#F5F3FF", ec="#C4B5FD", lw=0.8), zorder=18)

    # Downlink to S2
    _straight_arrow(ax, (10.35, 8.47), (10.35, 5.38), col="#7C3AED", lw=1.7, ls="--")
    ax.text(10.35, 6.95, "Set $p_{8..13}$\n($\\mu_2,\\theta_2$)", ha="center", va="center",
            fontsize=7.3, fontweight="bold", color="#6D28D9",
            bbox=dict(boxstyle="round,pad=0.14", fc="#F5F3FF", ec="#C4B5FD", lw=0.8), zorder=18)

    # Telemetry Uplink from D1/D2 to SDN Plane (Green dash-dot)
    _ortho_arrow(ax, [(14.35, 7.28), (14.35, 9.15), (13.80, 9.15)], col="#059669", lw=1.8, ls="-.")
    ax.text(14.35, 8.05, "Censored Trial Score\n$Y_k(p),\\; \\widehat{S}_p(\\tau_{\\mathrm{cens}}),\\; \\widehat{\\Lambda}$",
            ha="center", va="center", fontsize=7.3, fontweight="bold", color="#065F46",
            bbox=dict(boxstyle="round,pad=0.14", fc="#ECFDF5", ec="#6EE7B7", lw=0.8), zorder=18)

    # 8. Bottom Delayed TCP Congestion & Shock Feedback Loop (Red dashed lane along y = 0.42)
    _ortho_arrow(ax, [(10.35, 3.12), (10.35, 0.42), (1.65, 0.42), (1.65, 0.82)], col="#DC2626", lw=1.8, ls="--")
    # Tap loss from S1 cleanly along the open vertical corridor x = 7.95 (between S1/S3 and S2)
    ax.plot([7.63, 7.95, 7.95], [4.95, 4.95, 0.42], color="#DC2626", lw=1.5, ls=":", zorder=12)
    ax.add_patch(Circle((7.95, 0.42), 0.07, facecolor="#DC2626", edgecolor="none", zorder=16))
    ax.text(4.75, 0.42,
            "Delayed Loss & Hazard Feedback Bus:  $\\lambda(t) = \\lambda_0 \\max\\{0.40,\\; 1 - 1.8\\,\\bar{L}(t-\\tau)\\}$  ($\\tau = 5$ steps)",
            ha="center", va="center", fontsize=7.6, fontweight="bold", color="#991B1B",
            bbox=dict(boxstyle="round,pad=0.16", fc="#FEF2F2", ec="#FCA5A5", lw=0.9), zorder=18)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIGS, f"fig1_network_architecture.{ext}"), dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close()
    print("Rendered Fig1: fig1_network_architecture.pdf")


# ===========================================================================
# FIGURE 2: TWO-LEVEL ITERATE-TEST CORE & LIFE-TESTING PIPELINE SCHEMATIC
# ===========================================================================
def render_fig2_iterate_test_pipeline():
    fig, ax = plt.subplots(figsize=(13.6, 8.2), dpi=300)
    ax.set_xlim(0.0, 15.0)
    ax.set_ylim(0.0, 9.2)
    ax.axis("off")

    # Outer Level Container Box (Top)
    _add_card(ax, 7.5, 8.05, 14.4, 1.75, fc="#F8FAFC", ec="#1D4ED8", lw=1.8, z=1)
    ax.text(0.55, 8.65, "OUTER LEVEL: RELIABILITY FRAMING EPOCH  $(F,\\; U,\\; \\tau_{\\mathrm{cens}})$",
            ha="left", va="center", fontsize=9.6, fontweight="bold", color="#1E3A8A", zorder=5)

    # Three compact cards inside Outer Level
    def draw_mini_stage(x, y, w, h, title, sub, bg="#EFF6FF", ec="#2563EB"):
        _add_card(ax, x, y, w, h, fc=bg, ec=ec, lw=1.3, z=4)
        ax.text(x, y + 0.16, title, ha="center", va="center", fontsize=8.3, fontweight="bold", color="#0F172A", zorder=6)
        ax.text(x, y - 0.16, sub, ha="center", va="center", fontsize=7.3, color="#334155", zorder=6)

    draw_mini_stage(2.6, 7.75, 3.7, 0.90, "1. System & Telemetry Audit",
                    "Identify failure modes, Weibull $\\beta$,\nand shock regime ($\\alpha, H,\\lambda_0$)")
    draw_mini_stage(7.5, 7.75, 4.1, 0.90, "2. Commit to Frame $(F, U, \\tau_{\\mathrm{cens}})$",
                    "14 binary knobs $|2^F|=16{,}384$,\nsurvival utility $U(p)$, window $\\tau_{\\mathrm{cens}}$")
    draw_mini_stage(12.4, 7.75, 3.7, 0.90, "3. Stopping / Reframe Rule",
                    "Reframe when $\\mathrm{EI}_k - c_{\\mathrm{iter}} < G_{\\mathrm{ref}} - c_{\\mathrm{ref}}$\n(Kuraś et al., 2026, Sec. 3.6)",
                    bg="#FFFBEB", ec="#D97706")

    _straight_arrow(ax, (4.46, 7.75), (5.44, 7.75), col="#1D4ED8", lw=1.7)
    _straight_arrow(ax, (9.56, 7.75), (10.54, 7.75), col="#D97706", lw=1.7, ls="--")

    # Inner Level Container Box (Bottom)
    _add_card(ax, 7.5, 3.35, 14.4, 6.25, fc="#F0FDF4", ec="#15803D", lw=1.8, z=1)
    ax.text(0.55, 6.18,
            "INNER LEVEL: ITERATE–TEST RELIABILITY LOOP WITHIN A FRAME  (Theorem 1 & 2,  k = 1, 2, ..., K)",
            ha="left", va="center", fontsize=9.6, fontweight="bold", color="#14532D", zorder=5)

    # Helper for 8 Detailed Stage Cards in 2x4 Grid
    def draw_stage_card(x, y, num, title, bullets, hdr_col="#0284C7", ec_col="#0369A1"):
        w, h = 3.15, 2.05
        _add_card(ax, x, y, w, h, fc="#FFFFFF", ec=ec_col, lw=1.5, z=4)
        ax.add_patch(FancyBboxPatch(
            (x - w / 2 + 0.03, y + h / 2 - 0.38), w - 0.06, 0.35,
            boxstyle="round,pad=0.01,rounding_size=0.06",
            facecolor=hdr_col, edgecolor="none", zorder=5
        ))
        ax.text(x, y + h / 2 - 0.20, f"Stage {num}: {title}",
                ha="center", va="center", fontsize=8.1, fontweight="bold", color="#FFFFFF", zorder=6)
        y0 = y + h / 2 - 0.62
        for idx_b, b_txt in enumerate(bullets):
            ax.text(x - w / 2 + 0.14, y0 - idx_b * 0.39, f"• {b_txt}",
                    ha="left", va="center", fontsize=7.35, color="#1E293B", zorder=6)

    # Row 1 of Inner Loop (Left -> Right, y = 4.75)
    xs = [2.1, 5.7, 9.3, 12.9]
    y_top = 4.75
    y_bot = 1.65

    draw_stage_card(xs[0], y_top, "I", "Retention Archive $\\mathcal{P}_k$",
                    ["Elitist memory $\\mathcal{P}_k \\subseteq \\mathcal{P}_{k+1}$",
                     "Running means $M_k^p = \\frac{1}{n_k^p}\\sum Y_i$",
                     "Non-dominated Pareto set $\\mathcal{A}_k$",
                     "No tested config ever evicted"],
                    hdr_col="#059669", ec_col="#047857")

    draw_stage_card(xs[1], y_top, "II", "Local Ideation $\\mathcal{N}_r(p)$",
                    ["Sample base $p \\in \\mathcal{P}_k$ (or leader)",
                     "Generate $m=5$ Hamming variants",
                     "Distance $1 \\leq d_H(p, p') \\leq r=2$",
                     "Persistent generativity $\\sum \\delta_k = \\infty$"],
                    hdr_col="#0284C7", ec_col="#0369A1")

    draw_stage_card(xs[2], y_top, "III", "Coarse Pre-Screening",
                    ["Short trial $\\tau_{\\mathrm{pre}} = 60$ steps",
                     "Filters catastrophic overload states",
                     "Ranks $m=5$ candidate ideas",
                     "Selects top $n_{\\mathrm{new}}$ challengers"],
                    hdr_col="#7C3AED", ec_col="#6D28D9")

    draw_stage_card(xs[3], y_top, "IV", "Budget Split $(n_{\\mathrm{new}}, n_{\\mathrm{retest}})$",
                    ["Fixed budget $B = 5$ tests / round",
                     "Balanced split: $3$ new + $2$ re-tests",
                     "Adaptive R&S: OCBA / UCB1 / TS",
                     "Test-once ($5/0$): no re-testing"],
                    hdr_col="#4F46E5", ec_col="#4338CA")

    # Row 2 of Inner Loop (Right -> Left, y = 1.65)
    draw_stage_card(xs[3], y_bot, "V", "SDN / Switch Deployment",
                    ["Program 14-bit register $p \\in \\{0,1\\}^{14}$",
                     "Set $\\mu_1,\\theta_1,\\kappa_1$ and $\\mu_2,\\theta_2$",
                     "Cold-reset (i.i.d.) vs. Persistent",
                     "Optional burn-in $\\tau_{\\mathrm{burn}} = 30$ steps"],
                    hdr_col="#D97706", ec_col="#B45309")

    draw_stage_card(xs[2], y_bot, "VI", "Right-Censored Life-Test",
                    ["Observation window $\\tau_{\\mathrm{cens}} = 250$",
                     "Weibull–Cox hazard $h_p(t)$ & shocks",
                     "Measure $\\widehat{T},\\; \\widehat{D},\\; \\widehat{L},\\; \\widehat{S}_p(\\tau_{\\mathrm{cens}})$",
                     "Short-window divergence $\\widehat{\\Lambda}(q_1)$"],
                    hdr_col="#DC2626", ec_col="#B91C1C")

    draw_stage_card(xs[1], y_bot, "VII", "Reliability Scoring Filter",
                    ["Smooth score: $Y_{\\mathrm{smooth}}(p) \\in [0,1]$",
                     "SLA penalty: $-0.15\\,\\mathbb{I}\\{\\widehat{\\Lambda}>0.05\\}$",
                     "Penalty adds variance $0.15^2 f_p(1-f_p)$",
                     "Update posterior/mean $M_{k+1}^p$"],
                    hdr_col="#0891B2", ec_col="#0E7490")

    draw_stage_card(xs[0], y_bot, "VIII", "Shipped Favourite $p_k^*$",
                    ["Deploy $p_k^* = \\arg\\max_{p\\in\\mathcal{P}_k} M_k^p$",
                     "True shipped utility $U(p_k^*)$",
                     "Curse bias: $M_k^{p_k^*} - U(p_k^*)$",
                     "Choice gap: $\\max_{p\\in\\mathcal{P}_k} U(p) - U(p_k^*)$"],
                    hdr_col="#15803D", ec_col="#166534")

    # Arrows connecting Outer <-> Inner and Stages I -> VIII -> I
    # Frame down arrow into Stage I/II
    _straight_arrow(ax, (6.2, 7.18), (6.2, 6.48), col="#1D4ED8", lw=2.0)
    ax.text(5.35, 6.82, "Frame $(F, U)$", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#1E3A8A")

    # Reframe up arrow from Inner Loop to Stage 3
    _straight_arrow(ax, (12.4, 6.48), (12.4, 7.18), col="#D97706", lw=2.0, ls="--")
    ax.text(13.25, 6.82, "Epoch Reframe", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#B45309")

    # Row 1 horizontal arrows: I -> II -> III -> IV
    for i in range(3):
        _straight_arrow(ax, (xs[i] + 1.58, y_top), (xs[i + 1] - 1.58, y_top), col="#1E293B", lw=2.0)

    # Vertical arrow IV -> V
    _straight_arrow(ax, (xs[3], y_top - 1.03), (xs[3], y_bot + 1.03), col="#4F46E5", lw=2.2)
    ax.text(xs[3] + 0.15, 3.20, "Schedule $5$ trials\n($n_{\\mathrm{new}} + n_{\\mathrm{retest}}$)",
            ha="left", va="center", fontsize=7.4, fontweight="bold", color="#4338CA")

    # Row 2 horizontal arrows: V -> VI -> VII -> VIII
    for i in range(3, 0, -1):
        _straight_arrow(ax, (xs[i] - 1.58, y_bot), (xs[i - 1] + 1.58, y_bot), col="#1E293B", lw=2.0)

    # Vertical return arrow VIII -> I (Next Round k + 1)
    _straight_arrow(ax, (xs[0], y_bot + 1.03), (xs[0], y_top - 1.03), col="#15803D", lw=2.2)
    ax.text(xs[0] - 0.15, 3.20, "Next Round\n$k \\leftarrow k+1$",
            ha="right", va="center", fontsize=7.6, fontweight="bold", color="#15803D")

    # Direct Re-Testing Shortcut Arrow across middle corridor (y = 3.20) from Stage I/VIII leaders to Stage V
    _ortho_arrow(ax, [(xs[0] + 1.20, y_top - 1.03), (xs[0] + 1.20, 3.20), (xs[3] - 0.90, 3.20), (xs[3] - 0.90, y_bot + 1.03)],
                 col="#7C3AED", lw=1.8, ls="--")
    ax.text(7.5, 3.20,
            "Leader Re-Testing Stream:  Re-evaluate Top Incumbents in $\\mathcal{P}_k$ ($n_{\\mathrm{retest}}$ replications) to Suppress Optimizer's Curse",
            ha="center", va="center", fontsize=7.6, fontweight="bold", color="#5B21B6",
            bbox=dict(boxstyle="round,pad=0.18", fc="#F5F3FF", ec="#C4B5FD", lw=0.9), zorder=18)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIGS, f"fig2_iterate_test_pipeline.{ext}"), dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close()
    print("Rendered Fig2: fig2_iterate_test_pipeline.pdf")


# ===========================================================================
# FIGURE 3: 7-EXPERIMENT VALIDATION & GROUND-TRUTH BENCHMARKING PROTOCOL
# ===========================================================================
def render_fig3_experimental_protocol():
    fig, ax = plt.subplots(figsize=(13.6, 8.0), dpi=300)
    ax.set_xlim(0.0, 15.0)
    ax.set_ylim(0.0, 8.8)
    ax.axis("off")

    # Top Tier: Exhaustive 16,384 Ground-Truth Oracle
    _add_card(ax, 7.5, 7.85, 14.4, 1.35, fc="#EFF6FF", ec="#1D4ED8", lw=1.8, z=2)
    ax.add_patch(FancyBboxPatch(
        (0.34, 8.16), 14.32, 0.34, boxstyle="round,pad=0.01,rounding_size=0.06",
        facecolor="#1E40AF", edgecolor="none", zorder=4
    ))
    ax.text(7.5, 8.33,
            "TIER I — EXHAUSTIVE GROUND-TRUTH ORACLE  ($|2^F| = 16{,}384$ Configurations $\\times$ 16 Independent Long Seeds $\\times$ 400 Steps)",
            ha="center", va="center", fontsize=8.8, fontweight="bold", color="#FFFFFF", zorder=5)
    ax.text(7.5, 7.62,
            "Computes Exact Population Parameters for Every $p \\in \\{0,1\\}^{14}$:   "
            "True Smooth Utility $U_{\\mathrm{smooth}}(p)$,   SLA-Penalized Utility $U_{\\mathrm{pen}}(p)$,   "
            "Weibull–Cox Survival $R_p(\\tau_{\\mathrm{cens}})$,\n"
            "Mean Time To Failure $\\mathrm{MTTF}(p)$,   Flagging Probability $f_p = \\mathbb{P}(\\widehat{\\Lambda}>0.05)$,   "
            "Provisioning Cost $C_{\\mathrm{cost}}(p)$,   and Exact Hamming-$r$ Local Optima $\\mathcal{O}_r$",
            ha="center", va="center", fontsize=7.6, color="#1E293B", zorder=5)

    # Middle Tier: 7 Experimental Modules (Row A: E1, E2, E3, E4; Row B: E5, E6, E7)
    def draw_exp_box(x, y, w, h, code, title, lines, col="#0284C7"):
        _add_card(ax, x, y, w, h, fc="#FFFFFF", ec=col, lw=1.5, z=4)
        ax.add_patch(FancyBboxPatch(
            (x - w / 2 + 0.03, y + h / 2 - 0.36), w - 0.06, 0.33,
            boxstyle="round,pad=0.01,rounding_size=0.05",
            facecolor=col, edgecolor="none", zorder=5
        ))
        ax.text(x, y + h / 2 - 0.195, f"{code}: {title}",
                ha="center", va="center", fontsize=8.0, fontweight="bold", color="#FFFFFF", zorder=6)
        y0 = y + h / 2 - 0.58
        for idx_l, line in enumerate(lines):
            ax.text(x - w / 2 + 0.12, y0 - idx_l * 0.35, f"• {line}",
                    ha="left", va="center", fontsize=7.25, color="#1E293B", zorder=6)

    # Row A (y = 5.35): 4 boxes
    xs_a = [2.0, 5.67, 9.33, 13.0]
    y_a = 5.35
    draw_exp_box(xs_a[0], y_a, 3.35, 1.95, "Exp. 1", "Queue & Hazard Dynamics",
                 ["Load $\\lambda_0 \\in [30,90]$, delay $\\tau \\in [1,40]$",
                  "200 random configs (overload sweep)",
                  "Abry–Veitch wavelet Hurst $\\widehat{H}$",
                  "Fourier surrogates & $0\\text{--}1$ chaos test"],
                 col="#0284C7")

    draw_exp_box(xs_a[1], y_a, 3.35, 1.95, "Exp. 2", "16,384 Landscape & Orders",
                 ["Full enumeration across 4 regimes",
                  "Stochastic orders $\\leq_{\\mathrm{st}},\\; \\leq_{\\mathrm{hr}}$",
                  "Local optima count ($r=2$: 1 to 9)",
                  "Hazard-dominance separation check"],
                 col="#059669")

    draw_exp_box(xs_a[2], y_a, 3.35, 1.95, "Exp. 3", "Censoring & Curse Anatomy",
                 ["Type-I censoring $\\tau_{\\mathrm{cens}} \\in [50, 600]$",
                  "Bernoulli jump $0.15^2 f_p(1-f_p)$",
                  "Up to $12\\times$ SD inflation from penalty",
                  "Smooth survival vs. binary threshold"],
                 col="#DC2626")

    draw_exp_box(xs_a[3], y_a, 3.35, 1.95, "Exp. 4", "Life-Testing Budget Split",
                 ["Splits $n_{\\mathrm{new}}/n_{\\mathrm{retest}} \\in \\{1/4..5/0\\}$",
                  "100 paired CRN replications",
                  "Horizons $k \\in \\{10, 25, 50, 100\\}$ rounds",
                  "Poisson, Pareto LRD & Chaotic regimes"],
                 col="#7C3AED")

    # Row B (y = 2.75): 3 wider boxes
    xs_b = [2.65, 7.50, 12.35]
    y_b = 2.75
    draw_exp_box(xs_b[0], y_b, 4.65, 1.95, "Exp. 5", "Sequential R&S & Bandit Benchmarks",
                 ["Head-to-head under 500 life-tests ($n=100$):",
                  "Iterate–Test ($3/2,\\; 1/4,\\; 5/0$) vs. OCBA (Chen 2000)",
                  "UCB1, Thompson Sampling & Successive Halving",
                  "Quantifies value of 60-step coarse pre-screening"],
                 col="#4F46E5")

    draw_exp_box(xs_b[1], y_b, 4.65, 1.95, "Exp. 6", "Persistent Online Switching & Bias",
                 ["Online switching without resetting queues $q_1, q_2$",
                  "Burn-in $\\tau_{\\mathrm{burn}}=0$ vs. $\\tau_{\\mathrm{burn}}=30$ vs. Cold Reset",
                  "Confirmation bias $b \\in [0, 0.30]$ (Proposition 1)",
                  "Non-stationary load drift $D$ (Corollary 1)"],
                 col="#D97706")

    draw_exp_box(xs_b[2], y_b, 4.65, 1.95, "Exp. 7", "Pareto Reliability–Cost Archive",
                 ["Competing objectives: Reliability $U_{\\mathrm{pen}}$ vs. Cost $C_{\\mathrm{cost}}$",
                  "Exact 16,384 Pareto frontier ($19$ global optima)",
                  "Re-tested Archive ($3/2$) vs. Test-Once Archive ($5/0$)",
                  "Hypervolume recovery & false-dominance suppression"],
                 col="#0E7490")

    # Bottom Tier: Paired Statistical Verification & Reproducibility Bar
    _add_card(ax, 7.5, 0.78, 14.4, 1.10, fc="#F0FDF4", ec="#15803D", lw=1.7, z=2)
    ax.text(7.5, 0.96,
            "TIER III — PAIRED COMMON-RANDOM-NUMBERS (CRN) INFERENCE & AUTOMATED NUMERICAL VERIFICATION",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color="#14532D", zorder=5)
    ax.text(7.5, 0.56,
            "Paired 95% Confidence Intervals ($n=100$ Replications)  $\\cdot$  Bonferroni Multiple-Comparison Control  $\\cdot$  "
            "Automated Script `verify_smps_numbers.py` Checks Every Number in Manuscript Against `smps_results.json`",
            ha="center", va="center", fontsize=7.5, color="#166534", zorder=5)

    # Clean vertical arrows from Tier I -> Row A, Row A -> Row B, Row B -> Tier III
    for x_val in xs_a:
        _straight_arrow(ax, (x_val, 7.16), (x_val, 6.34), col="#1D4ED8", lw=1.7)
    for x_val in xs_b:
        _straight_arrow(ax, (x_val, 4.36), (x_val, 3.74), col="#334155", lw=1.7)
        _straight_arrow(ax, (x_val, 1.76), (x_val, 1.34), col="#15803D", lw=1.7)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIGS, f"fig3_experimental_protocol.{ext}"), dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close()
    print("Rendered Fig3: fig3_experimental_protocol.pdf")


if __name__ == "__main__":
    render_fig1_network_architecture()
    render_fig2_iterate_test_pipeline()
    render_fig3_experimental_protocol()
