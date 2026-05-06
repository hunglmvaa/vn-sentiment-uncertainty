import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ============================================================
# Global style
# ============================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 22,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# ============================================================
# Data
# ============================================================

# Panel A
strata = ["Low\n(64.1%)", "Medium\n(20.1%)", "High\n(15.9%)"]
baseline_acc = np.array([86.4, 61.5, 48.8])
aug_acc = np.array([92.0, 79.0, 74.0])
weighted = [3.59, 3.52, 4.02]

# Ghi đúng theo paper, KHÔNG lấy bằng phép trừ từ số làm tròn
gain_labels = ["+5.6 pp", "+17.5 pp", "+25.3 pp"]

# Panel B
strategies = ["Standard\nAug.", "UGA\nRun 1", "UGA\nRun 2"]
f1_values = [86.63, 75.73, 78.04]
std_mean = 86.63
std_sigma = 1.13
std_low = std_mean - std_sigma
std_high = std_mean + std_sigma
drop_labels = ["", "−10.9 pp", "−8.6 pp"]

# Panel C
bald_labels = [r"$D_{\mathrm{med}}$", r"$D_{\mathrm{high}}$"]
epistemic = np.array([51, 23])
aleatoric = np.array([49, 77])

# ============================================================
# Colors
# ============================================================
baseline_color = "#73889C"
aug_color = "#18A9B5"
standard_color = "#1E5AA8"
uga1_color = "#D61F1F"
uga2_color = "#F26B21"
epistemic_color = "#1170B8"
aleatoric_color = "#F58220"
grid_color = "#CCCCCC"
gray_band = "#E8E8E8"
green_color = "#2E8B3C"

# ============================================================
# Figure layout
# ============================================================
fig = plt.figure(figsize=(16.5, 10))

gs = fig.add_gridspec(
    nrows=2,
    ncols=3,
    height_ratios=[13, 2.2],   # tăng chiều cao footer
    wspace=0.22,
    hspace=0.12
)

axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[0, 2])
axFooter = fig.add_subplot(gs[1, :])
axFooter.axis("off")

# Căn khung tổng thể tốt hơn
fig.subplots_adjust(
    top=0.84,
    bottom=0.10,
    left=0.06,
    right=0.97
)

fig.suptitle(
    "The Diagnostic–Intervention Gap in Uncertainty-Guided Augmentation",
    fontweight="bold",
    y=0.94
)

# ============================================================
# Panel A — Stratified Gains
# ============================================================
x = np.arange(len(strata))
bar_w = 0.32

bars1 = axA.bar(
    x - bar_w / 2,
    baseline_acc,
    width=bar_w,
    label="Baseline",
    color=baseline_color,
    edgecolor="black",
    linewidth=0.4
)

bars2 = axA.bar(
    x + bar_w / 2,
    aug_acc,
    width=bar_w,
    label="Augmented",
    color=aug_color,
    edgecolor="black",
    linewidth=0.4
)

axA.set_title("A. Stratified Gains", fontweight="bold", pad=42)
axA.text(
    0.5, 1.06,
    "Aggregate metrics hide where augmentation helps.",
    transform=axA.transAxes,
    ha="center",
    va="bottom",
    style="italic",
    fontsize=12
)

axA.set_ylabel("Accuracy (%)", fontweight="bold")
axA.set_ylim(0, 105)
axA.set_xticks(x)
axA.set_xticklabels(strata)
axA.grid(axis="y", linestyle="--", alpha=0.7, color=grid_color)
axA.set_axisbelow(True)

# Value labels
for b in list(bars1) + list(bars2):
    axA.text(
        b.get_x() + b.get_width()/2,
        b.get_height() + 1.5,
        f"{b.get_height():.1f}",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=11
    )

# Gain brackets
for i in range(len(x)):
    y = max(baseline_acc[i], aug_acc[i]) + 7
    x1 = x[i] - bar_w / 2
    x2 = x[i] + bar_w / 2
    axA.plot([x1, x1, x2, x2], [y - 2, y, y, y - 2], color="black", linewidth=1.8)
    axA.text(
        x[i], y + 1.1, gain_labels[i],
        ha="center", va="bottom",
        fontweight="bold", fontsize=12
    )

# Weighted text
for i in range(len(x)):
    axA.text(
        x[i], -8.0,
        f"Weighted: +{weighted[i]:.2f} pp",
        ha="center", va="top",
        fontsize=9,
        style="italic"
    )

# Legend kéo lên nhẹ để không đụng footer
axA.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, -0.22),
    ncol=2,
    frameon=False
)

# ============================================================
# Panel B — UGA Failure
# ============================================================
xb = np.arange(len(strategies))

axB.axhspan(std_low, std_high, color=gray_band, alpha=1.0, zorder=0)
axB.axhline(std_low, color="gray", linestyle="--", linewidth=1.2)
axB.axhline(std_high, color="gray", linestyle="--", linewidth=1.2)

axB.text(
    2.42, std_mean,
    f"Standard Aug. ±1σ\n({std_low:.2f} – {std_high:.2f})",
    ha="right", va="center",
    fontsize=10, fontweight="bold"
)

# stems
for i, val in enumerate(f1_values):
    axB.vlines(xb[i], ymin=60, ymax=val, color="gray", linestyle="--", linewidth=1.6)

# points
axB.scatter(xb[0], f1_values[0], s=180, color=standard_color, edgecolor="white", linewidth=1.5, zorder=3)
axB.scatter(xb[1], f1_values[1], s=180, color=uga1_color, edgecolor="white", linewidth=1.5, zorder=3)
axB.scatter(xb[2], f1_values[2], s=180, color=uga2_color, edgecolor="white", linewidth=1.5, zorder=3)

# value labels
for i, val in enumerate(f1_values):
    axB.text(
        xb[i], val + 1.3,
        f"{val:.2f}",
        ha="center", va="bottom",
        fontweight="bold", fontsize=12
    )

# drop labels
axB.text(xb[1], f1_values[1] - 4.0, drop_labels[1], ha="center", color=uga1_color, fontweight="bold", fontsize=12)
axB.text(xb[2], f1_values[2] - 4.0, drop_labels[2], ha="center", color=uga2_color, fontweight="bold", fontsize=12)

axB.set_title("B. UGA Failure", fontweight="bold", pad=42)
axB.text(
    0.5, 1.06,
    "Naive maximum-entropy targeting is harmful.",
    transform=axB.transAxes,
    ha="center",
    va="bottom",
    style="italic",
    fontsize=12
)

axB.set_ylabel("Macro-F1 (%)", fontweight="bold")
axB.set_ylim(60, 100)
axB.set_xticks(xb)
axB.set_xticklabels(strategies)
axB.grid(axis="y", linestyle="--", alpha=0.7, color=grid_color)
axB.set_axisbelow(True)

axB.text(
    0.5, -0.13,
    "High-focused UGA: Low×0, Med×2, High×4",
    transform=axB.transAxes,
    ha="center", va="top",
    fontsize=10, style="italic"
)

# ============================================================
# Panel C — BALD Decomposition
# ============================================================
xc = np.arange(len(bald_labels))
bar_c_w = 0.42

axC.bar(
    xc, epistemic,
    width=bar_c_w,
    color=epistemic_color,
    edgecolor="black",
    linewidth=0.4,
    label="Epistemic"
)

axC.bar(
    xc, aleatoric,
    width=bar_c_w,
    bottom=epistemic,
    color=aleatoric_color,
    edgecolor="black",
    linewidth=0.4,
    label="Aleatoric"
)

axC.set_title("C. BALD Decomposition", fontweight="bold", pad=42)
axC.set_ylabel("Proportion (%)", fontweight="bold")
axC.set_ylim(0, 105)
axC.set_xlim(-0.55, 1.55)   # mở rộng để chứa callout bên trong panel
axC.set_xticks(xc)
axC.set_xticklabels(bald_labels, fontsize=12, fontweight="bold")
axC.grid(axis="y", linestyle="--", alpha=0.7, color=grid_color)
axC.set_axisbelow(True)

# percent labels
for i in range(len(xc)):
    axC.text(
        xc[i], epistemic[i] / 2,
        f"{epistemic[i]}%",
        ha="center", va="center",
        color="white", fontweight="bold", fontsize=14
    )
    axC.text(
        xc[i], epistemic[i] + aleatoric[i] / 2,
        f"{aleatoric[i]}%",
        ha="center", va="center",
        color="white", fontweight="bold", fontsize=14
    )

# Callout D_med - đặt gọn trong panel
axC.annotate(
    "More\nlearnable\ntarget",
    xy=(0, 42),
    xycoords="data",
    xytext=(-0.38, 50),
    textcoords="data",
    ha="center", va="center",
    color=standard_color,
    fontweight="bold",
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=standard_color, lw=1.5),
    arrowprops=dict(arrowstyle="->", color=standard_color, lw=1.6)
)

# Callout D_high - cũng giữ trong panel
axC.annotate(
    "Mostly\naleatoric",
    xy=(1, 82),
    xycoords="data",
    xytext=(1.42, 63),
    textcoords="data",
    ha="center", va="center",
    color=aleatoric_color,
    fontweight="bold",
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=aleatoric_color, lw=1.5),
    arrowprops=dict(arrowstyle="->", color=aleatoric_color, lw=1.6)
)

axC.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=2,
    frameon=False
)

axC.text(
    0.5, -0.13,
    r"Best future augmentation target: $D_{\mathrm{med}}$",
    transform=axC.transAxes,
    ha="center", va="top",
    fontsize=10, style="italic"
)

# ============================================================
# Rounded boxes around panels
# ============================================================
def add_panel_box(ax, pad_x=0.012, pad_y=0.016):
    pos = ax.get_position()
    box = FancyBboxPatch(
        (pos.x0 - pad_x, pos.y0 - pad_y),
        pos.width + 2 * pad_x,
        pos.height + 2 * pad_y,
        boxstyle="round,pad=0.02,rounding_size=0.015",
        transform=fig.transFigure,
        fill=False,
        linewidth=1.4,
        edgecolor="black",
        zorder=20
    )
    fig.patches.append(box)

for ax in [axA, axB, axC]:
    add_panel_box(ax)

# ============================================================
# Footer
# ============================================================
axFooter.set_xlim(0, 1)
axFooter.set_ylim(0, 1)

# Icons/text
axFooter.text(0.16, 0.5, "◉ Observation", color=standard_color, fontsize=22, fontweight="bold",
              ha="center", va="center")
axFooter.text(0.50, 0.5, "✕ Negative result", color=uga1_color, fontsize=22, fontweight="bold",
              ha="center", va="center")
axFooter.text(0.84, 0.5, "● Explanation", color=green_color, fontsize=22, fontweight="bold",
              ha="center", va="center")

# arrows
arrow1 = FancyArrowPatch((0.28, 0.5), (0.39, 0.5), arrowstyle="-|>", mutation_scale=25,
                         linewidth=2.2, color="gray")
arrow2 = FancyArrowPatch((0.61, 0.5), (0.73, 0.5), arrowstyle="-|>", mutation_scale=25,
                         linewidth=2.2, color="gray")
axFooter.add_patch(arrow1)
axFooter.add_patch(arrow2)

# ============================================================
# Save
# ============================================================
from pathlib import Path
out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)

plt.savefig(out_dir / "diagnostic_intervention_gap_fixed.png", dpi=600, bbox_inches="tight")
plt.savefig(out_dir / "diagnostic_intervention_gap_fixed.svg", bbox_inches="tight")
plt.show()