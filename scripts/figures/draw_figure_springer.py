# ============================================================
# Springer-style composite figure (improved bottom spacing)
# Figure: The Diagnostic–Intervention Gap
# Output:
#   outputs_springer/diagnostic_intervention_gap_springer_v2.png
#   outputs_springer/diagnostic_intervention_gap_springer_v2.pdf
#   outputs_springer/diagnostic_intervention_gap_springer_v2.svg
# ============================================================

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ============================================================
# 1. Global style: Springer-friendly
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],

    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,

    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,

    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",

    "savefig.dpi": 600
})

# ============================================================
# 2. Data
# ============================================================

# Panel A
strata = ["Low\n(64.1%)", "Medium\n(20.1%)", "High\n(15.9%)"]
baseline_acc = np.array([86.4, 61.5, 48.8])
aug_acc = np.array([92.0, 79.0, 74.0])
gain_labels = ["+5.6", "+17.5", "+25.3"]
weighted_labels = ["+3.59", "+3.52", "+4.02"]

# Panel B
strategies = ["Standard\nAug.", "UGA\nRun 1", "UGA\nRun 2"]
f1_values = np.array([86.63, 75.73, 78.04])
std_mean = 86.63
std_sigma = 1.13
std_low = std_mean - std_sigma
std_high = std_mean + std_sigma
drop_labels = ["", "−10.9", "−8.6"]

# Panel C
bald_labels = [r"$D_{\mathrm{med}}$", r"$D_{\mathrm{high}}$"]
epistemic = np.array([51, 23])
aleatoric = np.array([49, 77])

# ============================================================
# 3. Colors
# ============================================================

baseline_color = "#7A8A99"
aug_color = "#009E73"
standard_color = "#0072B2"
uga1_color = "#D55E00"
uga2_color = "#E69F00"
epistemic_color = "#0072B2"
aleatoric_color = "#D55E00"
grid_color = "#D8D8D8"
band_color = "#EAEAEA"
text_gray = "#333333"

# ============================================================
# 4. Figure layout
# ============================================================

# Tăng chiều cao một chút để phần đáy thoáng hơn
fig = plt.figure(figsize=(7.4, 3.9), constrained_layout=False)

gs = fig.add_gridspec(
    nrows=1,
    ncols=3,
    width_ratios=[1.12, 1.05, 1.0],
    wspace=0.38
)

axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[0, 1])
axC = fig.add_subplot(gs[0, 2])

# Tăng bottom để tách legend và dòng interpretation
fig.subplots_adjust(
    left=0.055,
    right=0.995,
    top=0.86,
    bottom=0.34,
    wspace=0.38
)

fig.suptitle(
    "The Diagnostic–Intervention Gap in Uncertainty-Guided Augmentation",
    fontsize=11,
    fontweight="bold",
    y=0.985
)

# ============================================================
# Helper functions
# ============================================================

def add_panel_label(ax, label, title):
    ax.text(
        -0.12, 1.13,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold"
    )
    ax.text(
        0.00, 1.13,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold"
    )

def add_subtitle(ax, subtitle):
    ax.text(
        0.00, 1.045,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
        style="italic",
        color=text_gray
    )

def format_axis(ax):
    ax.grid(axis="y", linestyle=":", linewidth=0.7, color=grid_color)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=8)

# ============================================================
# 5. Panel A — Stratified Gains
# ============================================================

x = np.arange(len(strata))
bar_w = 0.32

axA.bar(
    x - bar_w / 2,
    baseline_acc,
    width=bar_w,
    color=baseline_color,
    edgecolor="black",
    linewidth=0.35,
    label="Baseline"
)

axA.bar(
    x + bar_w / 2,
    aug_acc,
    width=bar_w,
    color=aug_color,
    edgecolor="black",
    linewidth=0.35,
    label="Augmented"
)

add_panel_label(axA, "(A)", "Stratified gains")
add_subtitle(axA, "Aggregate metrics hide where gains occur.")

axA.set_ylabel("Accuracy (%)")
axA.set_ylim(0, 105)
axA.set_xticks(x)
axA.set_xticklabels(strata)
format_axis(axA)

# Value labels
for i in range(len(x)):
    axA.text(
        x[i] - bar_w / 2,
        baseline_acc[i] + 1.4,
        f"{baseline_acc[i]:.1f}",
        ha="center",
        va="bottom",
        fontsize=7.8,
        fontweight="bold"
    )
    axA.text(
        x[i] + bar_w / 2,
        aug_acc[i] + 1.4,
        f"{aug_acc[i]:.1f}",
        ha="center",
        va="bottom",
        fontsize=7.8,
        fontweight="bold"
    )

# Gain brackets and weighted contribution
for i in range(len(x)):
    y = max(baseline_acc[i], aug_acc[i]) + 6.3
    x1 = x[i] - bar_w / 2
    x2 = x[i] + bar_w / 2

    axA.plot(
        [x1, x1, x2, x2],
        [y - 1.8, y, y, y - 1.8],
        color="black",
        linewidth=0.9
    )
    axA.text(
        x[i],
        y + 1.1,
        f"{gain_labels[i]} pp",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold"
    )

    axA.text(
        x[i],
        4.8,
        f"w: {weighted_labels[i]} pp",
        ha="center",
        va="center",
        fontsize=7,
        color=text_gray
    )

# Kéo legend lên cao hơn một chút
axA.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.20),
    ncol=2,
    frameon=False,
    handlelength=1.2,
    columnspacing=1.4
)

# ============================================================
# 6. Panel B — UGA Failure
# ============================================================

xb = np.arange(len(strategies))

axB.axhspan(
    std_low,
    std_high,
    color=band_color,
    zorder=0
)
axB.axhline(std_low, color="gray", linestyle="--", linewidth=0.8)
axB.axhline(std_high, color="gray", linestyle="--", linewidth=0.8)

for i, val in enumerate(f1_values):
    axB.vlines(
        xb[i],
        60,
        val,
        color="gray",
        linestyle="--",
        linewidth=0.9,
        zorder=1
    )

axB.scatter(
    xb[0], f1_values[0],
    s=42, color=standard_color,
    edgecolor="white", linewidth=0.6, zorder=3
)
axB.scatter(
    xb[1], f1_values[1],
    s=42, color=uga1_color,
    edgecolor="white", linewidth=0.6, zorder=3
)
axB.scatter(
    xb[2], f1_values[2],
    s=42, color=uga2_color,
    edgecolor="white", linewidth=0.6, zorder=3
)

add_panel_label(axB, "(B)", "UGA failure")
add_subtitle(axB, "Naive maximum-entropy targeting is harmful.")

axB.set_ylabel("Macro-F1 (%)")
axB.set_ylim(60, 100)
axB.set_xticks(xb)
axB.set_xticklabels(strategies)
format_axis(axB)

for i, val in enumerate(f1_values):
    axB.text(
        xb[i],
        val + 1.1,
        f"{val:.2f}",
        ha="center",
        va="bottom",
        fontsize=7.8,
        fontweight="bold"
    )

axB.text(
    xb[1],
    f1_values[1] - 4.2,
    f"{drop_labels[1]} pp",
    ha="center",
    va="center",
    fontsize=7.8,
    fontweight="bold",
    color=uga1_color
)

axB.text(
    xb[2],
    f1_values[2] - 4.2,
    f"{drop_labels[2]} pp",
    ha="center",
    va="center",
    fontsize=7.8,
    fontweight="bold",
    color=uga2_color
)

axB.text(
    2.18,
    std_mean,
    "Standard Aug.\n±1σ",
    ha="right",
    va="center",
    fontsize=7.2,
    color=text_gray
)

# Giữ note này bên trong panel, sát đáy panel nhưng không quá thấp
axB.text(
    0.5,
    0.035,
    "High-focused UGA: Low×0, Med×2, High×4",
    transform=axB.transAxes,
    ha="center",
    va="bottom",
    fontsize=7.0,
    style="italic",
    color=text_gray
)

# ============================================================
# 7. Panel C — BALD Decomposition
# ============================================================

xc = np.arange(len(bald_labels))
bar_w_c = 0.42

axC.bar(
    xc,
    epistemic,
    width=bar_w_c,
    color=epistemic_color,
    edgecolor="black",
    linewidth=0.35,
    label="Epistemic"
)

axC.bar(
    xc,
    aleatoric,
    bottom=epistemic,
    width=bar_w_c,
    color=aleatoric_color,
    edgecolor="black",
    linewidth=0.35,
    label="Aleatoric"
)

add_panel_label(axC, "(C)", "BALD decomposition")
add_subtitle(axC, "Learnable uncertainty differs from total entropy.")

axC.set_ylabel("Proportion (%)")
axC.set_ylim(0, 105)
axC.set_xlim(-0.55, 1.55)
axC.set_xticks(xc)
axC.set_xticklabels(bald_labels)
format_axis(axC)

for i in range(len(xc)):
    axC.text(
        xc[i],
        epistemic[i] / 2,
        f"{epistemic[i]}%",
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color="white"
    )
    axC.text(
        xc[i],
        epistemic[i] + aleatoric[i] / 2,
        f"{aleatoric[i]}%",
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color="white"
    )

axC.annotate(
    "more\nlearnable",
    xy=(0.0, 51),
    xytext=(-0.47, 63),
    ha="center",
    va="center",
    fontsize=7.2,
    color=epistemic_color,
    bbox=dict(
        boxstyle="round,pad=0.22",
        fc="white",
        ec=epistemic_color,
        lw=0.8
    ),
    arrowprops=dict(
        arrowstyle="->",
        color=epistemic_color,
        lw=0.8,
        shrinkA=2,
        shrinkB=2
    )
)

axC.annotate(
    "mostly\naleatoric",
    xy=(1.0, 83),
    xytext=(1.47, 63),
    ha="center",
    va="center",
    fontsize=7.2,
    color=aleatoric_color,
    bbox=dict(
        boxstyle="round,pad=0.22",
        fc="white",
        ec=aleatoric_color,
        lw=0.8
    ),
    arrowprops=dict(
        arrowstyle="->",
        color=aleatoric_color,
        lw=0.8,
        shrinkA=2,
        shrinkB=2
    )
)

# Kéo legend lên cao hơn một chút
axC.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.20),
    ncol=2,
    frameon=False,
    handlelength=1.2,
    columnspacing=1.2
)

# ============================================================
# 8. Bottom interpretation strip
# ============================================================

# Hạ đường line phân cách xuống thấp hơn
fig.add_artist(Rectangle(
    (0.055, 0.115),
    0.94,
    0.002,
    transform=fig.transFigure,
    color="#BDBDBD",
    lw=0
))

# Hạ dòng interpretation xuống thấp hơn để tách xa legend
fig.text(
    0.50,
    0.055,
    "Observation: gains are non-uniform  →  Negative result: high-focused UGA fails  →  Explanation: D_high is aleatoric-heavy",
    ha="center",
    va="center",
    fontsize=8.0,
    color=text_gray
)

# ============================================================
# 9. Save outputs
# ============================================================

out_dir = Path("outputs_springer")
out_dir.mkdir(exist_ok=True)

png_path = out_dir / "diagnostic_intervention_gap_springer_v2.png"
pdf_path = out_dir / "diagnostic_intervention_gap_springer_v2.pdf"
svg_path = out_dir / "diagnostic_intervention_gap_springer_v2.svg"

fig.savefig(png_path, dpi=600, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(svg_path, bbox_inches="tight")

plt.show()

print(f"Saved:\n- {png_path}\n- {pdf_path}\n- {svg_path}")