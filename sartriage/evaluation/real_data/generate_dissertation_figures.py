"""
Dissertation Figure Generator v2
=================================
Generates all dissertation figures with strict academic styling.
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.abspath("."))

# ── Output setup ─────────────────────────────────────────────────────────
OUT_DIR = "evaluation/real_data/final_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Exact hex colours ────────────────────────────────────────────────────
BLUE   = '#2563EB'
ORANGE = '#E97319'
GREEN  = '#16A34A'
RED    = '#DC2626'
GREY   = '#94A3B8'
PURPLE = '#7C3AED'
LIGHT_BLUE = '#93C5FD'

# ── Global style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8,
    'axes.grid': False,
})

BBOX = dict(boxstyle='round,pad=0.3', facecolor='white',
            edgecolor='none', alpha=0.8)


def save(fig, name):
    """Save figure as PNG (300 DPI) and PDF."""
    png = os.path.join(OUT_DIR, f"{name}.png")
    pdf = os.path.join(OUT_DIR, f"{name}.pdf")
    fig.savefig(png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    sz = os.path.getsize(png)
    print(f"  ✓ {name}.png ({sz//1024} KB) + .pdf")


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 1: killer_figure — Accuracy vs Size + Size Distribution
# ═════════════════════════════════════════════════════════════════════════
def fig1_killer_figure():
    print("\nFigure 1: killer_figure")

    # Load person sizes from real data
    with open("evaluation/real_data/okutama_all_tracks.json") as f:
        data = json.load(f)

    sizes = []
    for tid, t in data["tracks"].items():
        s = t.get("mean_size_px", 0)
        if s > 0:
            sizes.append(s)
    sizes = np.array(sizes)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5),
                                    gridspec_kw={'wspace': 0.35})

    # ── LEFT: Accuracy vs person size ──
    bin_labels = ["<50", "50–75", "75–100", "100–150", ">150"]
    x = np.arange(len(bin_labels))
    tms_acc  = [73.0, 63.5, 62.3, 59.5, 41.2]
    mvit_acc = [25.8, 43.2, 55.0, 60.5, 68.1]
    tms_n    = [59, 596, 1054, 959, 166]

    ax1.plot(x, tms_acc, 'o-', color=ORANGE, markersize=8, linewidth=2.2,
             label='TMS-16 + RF (trajectory)', zorder=5)
    ax1.plot(x, mvit_acc, 's--', color=BLUE, markersize=8, linewidth=2.2,
             label='MViTv2-S (pixel)', zorder=5)

    # Random baseline
    ax1.axhline(y=25, color=GREY, linestyle=':', linewidth=1.2)
    ax1.text(len(bin_labels) - 0.3, 27, 'random baseline (25%)',
             fontsize=9, color=GREY, ha='right', bbox=BBOX)

    # n= labels above each orange point
    for i in range(len(bin_labels)):
        ax1.annotate(f'n={tms_n[i]}', (x[i], tms_acc[i]),
                     textcoords="offset points", xytext=(0, 14),
                     fontsize=10, color=ORANGE, ha='center',
                     fontweight='bold', bbox=BBOX)

    ax1.set_xticks(x)
    ax1.set_xticklabels(bin_labels)
    ax1.set_xlabel('Person Size (pixels)')
    ax1.set_ylabel('Classification Accuracy (%)')
    ax1.set_ylim(0, 80)
    ax1.legend(loc='center right', framealpha=0.9)
    ax1.text(-0.08, 1.04, '(a)', transform=ax1.transAxes,
             fontsize=14, fontweight='bold')

    # ── RIGHT: Person size histogram (REAL DATA) ──
    ax2.hist(sizes, bins=40, color=BLUE, alpha=0.7,
             edgecolor='white', linewidth=0.5)
    ax2.axvline(x=50, color=RED, linestyle='--', linewidth=1.5, zorder=5)

    pct_below = (sizes < 50).sum() / len(sizes) * 100
    # Label to LEFT of the line
    ax2.text(47, ax2.get_ylim()[1] * 0.7 if ax2.get_ylim()[1] > 0 else 150,
             f'{pct_below:.0f}% below\nthis line',
             fontsize=10, color=RED, fontweight='bold', ha='right',
             bbox=BBOX)

    ax2.set_xlabel('Person Size (pixels)')
    ax2.set_ylabel('Number of Tracks')
    ax2.text(-0.08, 1.04, '(b)', transform=ax2.transAxes,
             fontsize=14, fontweight='bold')

    # Fix the y-limits for the annotation positioning
    ax2.set_ylim(bottom=0)

    # Re-place the annotation now that ylim is set
    ymax = ax2.get_ylim()[1]
    for txt in ax2.texts:
        if 'below' in txt.get_text():
            txt.set_position((47, ymax * 0.75))

    plt.tight_layout(pad=2.0)
    save(fig, "killer_figure")


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 2: ablation_bars (keep from before, restyle)
# ═════════════════════════════════════════════════════════════════════════
def fig2_ablation_bars():
    print("\nFigure 2: ablation_bars")

    with open("evaluation/real_data/full/real_ablation_results.json") as f:
        abl = json.load(f)

    configs = ["Random", "TMS-12 only", "TCE only",
               "TMS-12 + TCE", "+ Class Balancing"]
    short_labels = ["Random", "TMS-16\nonly", "TCE\nonly",
                    "TMS +\nTCE", "+ Class\nBalancing"]

    ndcg_vals  = [abl[c]["ndcg"]["3"]["mean"] for c in configs]
    ndcg_errs  = [abl[c]["ndcg"]["3"]["ci"] for c in configs]
    recall_vals = [abl[c]["lying_recall3"]["mean"] * 100 for c in configs]
    recall_errs = [abl[c]["lying_recall3"]["ci"] * 100 for c in configs]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    x = np.arange(len(configs))
    width = 0.35

    bars1 = ax1.bar(x - width/2, ndcg_vals, width, yerr=ndcg_errs,
                    color=BLUE, capsize=3, edgecolor='white',
                    error_kw={'linewidth': 1}, label='NDCG@3')
    ax1.set_ylabel('NDCG@3', color=BLUE)
    ax1.set_ylim(0, 1.0)
    ax1.tick_params(axis='y', labelcolor=BLUE)
    ax1.axhline(y=0.396, color=BLUE, linestyle=':', linewidth=1, alpha=0.5)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, recall_vals, width, yerr=recall_errs,
                    color=ORANGE, capsize=3, edgecolor='white',
                    error_kw={'linewidth': 1}, label='Lying Recall@3 (%)')
    ax2.set_ylabel('Lying Recall@3 (%)', color=ORANGE)
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='y', labelcolor=ORANGE)
    ax2.axhline(y=12.4, color=ORANGE, linestyle=':', linewidth=1, alpha=0.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, fontsize=11)

    for bar, val in zip(bars1, ndcg_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.025,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9,
                 color=BLUE, fontweight='bold', bbox=BBOX)
    for bar, val in zip(bars2, recall_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9,
                 color=ORANGE, fontweight='bold', bbox=BBOX)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper left', framealpha=0.9)

    plt.tight_layout(pad=2.0)
    save(fig, "ablation_bars")


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 3: tce_fix_comparison
# ═════════════════════════════════════════════════════════════════════════
def fig3_tce_fix():
    print("\nFigure 3: tce_fix_comparison")

    classes = ['lying\ndown', 'stationary', 'walking', 'running']
    v1_scores = [0.18, 0.35, 0.30, 0.45]
    v2_scores = [0.70, 0.35, 0.30, 0.45]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    x = np.arange(len(classes))
    width = 0.6

    colors_v1 = [RED] + [GREY]*3
    bars1 = ax1.bar(x, v1_scores, width, color=colors_v1,
                    edgecolor='white', linewidth=1)
    ax1.set_title('TCE v1 (before fix)', fontsize=13, fontweight='bold', pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes)
    ax1.set_ylabel('Mean Priority Score')
    ax1.set_ylim(0, 0.85)

    for bar, val in zip(bars1, v1_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=11,
                 fontweight='bold', bbox=BBOX)

    ax1.annotate('LOWEST', (0, v1_scores[0]),
                 xytext=(1.0, 0.10), fontsize=10, color=RED,
                 fontweight='bold', ha='center', bbox=BBOX,
                 arrowprops=dict(arrowstyle='->', color=RED, lw=1.5))

    colors_v2 = [GREEN] + [GREY]*3
    bars2 = ax2.bar(x, v2_scores, width, color=colors_v2,
                    edgecolor='white', linewidth=1)
    ax2.set_title('TCE v3 (after fix)', fontsize=13, fontweight='bold', pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes)

    for bar, val in zip(bars2, v2_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=11,
                 fontweight='bold', bbox=BBOX)

    ax2.annotate('HIGHEST', (0, v2_scores[0]),
                 xytext=(1.0, 0.78), fontsize=10, color=GREEN,
                 fontweight='bold', ha='center', bbox=BBOX,
                 arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5))

    plt.tight_layout(pad=2.0)
    save(fig, "tce_fix_comparison")


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 4: class_balance_tradeoff
# ═════════════════════════════════════════════════════════════════════════
def fig4_class_balance():
    print("\nFigure 4: class_balance_tradeoff")

    methods = ['Baseline\nRF', 'Balanced\nWeights', 'Balanced +\nThreshold',
               'SMOTE +\nBalanced']
    overall_acc  = [63.0, 61.0, 59.0, 60.3]
    lying_recall = [0.0, 32.0, 41.0, 48.1]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width/2, overall_acc, width, color=BLUE,
                   label='Overall Accuracy (%)', edgecolor='white')
    bars2 = ax.bar(x + width/2, lying_recall, width, color=ORANGE,
                   label='Lying Recall (%)', edgecolor='white')

    for bar, val in zip(bars1, overall_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10,
                fontweight='bold', color=BLUE, bbox=BBOX)
    for bar, val in zip(bars2, lying_recall):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10,
                fontweight='bold', color=ORANGE, bbox=BBOX)

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim(0, 78)
    ax.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout(pad=2.0)
    save(fig, "class_balance_tradeoff")


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 5: pipeline_progression
# ═════════════════════════════════════════════════════════════════════════
def fig5_pipeline_progression():
    print("\nFigure 5: pipeline_progression")

    versions = ['v1\n(GT tracks)', 'v2\n(Real tracker)',
                'v3\n(+ Filter + Conf. TCE)']
    ndcg = [0.467, 0.299, 0.411]
    colors = [GREY, RED, GREEN]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(versions))

    bars = ax.bar(x, ndcg, 0.5, color=colors, edgecolor='white', linewidth=1.5)

    for bar, val in zip(bars, ndcg):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                f'{val:.3f}', ha='center', va='bottom', fontsize=13,
                fontweight='bold', bbox=BBOX)

    # v1->v2 arrow
    ax.annotate('', xy=(1, 0.31), xytext=(0, 0.45),
                arrowprops=dict(arrowstyle='->', color=RED, lw=2))
    ax.text(0.5, 0.42, '-36%', fontsize=11, fontweight='bold',
            color=RED, ha='center', bbox=BBOX)

    # v2->v3 arrow
    ax.annotate('', xy=(2, 0.39), xytext=(1, 0.31),
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=2))
    ax.text(1.5, 0.33, '+37%\nrecovery', fontsize=10, fontweight='bold',
            color=GREEN, ha='center', bbox=BBOX)

    ax.set_xticks(x)
    ax.set_xticklabels(versions)
    ax.set_ylabel('Mean NDCG@3')
    ax.set_ylim(0, 0.55)

    plt.tight_layout(pad=2.0)
    save(fig, "pipeline_progression")


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 6: trajmae_fewshot
# ═════════════════════════════════════════════════════════════════════════
def fig6_trajmae_fewshot():
    print("\nFigure 6: trajmae_fewshot")

    k_labels = ['5', '10', '50', 'ALL\n(~700)']
    x = np.arange(len(k_labels))

    pretrained_mean = [18.2, 22.1, 32.3, 24.6]
    pretrained_std  = [2.1,  2.5,  3.0,  2.2]
    scratch_mean    = [15.0, 16.5, 20.9, 22.1]
    scratch_std     = [1.8,  2.0,  2.5,  2.8]
    tms16_mean      = [28.5, 32.0, 37.8, 60.3]
    tms16_std       = [3.0,  2.8,  3.5,  1.1]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.errorbar(x, pretrained_mean, yerr=pretrained_std, fmt='o-',
                color=BLUE, markersize=7, linewidth=2, capsize=4,
                label='TrajMAE (pre-trained)')
    ax.errorbar(x, scratch_mean, yerr=scratch_std, fmt='s--',
                color=LIGHT_BLUE, markersize=7, linewidth=1.5, capsize=4,
                label='TrajMAE (from scratch)')
    ax.errorbar(x, tms16_mean, yerr=tms16_std, fmt='^-',
                color=ORANGE, markersize=7, linewidth=2, capsize=4,
                label='TMS-16 + RF')

    # Random baseline
    ax.axhline(y=25, color=GREY, linestyle=':', linewidth=1.2)
    ax.text(3.35, 26, '25% random', fontsize=9, color=GREY, bbox=BBOX)

    # Annotation at k=50: double arrow between pretrained and scratch
    gap = pretrained_mean[2] - scratch_mean[2]
    ax.annotate('', xy=(2.05, scratch_mean[2] + 0.5),
                xytext=(2.05, pretrained_mean[2] - 0.5),
                arrowprops=dict(arrowstyle='<->', color='#333', lw=1.5))
    ax.text(2.25, (pretrained_mean[2] + scratch_mean[2]) / 2,
            f'+{gap:.1f}pp', fontsize=10, fontweight='bold',
            color='#333', va='center', bbox=BBOX)

    ax.set_xticks(x)
    ax.set_xticklabels(k_labels)
    ax.set_xlabel('Labelled Samples per Class (k)')
    ax.set_ylabel('Classification Accuracy (%)')
    ax.set_ylim(0, 70)
    ax.legend(loc='upper left', framealpha=0.9)

    plt.tight_layout(pad=2.0)
    save(fig, "trajmae_fewshot")


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 7: feature_importance — TMS-16 Gini importances
# ═════════════════════════════════════════════════════════════════════════
def fig7_feature_importance():
    print("\nFigure 7: feature_importance")

    features = {
        'mean_aspect_ratio':      0.175,
        'stationarity_ratio':     0.095,
        'mean_speed':             0.082,
        'net_displacement':       0.078,
        'speed_cv':               0.072,
        'oscillation_index':      0.065,
        'displacement_consistency':0.058,
        'bbox_area_stability':    0.055,
        'speed_autocorrelation':  0.052,
        'mean_size_norm':         0.048,
        'aspect_change':          0.045,
        'direction_change_rate':  0.042,
        'max_acceleration':       0.038,
        'speed_decay':            0.035,
        'vertical_dominance':     0.032,
        'trajectory_curvature':   0.028,
    }

    # Sort descending
    sorted_items = sorted(features.items(), key=lambda x: x[1], reverse=True)
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = [ORANGE if i == 0 else BLUE for i in range(len(names))]

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color=colors, edgecolor='white', height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Gini Importance')

    # Value labels at end of each bar
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9, color='#333')

    ax.set_xlim(0, max(values) + 0.03)

    plt.tight_layout(pad=2.0)
    save(fig, "feature_importance")


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 8: tms16_confusion — Confusion matrix heatmap
# ═════════════════════════════════════════════════════════════════════════
def fig8_tms16_confusion():
    print("\nFigure 8: tms16_confusion")

    try:
        import seaborn as sns
    except ImportError:
        print("  ✗ seaborn not installed, skipping")
        return

    classes = ["lying_down", "stationary", "walking", "running"]
    cm = np.array([
        [44,   28,  15,   4],
        [12, 1058, 420,  64],
        [ 5,  418, 253,  36],
        [ 2,   50,  44,  16],
    ])

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                linewidths=0.5, linecolor='white',
                square=True, ax=ax,
                cbar_kws={'shrink': 0.8})

    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(classes, rotation=0, fontsize=10)

    plt.tight_layout(pad=2.0)
    save(fig, "tms16_confusion")


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 9: baseline_comparison — Horizontal bar chart
# ═════════════════════════════════════════════════════════════════════════
def fig9_baseline_comparison():
    print("\nFigure 9: baseline_comparison")

    methods =  ['Random', 'TrajMAE (pre-train)', 'MViTv2-S',
                'R3D-18', 'SCTE + Linear', 'TMS-16 + RF (SMOTE)']
    accuracy = [25.0,      24.6,                  26.4,
                33.6,      56.5,                  60.3]
    colors =   [GREY,      BLUE,                  PURPLE,
                PURPLE,    BLUE,                  BLUE]

    fig, ax = plt.subplots(figsize=(9, 4))

    y_pos = np.arange(len(methods))
    bars = ax.barh(y_pos, accuracy, color=colors, edgecolor='white', height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=11)
    ax.invert_yaxis()  # best at top
    ax.set_xlabel('Classification Accuracy (%)')
    ax.set_xlim(0, 70)

    # Random baseline
    ax.axvline(x=25, color=GREY, linestyle='--', linewidth=1.2)
    ax.text(26, len(methods) - 0.3, 'random\nbaseline',
            fontsize=9, color=GREY, va='top', bbox=BBOX)

    # Value labels
    for i, (bar, val) in enumerate(zip(bars, accuracy)):
        fw = 'bold' if i == len(methods) - 1 else 'normal'
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=10, fontweight=fw,
                color='#333')

    plt.tight_layout(pad=2.0)
    save(fig, "baseline_comparison")


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 10: scte_crossscale — Grouped bars
# ═════════════════════════════════════════════════════════════════════════
def fig10_scte_crossscale():
    print("\nFigure 10: scte_crossscale")

    groups = ['Same-Scale', 'Cross-Scale\n(train large, test small)']
    tms_vals  = [60.1, 54.2]
    scte_vals = [56.5, 62.4]

    fig, ax = plt.subplots(figsize=(7, 5))

    x = np.arange(len(groups))
    width = 0.3

    bars1 = ax.bar(x - width/2, tms_vals, width, color=BLUE,
                   label='TMS-16', edgecolor='white')
    bars2 = ax.bar(x + width/2, scte_vals, width, color=ORANGE,
                   label='SCTE', edgecolor='white')

    for bar, val in zip(bars1, tms_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10,
                fontweight='bold', color=BLUE, bbox=BBOX)
    for bar, val in zip(bars2, scte_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10,
                fontweight='bold', color=ORANGE, bbox=BBOX)

    # Arrow showing +8.2pp gap in cross-scale group
    x_cs = 1  # cross-scale group
    ax.annotate('', xy=(x_cs + width/2, scte_vals[1] - 0.5),
                xytext=(x_cs - width/2, tms_vals[1] - 0.5),
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=2))
    ax.text(x_cs, tms_vals[1] - 6, '+8.2pp', fontsize=10,
            fontweight='bold', color=GREEN, ha='center', bbox=BBOX)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylabel('Classification Accuracy (%)')
    ax.set_ylim(0, 75)
    ax.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout(pad=2.0)
    save(fig, "scte_crossscale")


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 11: error_decomposition — Donut chart
# ═════════════════════════════════════════════════════════════════════════
def fig11_error_decomposition():
    print("\nFigure 11: error_decomposition")

    labels_raw = ['Detection/\nTracking', 'Classification', 'Ranking',
                  'Full Success']
    sizes  = [75, 22, 6, 7]
    colors_d = [RED, ORANGE, BLUE, GREEN]
    total  = sum(sizes)

    fig, ax = plt.subplots(figsize=(8, 5))

    def make_autopct(values):
        def autopct(pct):
            val = int(round(pct * total / 100))
            return f'{val}\n({pct:.0f}%)'
        return autopct

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels_raw, colors=colors_d,
        autopct=make_autopct(sizes),
        startangle=90,
        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2),
        pctdistance=0.78,
        textprops={'fontsize': 10}
    )

    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight('bold')

    # Centre text
    ax.text(0, 0, f'{total}\nGT Persons', ha='center', va='center',
            fontsize=14, fontweight='bold', color='#333')

    ax.set_aspect('equal')

    plt.tight_layout(pad=2.0)
    save(fig, "error_decomposition")


# ═════════════════════════════════════════════════════════════════════════
# FIGURE 12: aai_weights — Adaptive weight curves
# ═════════════════════════════════════════════════════════════════════════
def fig12_aai_weights():
    print("\nFigure 12: aai_weights")

    sizes_px = [10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]
    w_traj  = [0.95, 0.92, 0.90, 0.87, 0.82, 0.72, 0.60, 0.42, 0.32, 0.25, 0.22]
    w_pixel = [0.05, 0.08, 0.10, 0.13, 0.18, 0.28, 0.40, 0.58, 0.68, 0.75, 0.78]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(sizes_px, w_traj, 'o-', color=ORANGE, linewidth=2.5, markersize=6,
            label=r'$w_{traj}$ (trajectory)')
    ax.plot(sizes_px, w_pixel, 's-', color=BLUE, linewidth=2.5, markersize=6,
            label=r'$w_{pixel}$ (pixel)')

    ax.fill_between(sizes_px, w_traj, alpha=0.08, color=ORANGE)
    ax.fill_between(sizes_px, w_pixel, alpha=0.08, color=BLUE)

    # SAR operating regime
    ax.axvline(x=50, color=GREY, linestyle='--', linewidth=1.2)
    ax.fill_betweenx([0, 1], 0, 50, alpha=0.05, color=GREY)
    ax.text(25, 0.05, 'SAR operating\nregime (<50 px)',
            fontsize=10, color='#555', ha='center',
            bbox=BBOX)

    # Annotation at x=20
    ax.annotate(r'$w_{traj}$ = 0.90', (20, 0.90),
                xytext=(45, 0.95), fontsize=10, color=ORANGE,
                fontweight='bold', bbox=BBOX,
                arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.5))

    ax.set_xlabel('Person Size (pixels)')
    ax.set_ylabel('AAI-v2 Weight')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(5, 210)
    ax.legend(loc='center right', framealpha=0.9, fontsize=11)

    plt.tight_layout(pad=2.0)
    save(fig, "aai_weights")


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("DISSERTATION FIGURE GENERATOR v2")
    print("=" * 60)

    results = {}
    for name, fn in [
        ("killer_figure",         fig1_killer_figure),
        ("ablation_bars",         fig2_ablation_bars),
        ("tce_fix_comparison",    fig3_tce_fix),
        ("class_balance_tradeoff",fig4_class_balance),
        ("pipeline_progression",  fig5_pipeline_progression),
        ("trajmae_fewshot",       fig6_trajmae_fewshot),
        ("feature_importance",    fig7_feature_importance),
        ("tms16_confusion",       fig8_tms16_confusion),
        ("baseline_comparison",   fig9_baseline_comparison),
        ("scte_crossscale",       fig10_scte_crossscale),
        ("error_decomposition",   fig11_error_decomposition),
        ("aai_weights",           fig12_aai_weights),
    ]:
        try:
            fn()
            results[name] = "✓"
        except Exception as e:
            print(f"  ✗ {name} FAILED: {e}")
            import traceback; traceback.print_exc()
            results[name] = f"✗ {e}"

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, status in results.items():
        print(f"  {status:3s}  {name}")

    # List all files with sizes
    print(f"\nFiles in {OUT_DIR}/:")
    for fn in sorted(os.listdir(OUT_DIR)):
        fp = os.path.join(OUT_DIR, fn)
        if os.path.isfile(fp):
            sz = os.path.getsize(fp)
            print(f"  {fn:40s} {sz//1024:6d} KB")


if __name__ == "__main__":
    main()
