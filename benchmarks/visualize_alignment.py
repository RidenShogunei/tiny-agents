#!/usr/bin/env python3
"""
GoalMisAlignBench Visualizer
============================
Generates loss decomposition bar charts from summary.json.

Usage:
  python visualize.py /path/to/run_dir
  python visualize.py /path/to/run_dir --output /path/to/fig.png
  python visualize.py --aggregate dir1 dir2 ...
"""

import os
import sys
import json
import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "text.color": "#e6edf3",
    "grid.color": "#21262d",
    "grid.linewidth": 0.5,
})


def load_summary(path):
    with open(path) as f:
        return json.load(f)


def plot_overall_loss_decomposition(ax, summary):
    """Main bar chart: A, C, B accuracy + loss decomposition."""
    o = summary["overall"]
    labels = ["Mode A\n(Single)\nBaseline", "Mode C\n(Oracle)\nExtra Step Loss",
              "Mode B\n(Real)\nTotal Loss"]
    accs = [o["acc_A"], o["acc_C"], o["acc_B"]]
    colors = ["#58a6ff", "#d29922", "#f85149"]

    bars = ax.bar(labels, accs, color=colors, width=0.5, edgecolor="none",
                  zorder=3)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{acc:.0f}%", ha="center", va="bottom", fontweight="bold",
                color="#e6edf3", fontsize=13)

    # Loss arrows
    ax.annotate("", xy=(1, o["acc_C"]), xytext=(0, o["acc_A"]),
                arrowprops=dict(arrowstyle="->", color="#d29922", lw=2.5))
    ax.text(0.5, (o["acc_A"] + o["acc_C"]) / 2 + 2,
            f"A−C\n{abs(o['extra_step_loss']):.0f}pp",
            ha="center", va="bottom", color="#d29922", fontsize=9, fontweight="bold")

    ax.annotate("", xy=(2, o["acc_B"]), xytext=(1, o["acc_C"]),
                arrowprops=dict(arrowstyle="->", color="#f85149", lw=2.5))
    ax.text(1.5, (o["acc_C"] + o["acc_B"]) / 2 + 2,
            f"C−B\n{abs(o['misalign_loss']):.0f}pp",
            ha="center", va="bottom", color="#f85149", fontsize=9, fontweight="bold")

    ax.set_title("Goal Misalignment: Loss Decomposition", fontweight="bold", pad=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)


def plot_by_type(summary, ax):
    """Grouped bar chart per misalignment type."""
    types = sorted(summary["by_type"].keys(), key=lambda t: -summary["by_type"][t]["misalign_loss"])
    short_names = {t: t.replace("_", "\n") for t in types}
    x = np.arange(len(types))
    width = 0.25

    acc_a = [summary["by_type"][t]["acc_A"] for t in types]
    acc_c = [summary["by_type"][t]["acc_C"] for t in types]
    acc_b = [summary["by_type"][t]["acc_B"] for t in types]

    b_a = ax.bar(x - width, acc_a, width, label="Mode A (Single)", color="#58a6ff", edgecolor="none", zorder=3)
    b_c = ax.bar(x, acc_c, width, label="Mode C (Oracle)", color="#d29922", edgecolor="none", zorder=3)
    b_b = ax.bar(x + width, acc_b, width, label="Mode B (Real)", color="#f85149", edgecolor="none", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([short_names[t] for t in types], fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.yaxis.grid(True, zorder=0)
    ax.legend(loc="upper right", framealpha=0.3)
    ax.set_title("Accuracy by Misalignment Type", fontweight="bold", pad=8)

    # Mark highest misalignment types
    for i, t in enumerate(types):
        ml = summary["by_type"][t]["misalign_loss"]
        if ml > 5:
            ax.annotate(f"misalign\n{abs(ml):.0f}pp",
                       xy=(i + width, summary["by_type"][t]["acc_C"]),
                       xytext=(i + width, summary["by_type"][t]["acc_C"] + 12),
                       ha="center", va="bottom", color="#f85149",
                       fontsize=7, fontweight="bold")


def plot_loss_waterfall(summary, ax):
    """Stacked bar showing how baseline drops to real delegation."""
    o = summary["overall"]
    labels = ["Extra Step\nLoss (A−C)", "Goal Misalign\nLoss (C−B)",
              "Total\nLoss (A−B)"]
    losses = [abs(o["extra_step_loss"]), abs(o["misalign_loss"]),
              abs(o["total_loss"])]
    colors = ["#d29922", "#f85149", "#6e7681"]
    bottoms = [0, abs(o["extra_step_loss"]), 0]

    bars = ax.bar(labels, losses, color=colors, width=0.5, edgecolor="none", zorder=3)
    for bar, loss in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{loss:.0f}pp", ha="center", va="bottom",
                fontweight="bold", color="#e6edf3", fontsize=12)

    ax.set_title("Loss Waterfall", fontweight="bold", pad=8)
    ax.set_ylabel("Accuracy Drop (pp)")
    ax.set_ylim(0, max(losses) * 1.25)
    ax.yaxis.grid(True, zorder=0)


def plot_verdict_pie(summary, ax):
    """Pie chart of case verdict patterns."""
    verdicts = summary.get("verdicts", [])
    label_map = {
        "MISALIGNMENT": "MISALIGNMENT\n(C✓B✗)",
        "extra_step_hurts": "Extra Step Hurts\n(A✓C✗B✗)",
        "all_wrong": "All Wrong\n(A✗B✗C✗)",
        "delegation_helps": "Delegation Helps\n(A✗C✓)",
        "oracle_confuses": "Oracle Confuses\n(A✓B✓C✗)",
        None: "Baseline Correct\n(A✓B✓C✓)",
    }
    counts = {}
    for v in verdicts:
        label = label_map.get(v["label"], v["label"] or "other")
        counts[label] = counts.get(label, 0) + 1

    if not counts:
        ax.text(0.5, 0.5, "No verdict data", ha="center", va="center",
                transform=ax.transAxes, color="#8b949e")
        ax.set_axis_off()
        return

    labels, values = zip(*sorted(counts.items(), key=lambda x: -x[1]))
    color_map = {
        "MISALIGNMENT\n(C✓B✗)": "#f85149",
        "Extra Step Hurts\n(A✓C✗B✗)": "#d29922",
        "All Wrong\n(A✗B✗C✗)": "#6e7681",
        "Delegation Helps\n(A✗C✓)": "#3fb950",
        "Oracle Confuses\n(A✓B✓C✗)": "#db6d28",
        "Baseline Correct\n(A✓B✓C✓)": "#58a6ff",
    }
    colors = [color_map.get(l, "#8b949e") for l in labels]

    wedges, texts, autotexts = ax.pie(
        values, labels=None, colors=colors, autopct="%1.0f%%",
        startangle=90, pctdistance=0.75,
        wedgeprops={"edgecolor": "#161b22", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")
        at.set_color("#e6edf3")

    ax.legend(wedges, [f"{l} ({v})" for l, v in zip(labels, values)],
              loc="center left", bbox_to_anchor=(1, 0.5),
              framealpha=0.3, fontsize=8)
    ax.set_title("Case Verdict Distribution", fontweight="bold", pad=8)


def plot_difficulty(summary, ax):
    """Accuracy by difficulty level."""
    diff_order = ["easy", "medium", "hard"]
    available = sorted(summary["by_difficulty"].keys())
    diffs = [d for d in diff_order if d in available]
    x = np.arange(len(diffs))
    width = 0.25

    acc_a = [summary["by_difficulty"][d]["acc_A"] for d in diffs]
    acc_c = [summary["by_difficulty"][d]["acc_C"] for d in diffs]
    acc_b = [summary["by_difficulty"][d]["acc_B"] for d in diffs]

    ax.bar(x - width, acc_a, width, label="Mode A", color="#58a6ff", edgecolor="none", zorder=3)
    ax.bar(x, acc_c, width, label="Mode C", color="#d29922", edgecolor="none", zorder=3)
    ax.bar(x + width, acc_b, width, label="Mode B", color="#f85149", edgecolor="none", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in diffs])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.yaxis.grid(True, zorder=0)
    ax.legend(loc="upper right", framealpha=0.3)
    ax.set_title("Accuracy by Case Difficulty", fontweight="bold", pad=8)


def plot_single(summary, output_path=None):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"GoalMisAlignBench — {summary['n_cases']} cases | "
        f"Baseline={summary['overall']['acc_A']:.0f}% | "
        f"Misalign={summary['overall']['misalign_loss']:+.0f}pp | "
        f"ExtraStep={summary['overall']['extra_step_loss']:+.0f}pp",
        fontsize=13, fontweight="bold", color="#e6edf3", y=1.01
    )

    plot_overall_loss_decomposition(axes[0, 0], summary)
    plot_by_type(summary, axes[0, 1])
    plot_loss_waterfall(summary, axes[0, 2])
    plot_verdict_pie(summary, axes[1, 0])
    plot_difficulty(summary, axes[1, 1])

    # Right-bottom: summary table
    ax = axes[1, 2]
    ax.axis("off")
    o = summary["overall"]
    table_data = [
        ["Metric", "Value"],
        ["Cases", str(summary["n_cases"])],
        ["Mode A (baseline)", f"{o['acc_A']:.0f}%"],
        ["Mode C (oracle)", f"{o['acc_C']:.0f}%"],
        ["Mode B (real)", f"{o['acc_B']:.0f}%"],
        ["", ""],
        ["A−C (extra step)", f"{o['extra_step_loss']:+.0f}pp"],
        ["C−B (misalign)", f"{o['misalign_loss']:+.0f}pp"],
        ["A−B (total)", f"{o['total_loss']:+.0f}pp"],
    ]
    tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                   cellLoc="center", loc="center",
                   bbox=[0.1, 0.2, 0.8, 0.7])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#21262d")
            cell.set_text_props(color="#e6edf3", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#1c2128")
        else:
            cell.set_facecolor("#161b22")
        cell.set_edgecolor("#30363d")
        cell.set_text_props(color="#c9d1d9")
    ax.set_title("Summary", fontweight="bold", pad=8)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor="#0d1117")
        print(f"Saved: {output_path}")
    else:
        plt.show()


def plot_aggregate(agg, output_path=None):
    """Plot aggregate across multiple runs with error bars."""
    n_runs = agg["n_runs"]
    overall = agg["overall"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"GoalMisAlignBench — {n_runs} runs aggregated", fontsize=13, fontweight="bold", color="#e6edf3")

    # Overall bar
    ax = axes[0]
    labels = ["Mode A\n(Single)", "Mode C\n(Oracle)", "Mode B\n(Real)"]
    keys = ["acc_A", "acc_C", "acc_B"]
    colors = ["#58a6ff", "#d29922", "#f85149"]
    means = [overall[k]["mean"] for k in keys]
    stds  = [overall[k]["std"]  for k in keys]
    bars = ax.bar(labels, means, yerr=stds, color=colors, width=0.5,
                  edgecolor="none", capsize=5, zorder=3, error_kw={"elinewidth": 2})
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{m:.1f}%", ha="center", va="bottom", fontweight="bold", color="#e6edf3")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.yaxis.grid(True, zorder=0)
    ax.set_title("Overall Accuracy (mean ± std)", fontweight="bold")

    # Loss decomposition
    ax = axes[1]
    loss_labels = ["A−C\n(Extra Step)", "C−B\n(Misalign)", "A−B\n(Total)"]
    loss_keys  = ["extra_step_loss", "misalign_loss", "total_loss"]
    loss_means = [abs(overall[k]["mean"]) for k in loss_keys]
    loss_stds  = [overall[k]["std"] for k in loss_keys]
    loss_colors = ["#d29922", "#f85149", "#6e7681"]
    bars = ax.bar(loss_labels, loss_means, yerr=loss_stds, color=loss_colors,
                  width=0.5, edgecolor="none", capsize=5, zorder=3,
                  error_kw={"elinewidth": 2})
    for bar, m in zip(bars, loss_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{m:.1f}pp", ha="center", va="bottom", fontweight="bold", color="#e6edf3")
    ax.set_ylabel("Accuracy Drop (pp)")
    ax.set_ylim(0, max(loss_means) * 1.35)
    ax.yaxis.grid(True, zorder=0)
    ax.set_title("Loss Decomposition (mean ± std)", fontweight="bold")

    # Per-type with error bars
    ax = axes[2]
    types = sorted(agg["by_type"].keys())
    x = np.arange(len(types))
    width = 0.25
    means_a = [agg["by_type"][t]["acc_A"]["mean"] for t in types]
    means_c = [agg["by_type"][t]["acc_C"]["mean"] for t in types]
    means_b = [agg["by_type"][t]["acc_B"]["mean"] for t in types]
    std_a   = [agg["by_type"][t]["acc_A"]["std"] for t in types]
    std_c   = [agg["by_type"][t]["acc_C"]["std"] for t in types]
    std_b   = [agg["by_type"][t]["acc_B"]["std"] for t in types]

    short = [t.replace("_", "\n") for t in types]
    ax.bar(x - width, means_a, width, yerr=std_a, label="Mode A", color="#58a6ff",
           edgecolor="none", capsize=3, zorder=3, error_kw={"elinewidth": 1.5})
    ax.bar(x, means_c, width, yerr=std_c, label="Mode C", color="#d29922",
           edgecolor="none", capsize=3, zorder=3, error_kw={"elinewidth": 1.5})
    ax.bar(x + width, means_b, width, yerr=std_b, label="Mode B", color="#f85149",
           edgecolor="none", capsize=3, zorder=3, error_kw={"elinewidth": 1.5})
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 115)
    ax.yaxis.grid(True, zorder=0)
    ax.legend(loc="upper right", framealpha=0.3)
    ax.set_title("By Type (mean ± std)", fontweight="bold")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        print(f"Saved: {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Run directories with summary.json or a single aggregate dir")
    parser.add_argument("--output", "-o", help="Output image path")
    parser.add_argument("--aggregate", "-a", action="store_true", help="Aggregate multiple runs")
    args = parser.parse_args()

    if args.aggregate:
        import glob
        dirs = []
        for p in args.paths:
            if os.path.isdir(p):
                summary_file = os.path.join(p, "summary.json")
                if os.path.exists(summary_file):
                    dirs.append(p)
                else:
                    subdirs = sorted(glob.glob(os.path.join(p, "run_*")))
                    dirs.extend(subdirs)
        if not dirs:
            print("No summary.json found in any path.")
            sys.exit(1)
        summaries = [load_summary(os.path.join(d, "summary.json")) for d in dirs]

        # Build aggregate manually
        n_runs = len(summaries)
        overall_keys = ["acc_A", "acc_B", "acc_C", "extra_step_loss", "misalign_loss", "total_loss"]
        agg_overall = {}
        for k in overall_keys:
            vals = [s["overall"][k] for s in summaries]
            m = sum(vals) / n_runs
            std = (sum((v - m) ** 2 for v in vals) / n_runs) ** 0.5
            agg_overall[k] = {"mean": round(m, 2), "std": round(std, 2), "runs": vals}

        types = sorted(summaries[0]["by_type"].keys())
        type_keys = ["acc_A", "acc_B", "acc_C", "extra_step_loss", "misalign_loss"]
        agg_by_type = {}
        for t in types:
            agg_by_type[t] = {}
            for k in type_keys:
                vals = [s["by_type"][t][k] for s in summaries if t in s["by_type"]]
                if vals:
                    m = sum(vals) / len(vals)
                    std = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
                    agg_by_type[t][k] = {"mean": round(m, 2), "std": round(std, 2)}

        agg = {"n_runs": n_runs, "overall": agg_overall, "by_type": agg_by_type,
               "run_timestamps": [s["timestamp"] for s in summaries]}
        plot_aggregate(agg, args.output)
    else:
        for p in args.paths:
            if os.path.isdir(p):
                p = os.path.join(p, "summary.json")
            if os.path.exists(p):
                summary = load_summary(p)
                out = args.output
                if not out and len(args.paths) == 1:
                    base = os.path.dirname(p) or "."
                    out = os.path.join(base, "bench_chart.png")
                plot_single(summary, out)
            else:
                print(f"Not found: {p}")
