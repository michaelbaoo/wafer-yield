"""Phase 5: build the research-paper-style report as a PDF.

Every number and figure is read from the outputs folder written by phases 1
to 4, so the report always matches the latest pipeline run.

Usage (from the project root, after phases 1 to 4):
    python scripts/build_report.py
Writes:
    reports/wafer_yield_report.pdf
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "mathtext.fontset": "stix",
})

AUTHOR = "Michael Bao"
REPO = "github.com/michaelbaoo/wafer-yield"
DATE = "November 2026"
TITLE = "Wafer Probe Yield Monitoring and Defect Pareto Analysis on the WM-811K Dataset"

DARK, MID, LIGHT = "#1f3b5c", "#5b84b1", "#b8cbe0"
PATTERNS = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch"]


# ---------------------------------------------------------------- formatting

def n(x) -> str:
    return f"{int(round(x)):,}"


def pct(x, d=1) -> str:
    return f"{100 * x:.{d}f}%"


def millions(x, d=1) -> str:
    return f"{x / 1e6:.{d}f} million"


# ---------------------------------------------------------------- styles

def styles():
    base = dict(fontName="Times-Roman", fontSize=11, leading=14.2)
    return {
        "title": ParagraphStyle("title", fontName="Times-Bold", fontSize=16, leading=20,
                                alignment=TA_CENTER, spaceAfter=8),
        "author": ParagraphStyle("author", fontName="Times-Roman", fontSize=11, leading=14,
                                 alignment=TA_CENTER),
        "abs_head": ParagraphStyle("abs_head", fontName="Times-Bold", fontSize=10.5,
                                   alignment=TA_CENTER, spaceBefore=14, spaceAfter=4),
        "abstract": ParagraphStyle("abstract", fontName="Times-Roman", fontSize=10, leading=13,
                                   alignment=TA_JUSTIFY, leftIndent=36, rightIndent=36),
        "h1": ParagraphStyle("h1", fontName="Times-Bold", fontSize=12.5, leading=16,
                             spaceBefore=14, spaceAfter=5),
        "h2": ParagraphStyle("h2", fontName="Times-BoldItalic", fontSize=11, leading=14,
                             spaceBefore=9, spaceAfter=3),
        "body": ParagraphStyle("body", alignment=TA_JUSTIFY, spaceAfter=6, **base),
        "eq": ParagraphStyle("eq", fontName="Times-Roman", fontSize=11, leading=15,
                             alignment=TA_CENTER, spaceBefore=2, spaceAfter=6),
        "bullet": ParagraphStyle("bullet", alignment=TA_JUSTIFY, leftIndent=18, bulletIndent=6,
                                 spaceAfter=3, **base),
        "caption": ParagraphStyle("caption", fontName="Times-Roman", fontSize=9.5, leading=12,
                                  alignment=TA_JUSTIFY, spaceBefore=3, spaceAfter=10),
        "cell": ParagraphStyle("cell", fontName="Times-Roman", fontSize=9.5, leading=11.5),
        "cellb": ParagraphStyle("cellb", fontName="Times-Bold", fontSize=9.5, leading=11.5),
        "ref": ParagraphStyle("ref", fontName="Times-Roman", fontSize=10, leading=13,
                              leftIndent=18, firstLineIndent=-18, spaceAfter=3),
    }


def table(rows, col_widths, st, align_right_from=1):
    """Academic-style table: rules above and below the header and at the end."""
    data = [[Paragraph(str(c), st["cellb"]) for c in rows[0]]]
    data += [[Paragraph(str(c), st["cell"]) for c in r] for r in rows[1:]]
    t = Table(data, colWidths=col_widths, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 1.0, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 1.0, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
        ("ALIGN", (align_right_from, 0), (-1, -1), "RIGHT"),
    ]))
    return t


def table_block(caption, tbl, st):
    return KeepTogether([Paragraph(caption, st["caption"]), tbl, Spacer(1, 10)])


def figure_block(path, caption, st, width=6.0 * inch):
    img = Image(str(path))
    img.drawHeight = width * img.imageHeight / img.imageWidth
    img.drawWidth = width
    return KeepTogether([img, Paragraph(caption, st["caption"])])


# ---------------------------------------------------------------- figures

def fig_lot_yield(lots, path):
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    ax.hist(lots["lot_yield"], bins=80, color=MID, edgecolor="none")
    ax.axvline(lots["lot_yield"].median(), color=DARK, lw=1, ls="--", label="Median")
    ax.set_xlabel("Lot yield")
    ax.set_ylabel("Lots")
    ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_laney(spc, path, k=500):
    d = spc.iloc[:k]
    ooc = d["ooc_high_laney"] | d["ooc_low_laney"]
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    ax.step(d["seq"], d["ucl_laney"], where="mid", color="#888888", lw=0.6, label="Laney limits")
    ax.step(d["seq"], d["lcl_laney"], where="mid", color="#888888", lw=0.6)
    ax.axhline(d["center"].iloc[0], color="black", lw=0.8, label="Center line")
    ax.scatter(d.loc[~ooc, "seq"], d.loc[~ooc, "fail_rate"], s=4, color=MID, label="In control")
    ax.scatter(d.loc[ooc, "seq"], d.loc[ooc, "fail_rate"], s=10, color="#b03a2e", label="Out of control")
    ax.set_xlabel("Lot sequence (ordered by lot number)")
    ax.set_ylabel("Lot fail rate")
    ax.legend(frameon=False, fontsize=7.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_confusion(cm, path):
    norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.imshow(norm.to_numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cm)), cm.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(cm)), cm.index)
    for i in range(len(cm)):
        for j in range(len(cm)):
            v = norm.iat[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.5 else "black")
    ax.set_xlabel("Predicted pattern")
    ax.set_ylabel("True pattern")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_pareto_compare(labeled_p, all_p, path):
    lab = labeled_p.set_index("failure_type")["share_dies_lost"].reindex(PATTERNS).fillna(0)
    al = all_p.set_index("pattern")["share_patterned"].reindex(PATTERNS).fillna(0)
    order = al.sort_values(ascending=False).index
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(6.5, 2.9))
    ax.bar(x - 0.2, lab[order] * 100, 0.4, color=LIGHT, label="Labeled wafers only")
    ax.bar(x + 0.2, al[order] * 100, 0.4, color=DARK, label="All wafers (labels + predictions)")
    ax.set_xticks(x, order, rotation=25, ha="right")
    ax.set_ylabel("Share of patterned dies lost (%)")
    ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--report", default="reports/wafer_yield_report.pdf")
    args = parser.parse_args()
    out = Path(args.out)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    figdir = out / "report_figures"
    figdir.mkdir(parents=True, exist_ok=True)

    # ---- load everything the report cites
    s1 = json.loads((out / "summary.json").read_text())
    s2 = json.loads((out / "spc_summary.json").read_text())
    s3 = json.loads((out / "phase3_summary.json").read_text())
    s4 = json.loads((out / "phase4_summary.json").read_text())
    lots = pd.read_csv(out / "lots.csv")
    spc = pd.read_csv(out / "lots_spc.csv")
    pareto_lab = pd.read_csv(out / "pareto_patterns.csv")
    rep = pd.read_csv(out / "labeled_vs_unlabeled.csv").set_index("group")
    mix = pd.read_csv(out / "pattern_mix_flagged_lots.csv")
    clf = pd.read_csv(out / "classifier_report.csv", index_col=0)
    cm = pd.read_csv(out / "confusion_matrix.csv", index_col=0)
    pareto_all = pd.read_csv(out / "pareto_all_wafers_conf80.csv")
    wafers = pd.read_csv(out / "wafers.csv", usecols=["labeled", "wafer_yield", "n_fail"])

    # ---- derived numbers
    total_fails = float(wafers["n_fail"].sum())
    std_flagged_any = (s2["ooc_high_std"] + s2["ooc_low_std"]) / s2["n_lots"]
    laney_vs_chance = s2["ooc_high_laney"] / s2["expected_ooc_high_if_in_control"]
    excess_share = s2["ooc_high_laney_excess_fails"] / total_fails
    split = s3["dies_lost_split"]
    lab_y, unl_y = rep.loc["labeled"], rep.loc["unlabeled"]

    dead = wafers["wafer_yield"] < 0.2
    dead_unl = int((dead & ~wafers["labeled"]).sum())
    dead_lab = int((dead & wafers["labeled"]).sum())
    n_unl, n_lab = int((~wafers["labeled"]).sum()), int(wafers["labeled"].sum())
    dead_ratio = (dead_unl / n_unl) / (dead_lab / n_lab) if dead_lab else float("nan")

    patterned = pareto_all[~pareto_all["pattern"].isin(["none", "uncertain"])].copy()
    patterned["share_patterned"] = patterned["dies_lost"] / patterned["dies_lost"].sum()
    patterned = patterned.sort_values("dies_lost", ascending=False).reset_index(drop=True)
    patterned["cum"] = patterned["share_patterned"].cumsum()
    top1, top2, top3 = patterned.iloc[0], patterned.iloc[1], patterned.iloc[2]
    uncertain = pareto_all.set_index("pattern")["share_all_dies_lost"].get("uncertain", 0.0)

    lab_top = pareto_lab.iloc[0]
    mix_i = mix.set_index("failure_type")
    top_lift = mix[mix["failure_type"] != "none"].sort_values("lift", ascending=False).head(3)
    patt_share_flag = 1 - mix_i.loc["none", "share_flagged"]
    patt_share_other = 1 - mix_i.loc["none", "share_other"]
    none_share_labeled = ((mix_i.loc["none", "n_other"] + mix_i.loc["none", "n_flagged"])
                          / (mix["n_other"].sum() + mix["n_flagged"].sum()))

    weak = clf.drop(index="none").sort_values("precision").head(3)
    strong = clf.loc[["Edge-Ring"]].iloc[0]

    # ---- figures
    fig_lot_yield(lots, figdir / "fig1_lot_yield.png")
    fig_laney(spc, figdir / "fig2_laney.png")
    fig_confusion(cm, figdir / "fig3_confusion.png")
    fig_pareto_compare(pareto_lab, patterned, figdir / "fig4_pareto_compare.png")

    st = styles()
    S = []
    P = lambda text, style="body": S.append(Paragraph(text, st[style]))  # noqa: E731

    # ---- title block
    P(TITLE, "title")
    P(f"{AUTHOR}<br/>Independent Project, {DATE}<br/>{REPO}", "author")
    P("Abstract", "abs_head")
    P(
        f"Wafer probe maps record which dies on each wafer pass or fail electrical test, and are a "
        f"primary input to yield reviews in semiconductor assembly and test. This study analyzes "
        f"{n(s1['n_wafers'])} wafer maps across {n(s1['n_lots'])} lots from the public WM-811K "
        f"dataset. Die-weighted yield is {pct(s1['overall_yield'])}. Because lot-to-lot variation is "
        f"{s2['sigma_z']:.0f} times larger than binomial sampling noise, a standard p-chart flags "
        f"{pct(std_flagged_any, 0)} of lots and is not usable, while a Laney p&#8242; chart flags "
        f"{n(s2['ooc_high_laney'])} high-fail lots ({pct(s2['ooc_high_laney_share'])}) that hold "
        f"{pct(s2['ooc_high_laney_share_of_dies_lost'])} of all dies lost. Among labeled wafers, "
        f"{lab_top.failure_type} is the largest patterned loss ({pct(lab_top.share_dies_lost)}). "
        f"However, labeled wafers hold only {pct(split['labeled_patterned']['share'])} of all dies "
        f"lost and differ in yield distribution from unlabeled wafers (Kolmogorov-Smirnov statistic "
        f"{s3['ks_labeled_vs_unlabeled_yield']:.2f}). A gradient-boosted classifier, evaluated on "
        f"held-out lots at {pct(s4['macro_recall_test'])} macro recall, labels the remaining "
        f"{n(s4['n_unlabeled'])} wafers. The resulting Pareto ranks {top1.pattern} "
        f"({pct(top1.share_patterned)}) and {top2.pattern} ({pct(top2.share_patterned)}) above "
        f"{top3.pattern} ({pct(top3.share_patterned)}). A model-free check confirms that near-dead "
        f"wafers are {dead_ratio:.1f} times more common among unlabeled wafers. In this dataset, "
        f"lot excursion detection and containment is a larger lever than chronic edge loss.",
        "abstract",
    )

    # ---- 1 introduction
    P("1. Introduction", "h1")
    P(
        "Yield reviews in wafer probe and packaging programs answer three practical questions. "
        "Which lots are abnormal and need investigation, what kinds of defects account for the "
        "most lost dies, and how far can the available labels be trusted. This study builds a "
        "reproducible pipeline that answers each question on a large public dataset, and tracks "
        "how the answer to the second question changes once the limits of the labels are taken "
        "into account."
    )
    P("The analysis proceeds in four stages, each building on the last.")
    for b in [
        "Measure wafer and lot yield with consistent, die-weighted definitions.",
        "Monitor lot fail rate with a control chart suited to large lots, and rank flagged lots by excess fails.",
        "Rank defect patterns by dies lost, test whether labeled wafers represent the full dataset, "
        "and relate patterns to flagged lots.",
        "Train a defect pattern classifier, label every unlabeled wafer, and recompute the Pareto over all yield loss.",
    ]:
        S.append(Paragraph(b, st["bullet"], bulletText="•"))

    # ---- 2 data
    P("2. Data", "h1")
    P(
        f"WM-811K [1] contains {n(s1['n_wafers'])} wafer maps from real fabrication. Each map is a "
        f"grid in which 0 marks a position with no die, 1 a passing die, and 2 a failing die. Maps "
        f"come in {n(s1['distinct_map_shapes'])} distinct grid shapes, reflecting different products. "
        f"{n(s1['n_labeled'])} wafers ({pct(s1['share_labeled'])}) carry one of nine labels, eight "
        f"defect patterns plus <i>none</i>. The dataset includes lot identifiers but no timestamps "
        f"or product identifiers. Table 1 summarizes its scale."
    )
    S.append(table_block("Table 1. Dataset summary.", table([
        ["Quantity", "Value"],
        ["Wafers", n(s1["n_wafers"])],
        ["Lots", n(s1["n_lots"])],
        ["Median wafers per lot", n(s1["wafers_per_lot_median"])],
        ["Tested dies", n(s1["n_die_total"])],
        ["Labeled wafers", f"{n(s1['n_labeled'])} ({pct(s1['share_labeled'])})"],
        ["Distinct map shapes", n(s1["distinct_map_shapes"])],
    ], [2.6 * inch, 1.8 * inch], st), st))

    # ---- 3 methods
    P("3. Methods", "h1")
    P("3.1 Yield definitions", "h2")
    P(
        "Wafer yield is passing dies divided by tested dies on one wafer. Lot yield is total "
        "passing dies divided by total tested dies across the lot, so larger wafers carry more "
        "weight. Overall yield is computed the same way over the full dataset. Wafers with no "
        "tested dies are excluded from yield and counted separately."
    )
    P("3.2 Statistical process control", "h2")
    P(
        "Each lot <i>i</i> has fail rate <i>p<sub>i</sub></i> over <i>n<sub>i</sub></i> dies, and the "
        "pooled fail rate is <i>p</i><sub>avg</sub>. A standard p-chart sets limits at"
    )
    P("<i>p</i><sub>avg</sub> &#177; 3<i>&#963;<sub>i</sub></i>,&#160;&#160;&#160;"
      "<i>&#963;<sub>i</sub></i> = &#8730;(<i>p</i><sub>avg</sub>(1 &#8722; <i>p</i><sub>avg</sub>) / <i>n<sub>i</sub></i>)",
      "eq")
    P(
        "which assumes all variation between lots is binomial sampling noise. With tens of "
        "thousands of dies per lot, those limits are very narrow. The Laney p&#8242; chart [2] "
        "converts each lot to <i>z<sub>i</sub></i> = (<i>p<sub>i</sub></i> &#8722; <i>p</i><sub>avg</sub>) / "
        "<i>&#963;<sub>i</sub></i>, estimates their spread from the average moving range of "
        "consecutive z-scores, and widens the limits by that factor."
    )
    P("<i>&#963;<sub>z</sub></i> = mean|<i>z<sub>i</sub></i> &#8722; <i>z</i><sub><i>i</i>&#8722;1</sub>| / 1.128,"
      "&#160;&#160;&#160;limits <i>p</i><sub>avg</sub> &#177; 3<i>&#963;<sub>i</sub>&#963;<sub>z</sub></i>", "eq")
    P(
        "A value of <i>&#963;<sub>z</sub></i> near 1 indicates purely binomial variation. Because the "
        "moving range depends on lot order and the data has no timestamps, lots are ordered by the "
        "number in their lot name. For each flagged lot, excess fails are the observed fails minus "
        "<i>p</i><sub>avg</sub><i>n<sub>i</sub></i>, the fails expected at the average rate."
    )
    P("3.3 Pareto, representativeness, and lift", "h2")
    P(
        "Patterns are ranked by total dies lost rather than by wafer count, since a rare pattern "
        "can cost more dies than a common one. To test whether labeled wafers represent the "
        "dataset, their wafer yield distribution is compared with that of unlabeled wafers using "
        "the two-sample Kolmogorov-Smirnov (KS) statistic, the largest gap between the two "
        "empirical distribution functions. To relate patterns to the control chart, each pattern's "
        "share among labeled wafers in flagged lots is divided by its share in all other lots. This "
        "ratio, called lift, exceeds 1 when a pattern is over-represented in flagged lots."
    )
    P("3.4 Defect pattern classifier", "h2")
    P(
        "Each wafer map is converted to 85 engineered features so that maps of any shape enter "
        "one model. The features are the overall fail rate, fail rates in an 8 by 8 grid, in five "
        "concentric rings, and in eight angular sectors, the ratio of edge to overall fail rate, "
        "the mean and spread of failing dies' radial position, and statistics on connected "
        "clusters of failing dies, including how elongated the largest cluster is. A "
        "histogram-based gradient-boosted tree classifier is trained with balanced class weights, "
        f"because <i>none</i> makes up {pct(none_share_labeled, 0)} of "
        "labeled wafers. Train and test sets are split by lot, 80% and 20%, so no lot contributes "
        "wafers to both. Results are reported as per-class precision and recall and as macro "
        "recall, the unweighted mean of per-class recall."
    )
    P(
        "A final model trained on all labeled wafers then predicts a pattern and a confidence, the "
        "top class probability, for each unlabeled wafer. The all-wafer Pareto uses true labels "
        "where they exist and predictions elsewhere, and counts predictions below 0.8 confidence "
        "as <i>uncertain</i>."
    )

    # ---- 4 results
    P("4. Results", "h1")
    P("4.1 Yield overview", "h2")
    P(
        f"Overall die-weighted yield is {pct(s1['overall_yield'])}. Median wafer yield is "
        f"{pct(s1['median_wafer_yield'])} and median lot yield is {pct(s1['median_lot_yield'])}. "
        f"Overall yield sits above both medians, which is consistent with the positive relationship "
        f"between lot size and lot yield (Spearman rank correlation {s2['spearman_dies_vs_yield']:.2f}). "
        f"Larger lots carry more weight in the die-weighted figure and also tend to yield better. "
        f"Figure 1 shows the lot yield distribution."
    )
    S.append(figure_block(figdir / "fig1_lot_yield.png",
                          "Figure 1. Distribution of lot yield across all lots. The dashed line marks the median.",
                          st, width=5.8 * inch))

    P("4.2 Lot monitoring", "h2")
    P(
        f"The estimated <i>&#963;<sub>z</sub></i> is {s2['sigma_z']:.1f}, so lot-to-lot variation is "
        f"about {s2['sigma_z']:.0f} times what binomial noise alone would produce. At the median lot "
        f"size of {n(s2['median_dies_per_lot'])} dies, the standard chart's limits extend only "
        f"&#177;{100 * s2['median_std_limit_halfwidth']:.2f} percentage points from the center line, "
        f"and it flags {pct(std_flagged_any)} of lots on one side or the other. The Laney chart "
        f"flags {n(s2['ooc_high_laney'])} lots on the high side, about {laney_vs_chance:.0f} times "
        f"the {s2['expected_ooc_high_if_in_control']:.0f} expected by chance in a stable process "
        f"(Table 2). These lots hold {pct(s2['ooc_high_laney_share_of_dies_lost'])} of all dies lost, "
        f"and their excess fails total {millions(s2['ooc_high_laney_excess_fails'])} dies, or "
        f"{pct(excess_share)} of all yield loss."
    )
    S.append(table_block("Table 2. Standard p-chart and Laney p&#8242; chart on lot fail rate.", table([
        ["", "Standard p-chart", "Laney p&#8242; chart"],
        ["Lots flagged high", f"{n(s2['ooc_high_std'])} ({pct(s2['ooc_high_std_share'])})",
         f"{n(s2['ooc_high_laney'])} ({pct(s2['ooc_high_laney_share'])})"],
        ["Lots flagged low", n(s2["ooc_low_std"]), n(s2["ooc_low_laney"])],
        ["Median limit half-width", f"{100 * s2['median_std_limit_halfwidth']:.2f} pp",
         f"{100 * s2['median_laney_limit_halfwidth']:.2f} pp"],
        ["Share of dies lost in high-flagged lots", pct(s2["ooc_high_std_share_of_dies_lost"]),
         pct(s2["ooc_high_laney_share_of_dies_lost"])],
    ], [2.6 * inch, 1.6 * inch, 1.6 * inch], st), st))
    S.append(figure_block(figdir / "fig2_laney.png",
                          "Figure 2. Laney p&#8242; chart for the first 500 lots by lot number. Limits differ "
                          "by lot because each lot has a different die count.", st))

    P("4.3 Defect Pareto on labeled wafers", "h2")
    lab_rows = [["Pattern", "Wafers", "Dies lost", "Dies lost per wafer", "Share", "Cumulative"]]
    for _, r in pareto_lab.iterrows():
        lab_rows.append([r.failure_type, n(r.n_wafers), n(r.dies_lost), f"{r.dies_lost_per_wafer:.0f}",
                         pct(r.share_dies_lost), pct(r.cum_share_dies_lost)])
    P(
        f"Among labeled wafers with a pattern, {lab_top.failure_type} accounts for "
        f"{pct(lab_top.share_dies_lost)} of dies lost, and "
        f"{s3['patterns_to_reach_80pct_of_patterned_loss']} patterns reach 80% (Table 3). Ranking by "
        f"dies lost and ranking by count disagree for low-frequency, high-severity patterns. Random "
        f"wafers lose {pareto_lab.set_index('failure_type').loc['Random','dies_lost_per_wafer']:.0f} dies "
        f"each and Scratch wafers "
        f"{pareto_lab.set_index('failure_type').loc['Scratch','dies_lost_per_wafer']:.0f}."
    )
    S.append(table_block("Table 3. Patterned dies lost among labeled wafers, ranked by dies lost.",
                         table(lab_rows, [1.05 * inch, 0.8 * inch, 0.95 * inch, 1.3 * inch, 0.75 * inch, 0.95 * inch], st), st))

    P("4.4 Coverage and representativeness of labels", "h2")
    P(
        f"Labeled wafers with a pattern hold only {pct(split['labeled_patterned']['share'])} of all "
        f"dies lost. Labeled <i>none</i> wafers hold {pct(split['labeled_none']['share'])} and "
        f"unlabeled wafers {pct(split['unlabeled']['share'])}. The two groups also differ in yield "
        f"(Table 4). The KS statistic is {s3['ks_labeled_vs_unlabeled_yield']:.2f}. Unlabeled wafers "
        f"are more spread out, with more near-perfect wafers and more very poor ones, so the labeled "
        f"Pareto in Table 3 describes the labeled set and should not be read as a dataset-wide result."
    )
    S.append(table_block("Table 4. Wafer yield of labeled and unlabeled wafers.", table([
        ["Group", "Wafers", "10th pct.", "Median", "90th pct.", "Share below 80%"],
        ["Labeled", n(lab_y.n_wafers), pct(lab_y.p10), pct(lab_y["median"]), pct(lab_y.p90), pct(lab_y.share_below_80pct)],
        ["Unlabeled", n(unl_y.n_wafers), pct(unl_y.p10), pct(unl_y["median"]), pct(unl_y.p90), pct(unl_y.share_below_80pct)],
    ], [1.0 * inch, 0.95 * inch, 0.9 * inch, 0.9 * inch, 0.9 * inch, 1.2 * inch], st), st))

    P("4.5 Patterns in flagged lots", "h2")
    lift_txt = ", ".join(f"{r.failure_type} ({r.lift:.1f} times)" for r in top_lift.itertuples())
    P(
        f"In flagged lots, {pct(patt_share_flag, 0)} of labeled wafers carry a defect pattern, "
        f"compared with {pct(patt_share_other, 0)} in other lots. The most over-represented "
        f"patterns are {lift_txt} (Table 5). Scratch is under-represented "
        f"({mix_i.loc['Scratch','lift']:.2f}). Part of this effect is mechanical, since a few "
        f"high-severity wafers can push a lot above its limit by themselves. Lift therefore reflects "
        f"severity as well as association. Only {pct(s3['share_of_flagged_lot_wafers_labeled'])} of "
        f"wafers in flagged lots are labeled, and estimates for rare patterns rest on few wafers."
    )
    mix_rows = [["Pattern", "Wafers in flagged lots", "Share, flagged", "Share, other", "Lift"]]
    for r in mix.itertuples():
        mix_rows.append([r.failure_type, n(r.n_flagged), pct(r.share_flagged), pct(r.share_other), f"{r.lift:.2f}"])
    S.append(table_block("Table 5. Pattern mix among labeled wafers in Laney-flagged lots and in all other lots.",
                         table(mix_rows, [1.1 * inch, 1.5 * inch, 1.15 * inch, 1.15 * inch, 0.8 * inch], st), st))

    P("4.6 Classifier performance", "h2")
    weak_txt = ", ".join(f"{name} ({pct(r.precision, 0)})" for name, r in weak.iterrows())
    P(
        f"On {n(s4['n_test'])} wafers from held-out lots, macro recall is {pct(s4['macro_recall_test'])} "
        f"and accuracy is {pct(s4['accuracy_test'])} (Table 6, Figure 3). Edge-Ring is recognized with "
        f"{pct(strong.recall, 0)} recall at {pct(strong.precision, 0)} precision. Precision is lowest "
        f"for {weak_txt}. These patterns are localized, and because <i>none</i> wafers are so "
        f"numerous, even a small share of them predicted as a localized pattern outnumbers the "
        f"true cases. Balanced class weights make this trade, accepting false alarms in exchange "
        f"for recall on rare patterns."
    )
    clf_rows = [["Pattern", "Precision", "Recall", "F1", "Test wafers"]]
    for name, r in clf.iterrows():
        clf_rows.append([name, f"{r.precision:.2f}", f"{r.recall:.2f}", f"{r['f1-score']:.2f}", n(r.support)])
    S.append(table_block("Table 6. Classifier results on held-out lots.",
                         table(clf_rows, [1.2 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch, 1.1 * inch], st), st))
    S.append(figure_block(figdir / "fig3_confusion.png",
                          "Figure 3. Confusion matrix on held-out lots, normalized by row so the diagonal "
                          "shows recall for each true pattern.", st, width=4.3 * inch))

    P("4.7 Pareto over all wafers", "h2")
    P(
        f"Combining true labels with confident predictions changes the ranking (Table 7, Figure 4). "
        f"{top1.pattern} rises to the largest patterned loss at {pct(top1.share_patterned)}, followed "
        f"by {top2.pattern} at {pct(top2.share_patterned)}. Together they account for "
        f"{pct(top2.cum)}. {top3.pattern} falls to third at {pct(top3.share_patterned)}. Predictions "
        f"below 0.8 confidence account for {pct(uncertain)} of all dies lost and are excluded from "
        f"the shares. The top three patterns are also the ones the classifier predicts most "
        f"precisely, so the reordering does not depend on the weak classes."
    )
    P(
        f"Because the {top1.pattern} result depends on the classifier, it was checked against a rule "
        f"that does not use the model. Wafers below 20% yield number {n(dead_unl)} among unlabeled "
        f"wafers ({pct(dead_unl / n_unl, 2)}) and {n(dead_lab)} among labeled wafers "
        f"({pct(dead_lab / n_lab, 3)}), a ratio of {dead_ratio:.1f}. Near-dead wafers were largely "
        f"left unlabeled, which explains why the labeled Pareto underweights them."
    )
    all_rows = [["Pattern", "Wafers", "Of which predicted", "Dies lost", "Share", "Cumulative"]]
    for r in patterned.itertuples():
        all_rows.append([r.pattern, n(r.n_wafers), n(r.n_predicted), n(r.dies_lost),
                         pct(r.share_patterned), pct(r.cum)])
    S.append(table_block("Table 7. Patterned dies lost over all wafers, using true labels where available and "
                         "predictions with confidence of at least 0.8 elsewhere.",
                         table(all_rows, [1.05 * inch, 0.9 * inch, 1.25 * inch, 1.0 * inch, 0.75 * inch, 0.9 * inch], st), st))
    S.append(figure_block(figdir / "fig4_pareto_compare.png",
                          "Figure 4. Share of patterned dies lost by pattern, among labeled wafers only and over "
                          "all wafers.", st))

    # ---- 5 discussion
    P("5. Discussion", "h1")
    P(
        f"The two views of the data point to different priorities. On labeled wafers alone, "
        f"{lab_top.failure_type} is the dominant patterned loss, which would argue for a sustained "
        f"improvement effort on one chronic defect. Over all wafers, {top1.pattern} and "
        f"{top2.pattern} lead. These are the same patterns most over-represented in flagged lots, "
        f"which marks them as excursion patterns. They arrive in bursts, concentrate in specific "
        f"lots, and cost hundreds of dies per affected wafer. The labeled-only analysis would have "
        f"pointed effort at the wrong problem, and the check on label coverage is what revealed it."
    )
    P("For a program team, the results suggest three workstreams with different cadences and owners.")
    for b in [
        f"<b>Excursion detection.</b> Review a Laney p&#8242; chart of lot fail rate on a regular "
        f"cadence, with flagged lots ranked by excess fails. The flagged lots here are "
        f"{pct(s2['ooc_high_laney_share'])} of lots but hold {pct(s2['ooc_high_laney_share_of_dies_lost'])} "
        f"of dies lost.",
        f"<b>Containment for severe patterns.</b> Treat {top1.pattern} and {top2.pattern} wafers as "
        f"triggers for prompt investigation of their lot, since each occurrence costs far more than "
        f"a typical defect.",
        f"<b>Chronic improvement.</b> Keep {lab_top.failure_type} as a longer-term improvement target. "
        f"It remains a top-three loss in both views and is recognized reliably by the classifier, "
        f"which makes progress measurable.",
    ]:
        S.append(Paragraph(b, st["bullet"], bulletText="•"))
    P(
        "This study identifies where loss concentrates, not why. Linking each pattern to a process "
        "step requires engineering data such as tool, chamber, and probe card history, which "
        "WM-811K does not include."
    )

    # ---- 6 limitations
    P("6. Limitations", "h1")
    for b in [
        "Lot order is inferred from lot numbers. The moving-range estimate in the Laney chart assumes this order reflects production sequence, which cannot be confirmed.",
        f"Lots of different products are charted together. The positive relationship between lot size and yield (Spearman {s2['spearman_dies_vs_yield']:.2f}) suggests product mix contributes to the large <i>&#963;<sub>z</sub></i>. Charting by die count band as a stand-in for product would separate the two.",
        "The classifier is trained on labeled wafers, which differ from unlabeled ones. Held-out performance may overstate accuracy on unlabeled wafers, so the all-wafer Pareto is an estimate.",
        "Dies lost counts every failing die on a wafer, including background failures not caused by the labeled pattern, so pattern shares are upper bounds on the loss each pattern causes.",
    ]:
        S.append(Paragraph(b, st["bullet"], bulletText="•"))

    # ---- 7 conclusion
    P("7. Conclusion", "h1")
    P(
        f"A reproducible pipeline over {n(s1['n_wafers'])} wafer maps shows that a standard p-chart "
        f"fails at this lot size, that a Laney p&#8242; chart isolates {n(s2['ooc_high_laney'])} lots "
        f"holding {pct(s2['ooc_high_laney_share_of_dies_lost'])} of loss, and that the labeled-wafer "
        f"Pareto misses the patterns driving most patterned loss. Extending labels with a classifier "
        f"validated on held-out lots, and confirming the key result with a model-free check, moves "
        f"the priority from chronic {lab_top.failure_type} loss to excursion patterns led by "
        f"{top1.pattern} and {top2.pattern}. Code and outputs are available at {REPO}."
    )

    # ---- references
    P("References", "h1")
    for r in [
        "[1] M.-J. Wu, J.-S. R. Jang, and J.-L. Chen, “Wafer map failure pattern recognition and "
        "similarity ranking for large-scale data sets,” <i>IEEE Transactions on Semiconductor "
        "Manufacturing</i>, vol. 28, no. 1, pp. 1–12, 2015.",
        "[2] D. B. Laney, “Improved control charts for attributes,” <i>Quality Engineering</i>, "
        "vol. 14, no. 4, pp. 531–537, 2002.",
    ]:
        P(r, "ref")

    # ---- page footer
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Times-Roman", 9)
        canvas.drawCentredString(letter[0] / 2, 0.55 * inch, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(str(report_path), pagesize=letter, leftMargin=inch, rightMargin=inch,
                            topMargin=inch, bottomMargin=inch, title=TITLE, author=AUTHOR)
    doc.build(S, onFirstPage=footer, onLaterPages=footer)
    print(f"Wrote {report_path.resolve()}")


if __name__ == "__main__":
    main()