"""
generate_moe_charts.py
----------------------
Generates three sets of charts for MOE officer review:

  1b  — Per-degree readiness scatter (skill coverage vs job market coverage)
  2b  — Module score distribution by primary role family (box plots)
  2d  — Primary major readiness heatmap, one image per faculty
        + Common curriculum score distribution violin

All images are written to outputs/moe_charts/.
Run from the project root:
    python generate_moe_charts.py
"""

import ast
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
OUT_DIR = OUTPUTS / "moe_charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":        "sans-serif",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         150,
})

FACULTY_COLORS = {
    "SOC": "#2196F3", "BIZ": "#FF9800", "CDE": "#9C27B0",
    "CHS": "#4CAF50", "MED": "#F44336", "LAW": "#607D8B",
    "DEN": "#00BCD4", "YST": "#E91E63", "CDE/SOC": "#673AB7",
}

COMMON_TYPES = [
    "Communities and Engagement", "Cultures and Connections", "Singapore Studies",
    "Interdisciplinary Courses", "Digital Literacy", "Scientific Inquiry 2",
    "Data Literacy", "Critique and Expression",
]

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
jobs          = pd.read_csv(OUTPUTS / "jobs_clean.csv")
modules       = pd.read_csv(OUTPUTS / "modules_clean.csv")
module_scores = pd.read_csv(OUTPUTS / "module_role_scores.csv")
degree_map    = pd.read_csv(OUTPUTS / "degree_module_map.csv")
evidence      = pd.read_csv(OUTPUTS / "module_job_evidence.csv")


def parse_skills(value):
    if not isinstance(value, str) or value.strip() in ("", "nan", "[]"):
        return []
    try:
        return [str(s).strip().lower() for s in ast.literal_eval(value) if str(s).strip()]
    except Exception:
        return [s.strip().strip("'").lower() for s in value.strip("[]").split(",") if s.strip()]


jobs["_skills"]    = jobs["technical_skills"].apply(parse_skills)
modules["_skills"] = modules["technical_skills"].apply(parse_skills)
demand_all = Counter(sk for skills in jobs["_skills"] for sk in skills)

print(f"  Jobs: {len(jobs):,}  Modules: {len(modules):,}  Evidence pairs: {len(evidence):,}")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 1b — Per-degree readiness scatter
# ══════════════════════════════════════════════════════════════════════════════
def chart_1b():
    print("\n[1b] Per-degree readiness scatter...")

    TOP_DEMANDED_SKILLS = {sk for sk, _ in demand_all.most_common(50)}
    MATCH_THRESHOLD = 0.3
    total_jobs = len(jobs)

    req = degree_map[
        (degree_map["module_found"] == True) &
        (degree_map["is_unrestricted_elective"] == False)
    ]

    rows = []
    for degree_id, grp in req.groupby("degree_id"):
        info = grp.iloc[0]
        mods = set(grp["module_code"].tolist())
        jobs_supported = evidence[
            (evidence["module_code"].isin(mods)) & (evidence["rrf_score"] > MATCH_THRESHOLD)
        ]["job_id"].nunique()
        taught  = set(sk for s in grp["technical_skills"].dropna() for sk in parse_skills(s))
        covered = TOP_DEMANDED_SKILLS & taught
        rows.append({
            "degree_id":          degree_id,
            "major":              info["major"],
            "faculty_code":       info["faculty_code"],
            "job_coverage_pct":   jobs_supported / total_jobs * 100,
            "skill_coverage_pct": len(covered) / len(TOP_DEMANDED_SKILLS) * 100,
        })

    scatter_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11, 8))

    for fc, grp in scatter_df.groupby("faculty_code"):
        ax.scatter(
            grp["job_coverage_pct"], grp["skill_coverage_pct"],
            color=FACULTY_COLORS.get(fc, "#9E9E9E"),
            label=fc, s=80, alpha=0.85, edgecolors="white", linewidths=0.5,
        )

    med_x = scatter_df["job_coverage_pct"].median()
    med_y = scatter_df["skill_coverage_pct"].median()
    ax.axvline(med_x, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(med_y, color="gray", linestyle=":",  linewidth=0.8, alpha=0.5)

    ax.text(med_x + 0.5, med_y + 1.5,
            "High coverage both axes", fontsize=7, color="gray", style="italic")
    ax.text(scatter_df["job_coverage_pct"].min() + 0.5, med_y + 1.5,
            "Strong skills, narrow job reach", fontsize=7, color="gray", style="italic")
    ax.text(med_x + 0.5, scatter_df["skill_coverage_pct"].min() + 0.5,
            "Broad job reach, weaker skills", fontsize=7, color="gray", style="italic")

    ax.set_xlabel("Job market coverage (% of 1,994 entry-level jobs supported, RRF > 0.3)", fontsize=10)
    ax.set_ylabel("Skill coverage (% of top-50 demanded skills in required curriculum)", fontsize=10)
    ax.set_title("Chart 1b — Per-degree readiness: skill coverage vs job market coverage", fontsize=12)
    ax.legend(title="Faculty", fontsize=8, loc="lower right", ncol=2)

    plt.tight_layout()
    out = OUT_DIR / "degree_readiness_scatter.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 2b — Module score distribution by primary role family (box plots)
# ══════════════════════════════════════════════════════════════════════════════
def chart_2b():
    print("\n[2b] Module score distribution box plots...")

    primary_scores = (
        module_scores
        .sort_values(["module_code", "role_score"], ascending=[True, False])
        .groupby("module_code")
        .first()
        .reset_index()
    )

    role_order = (
        primary_scores.groupby("role_family")["role_score"]
        .median()
        .sort_values()
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(10, 9))

    data_by_role = [
        primary_scores[primary_scores["role_family"] == r]["role_score"].values
        for r in role_order
    ]

    bp = ax.boxplot(
        data_by_role,
        vert=False,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        flierprops=dict(marker="o", markersize=2.5, alpha=0.4, markerfacecolor="#9E9E9E"),
        whiskerprops=dict(linewidth=0.8),
        boxprops=dict(linewidth=0.8),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#90CAF9")
        patch.set_alpha(0.7)

    # Annotate module count per role
    for i, (role, data) in enumerate(zip(role_order, data_by_role), start=1):
        ax.text(
            ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 0.75,
            i, f"n={len(data)}", va="center", fontsize=6.5, color="#555",
        )

    ax.set_yticks(range(1, len(role_order) + 1))
    ax.set_yticklabels(role_order, fontsize=8)
    ax.set_xlabel("Role alignment score (primary role assignment)", fontsize=10)
    ax.set_title(
        "Chart 2b — Module score distribution by primary role family\n"
        "(each module counted once, at its highest-scoring role)",
        fontsize=11,
    )

    plt.tight_layout()
    out = OUT_DIR / "module_score_distribution_by_role.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 2d — Primary major heatmap (one image per faculty) + common violin
# ══════════════════════════════════════════════════════════════════════════════
def chart_2d():
    print("\n[2d] Primary major heatmaps per faculty + common curriculum violin...")

    # Build module sets
    common_codes = set(
        degree_map[
            degree_map["module_type"].isin(COMMON_TYPES) &
            (degree_map["module_found"] == True)
        ]["module_code"].unique()
    )
    core_codes_df = degree_map[
        (degree_map["is_unrestricted_elective"] == False) &
        (~degree_map["module_type"].isin(COMMON_TYPES)) &
        (degree_map["module_found"] == True)
    ][["degree_id", "major", "faculty_code", "module_code"]].drop_duplicates()

    core_ms   = module_scores[module_scores["module_code"].isin(core_codes_df["module_code"].unique())]
    common_ms = module_scores[module_scores["module_code"].isin(common_codes)]

    # Degree × role mean scores
    core_merged = core_ms.merge(
        core_codes_df[["module_code", "degree_id", "major", "faculty_code"]].drop_duplicates("module_code"),
        on="module_code", how="inner",
    )
    pivot_full = (
        core_merged
        .groupby(["degree_id", "major", "faculty_code", "role_family"])["role_score"]
        .mean()
        .reset_index()
        .pivot_table(
            index=["faculty_code", "degree_id", "major"],
            columns="role_family",
            values="role_score",
            fill_value=0,
        )
    )

    # Column order: ascending mean across all degrees (consistent across all faculty plots)
    col_order = pivot_full.mean().sort_values().index.tolist()
    pivot_full = pivot_full[col_order]
    vmax = pivot_full.values.max()

    faculties = sorted(pivot_full.index.get_level_values("faculty_code").unique())

    for fc in faculties:
        fc_data = pivot_full.xs(fc, level="faculty_code")
        if fc_data.empty:
            continue

        # Sort rows within faculty by dominant role (argmax)
        fc_data = fc_data.loc[
            fc_data.idxmax(axis=1).map(col_order.index).sort_values().index
        ]
        row_labels = fc_data.index.get_level_values("major")
        n_rows = len(fc_data)

        fig_h = max(4, n_rows * 0.55 + 2.5)
        fig, ax = plt.subplots(figsize=(16, fig_h))

        im = ax.imshow(
            fc_data.values,
            aspect="auto",
            cmap="YlOrRd",
            vmin=0,
            vmax=vmax,
        )

        ax.set_xticks(range(len(col_order)))
        ax.set_xticklabels(col_order, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels, fontsize=9)

        plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01,
                     label="Mean alignment score (primary major modules)")
        ax.set_title(
            f"Chart 2d — Primary major readiness by role family: {fc}\n"
            f"({n_rows} degree{'s' if n_rows > 1 else ''} — "
            f"mean score of core non-GE required modules)",
            fontsize=11, pad=12,
        )

        plt.tight_layout()
        safe_fc = fc.replace("/", "-")
        out = OUT_DIR / f"primary_major_heatmap_{safe_fc}.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out.name}")

    # ── Common curriculum violin ──────────────────────────────────────────────
    role_order_v = col_order   # same axis as heatmap for coherence

    fig2, ax2 = plt.subplots(figsize=(10, 10))
    vp_data = [common_ms[common_ms["role_family"] == r]["role_score"].values for r in role_order_v]
    valid   = [(r, d) for r, d in zip(role_order_v, vp_data) if len(d) >= 5]
    roles_v, data_v = zip(*valid)

    parts = ax2.violinplot(
        data_v,
        positions=range(len(roles_v)),
        vert=False,
        showmedians=True,
        showextrema=False,
        widths=0.75,
    )
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.5)
    for body in parts["bodies"]:
        body.set_facecolor("#FF9800")
        body.set_alpha(0.65)
        body.set_edgecolor("white")
        body.set_linewidth(0.5)

    medians = [np.median(d) for d in data_v]
    ax2.scatter(medians, range(len(roles_v)), color="#E65100", s=20, zorder=3)

    ax2.set_yticks(range(len(roles_v)))
    ax2.set_yticklabels(roles_v, fontsize=8)
    ax2.set_xlabel("Role alignment score", fontsize=10)
    ax2.set_title(
        f"Chart 2d — Common curriculum ({len(common_codes)} GE modules): "
        f"score distribution by role family",
        fontsize=11,
    )
    ax2.set_xlim(0, None)

    plt.tight_layout()
    out = OUT_DIR / "common_curriculum_violin.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ── Run all ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    chart_1b()
    chart_2b()
    chart_2d()

    generated = sorted(OUT_DIR.glob("*.png"))
    print(f"\nDone. {len(generated)} images written to {OUT_DIR.relative_to(ROOT)}/")
    for f in generated:
        print(f"  {f.name}")
