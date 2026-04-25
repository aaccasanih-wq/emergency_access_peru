"""
visualization.py
Static charts (matplotlib/seaborn) and maps (GeoPandas + Folium) for the
emergency healthcare access analysis of Peru.

Analytical mapping — figure → question answered
────────────────────────────────────────────────────────────────────────────
Q1  Territorial Availability (ipress_per_10k, atenciones_per_10k)
  fig1_q1_distributions.png  Dual histogram+KDE: distributions of both Q1
                              metrics (with WHO threshold and zero-activity note).
  fig2_q1_rankings.png       Two vertical bar ranking charts: top 10 districts
                              by ipress_per_10k (left) and top 10 districts
                              by atenciones_per_10k (right).
  map1_q1_dual.png           Two-panel choropleth with dark district borders:
                              ipress_per_10k (left) and atenciones_per_10k (right).

Q2  Settlement Access (dist_mediana_km)
  fig_q2_distribution.png    Histogram+KDE of dist_mediana_km with 10 km line.
  fig_q2_rankings.png        Vertical bar ranking: top/bottom 10 districts
                              by dist_mediana_km.
  map2_q2_dist_mediana.png   Choropleth of dist_mediana_km.
  mapa_q2_spatial_access.html  Interactive choropleth of dist_mediana_km.

Q3  District Comparison (indice_baseline)
  fig4_q3_ranking.png        Top-15 / Bottom-15 horizontal bar chart with
                              per-component score markers.
  map3_q3_baseline_quintile.png  Categorical choropleth of the five quintile classes.
  mapa_q3_indice_baseline.html   Interactive choropleth with all component values.

Q4  Methodological Sensitivity (indice_baseline vs indice_alternativo)
  fig5_q4_sensitivity.png    Two-panel: index agreement scatter + 5×5 quintile
                              transition heatmap.
  map4_q4_comparison.png     Two-panel: baseline quintile map (left) vs
                              quintile-shift map (right).
  mapa_q4_dual_comparison.html   Two-layer choropleth with layer control:
                              baseline quintile (default) vs quintile shift.

Outputs → output/figures/
────────────────────────────────────────────────────────────────────────────
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import folium
from folium.features import GeoJsonTooltip

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)

PALETTE_QUINTILE = {
    "Very low":  "#d73027",
    "Low":       "#fc8d59",
    "Medium":    "#fee090",
    "High":      "#91bfdb",
    "Very high": "#4575b4",
}
QUINTILE_ORDER = ["Very low", "Low", "Medium", "High", "Very high"]

COMP_COLORS = {
    "Facility availability": "#2166ac",
    "Emergency activity":    "#e08214",
    "Spatial access":        "#1a9641",
}
WHO_THRESHOLD = 2.0
FIG_DPI    = 150
OUTPUT_DIR = "output/figures"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str) -> str:
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")
    return path


def _vertical_bar_panel(ax, data, col, label_col, color, title, ylabel=""):
    """Draw one panel of a vertical-bar ranking chart (data already sorted)."""
    x = range(len(data))
    bars = ax.bar(x, data[col], color=color,
                  edgecolor="white", linewidth=0.5, width=0.65)
    ymax = data[col].max()
    ylim_top = max(ymax * 1.22, 0.5)
    ax.set_ylim(0, ylim_top)

    # Value labels above each bar
    for bar, val in zip(bars, data[col]):
        label = f"{val:,.0f}" if val >= 100 else f"{val:.1f}"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ylim_top * 0.02,
                label, ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(data[label_col], rotation=45, ha="right", fontsize=7.5)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(axis="y", linewidth=0.5, alpha=0.45)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.5, len(data) - 0.5)


def _plot_quintile_choropleth(gdf, column, ax, linewidth=0.1):
    """Categorical choropleth using PALETTE_QUINTILE; grey for NaN."""
    for cat in QUINTILE_ORDER:
        sub = gdf[gdf[column] == cat]
        if len(sub):
            sub.plot(ax=ax, color=PALETTE_QUINTILE[cat],
                     linewidth=linewidth, edgecolor="white")
    gdf[gdf[column].isna()].plot(
        ax=ax, color="lightgrey", linewidth=linewidth, edgecolor="white")


def _quintile_patches(include_nodata=True):
    patches = [mpatches.Patch(facecolor=PALETTE_QUINTILE[c], label=c)
               for c in QUINTILE_ORDER]
    if include_nodata:
        patches.append(mpatches.Patch(facecolor="lightgrey", label="No data"))
    return patches


def _map_off(ax, title, fontsize=12):
    ax.set_axis_off()
    ax.set_title(title, fontsize=fontsize, fontweight="bold", pad=8)



def _folium_base() -> folium.Map:
    return folium.Map(location=[-9.19, -75.0], zoom_start=5,
                      tiles="CartoDB positron")


def _slim_geo(gdf: gpd.GeoDataFrame, tolerance: float = 0.01) -> gpd.GeoDataFrame:
    """Return a copy with simplified geometry to reduce HTML file size (~1 km tolerance)."""
    out = gdf.copy()
    out["geometry"] = out["geometry"].simplify(tolerance, preserve_topology=True)
    return out


def _add_title(m: folium.Map, text: str) -> None:
    """Inject a centred title banner at the top of a Folium map."""
    html = (
        f'<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);'
        f'z-index:9999;background:rgba(255,255,255,0.92);padding:8px 18px;'
        f'border-radius:5px;box-shadow:0 2px 6px rgba(0,0,0,.3);'
        f'font-family:Arial,sans-serif;font-size:13px;font-weight:bold;'
        f'text-align:center;max-width:680px;">{text}</div>'
    )
    m.get_root().html.add_child(folium.Element(html))


def _html_legend(m: folium.Map, title: str, items: list) -> None:
    """Add a bottom-left colour legend to a Folium map."""
    rows = "".join(
        f'<div style="display:flex;align-items:center;margin-bottom:4px;">'
        f'<div style="width:16px;height:16px;background:{c};opacity:0.85;'
        f'margin-right:8px;flex-shrink:0;border-radius:2px;"></div>'
        f'<span>{lbl}</span></div>'
        for c, lbl in items
    )
    html = (
        f'<div style="position:fixed;bottom:30px;left:20px;z-index:9999;'
        f'background:rgba(255,255,255,0.93);padding:10px 14px;border-radius:6px;'
        f'box-shadow:0 2px 6px rgba(0,0,0,.3);font-family:Arial,sans-serif;'
        f'font-size:12px;max-width:240px;">'
        f'<b style="display:block;margin-bottom:6px;">{title}</b>{rows}</div>'
    )
    m.get_root().html.add_child(folium.Element(html))


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 4 — Static charts (matplotlib / seaborn)
# ═══════════════════════════════════════════════════════════════════════════════

def fig1_q1_distributions(df: gpd.GeoDataFrame) -> str:
    """
    Q1 — Two-panel: distributions of the two Q1 metrics.
    Left : histogram + KDE of ipress_per_10k with WHO threshold.
    Right: histogram of atenciones_per_10k — the spike at zero is the key finding.
    """
    avail  = df["ipress_per_10k"].dropna()
    activ  = df["atenciones_per_10k"].dropna()
    below  = (avail < WHO_THRESHOLD).sum()
    pct_b  = below / len(avail) * 100
    zero_a = (activ == 0).sum()
    pct_z  = zero_a / len(activ) * 100
    cap_a  = activ.quantile(0.97)
    cap_v  = avail.quantile(0.97)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(avail.clip(upper=cap_v), bins=55, kde=True,
                 color="#2166ac", edgecolor="white", linewidth=0.3, ax=ax1)
    ax1.axvline(WHO_THRESHOLD, color="#d73027", linewidth=2,
                linestyle="--", label=f"WHO minimum = {WHO_THRESHOLD}")
    ax1.set_xlim(0, cap_v)
    ax1.set_xlabel("IPRESS per 10,000 inhabitants")
    ax1.set_ylabel("Number of districts")
    ax1.set_title(
        f"Facility Availability\n"
        f"{below:,} / {len(avail):,} districts ({pct_b:.1f}%) below WHO minimum",
        fontsize=11)
    ax1.legend(fontsize=9)

    sns.histplot(activ.clip(upper=cap_a), bins=60, kde=False,
                 color="#e08214", edgecolor="white", linewidth=0.3, ax=ax2)
    ax2.set_xlim(0, cap_a)
    ax2.set_xlabel("Emergency consultations per 10,000 inhabitants (capped at p97)")
    ax2.set_ylabel("Number of districts")
    ax2.set_title(
        f"Emergency Activity Intensity\n"
        f"{zero_a:,} / {len(activ):,} districts ({pct_z:.1f}%) report zero consultations",
        fontsize=11)
    ax2.text(0.97, 0.96,
             f"Note: {pct_z:.1f}% of districts\nreport 0 consultations",
             transform=ax2.transAxes, ha="right", va="top", fontsize=8.5,
             color="#7f0000",
             bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd", alpha=0.85))

    fig.suptitle(
        "Q1 — Territorial Availability: Facility Presence and Emergency Activity Across Districts",
        fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, "fig1_q1_distributions.png")


def fig2_q1_rankings(df: gpd.GeoDataFrame, n: int = 10) -> str:
    """
    Q1 — Two vertical bar ranking charts (top-n only).
    Left : top-n districts by ipress_per_10k.
    Right: top-n districts by atenciones_per_10k.
    """
    valid = df.dropna(subset=["ipress_per_10k", "atenciones_per_10k"]).copy()
    valid["lbl"] = (valid["distrito"].str.title()
                    + "\n" + valid["departamen"].str.title())

    top_av = valid.nlargest(n, "ipress_per_10k").sort_values(
        "ipress_per_10k", ascending=False)
    top_ac = valid.nlargest(n, "atenciones_per_10k").sort_values(
        "atenciones_per_10k", ascending=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6))

    _vertical_bar_panel(
        ax1, top_av, "ipress_per_10k", "lbl", "#2166ac",
        f"Top {n} — Highest Facility Availability\n(IPRESS per 10,000 inhabitants)",
        "IPRESS per 10k")

    _vertical_bar_panel(
        ax2, top_ac, "atenciones_per_10k", "lbl", "#e08214",
        f"Top {n} — Highest Emergency Activity\n(Consultations per 10,000 inhabitants)",
        "Consultations per 10k")

    fig.suptitle(
        "Q1 — District Rankings: Facility Presence and Emergency Activity",
        fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, "fig2_q1_rankings.png")


def fig_q2_distribution(df: gpd.GeoDataFrame) -> str:
    """
    Q2 — Histogram + KDE of dist_mediana_km with 10 km threshold line.
    Shows how many districts exceed the acceptable-access boundary.
    """
    data     = df["dist_mediana_km"].dropna()
    cap      = data.quantile(0.97)
    above10  = (data > 10).sum()
    pct_10   = above10 / len(data) * 100
    med      = data.median()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data.clip(upper=cap), bins=55, kde=True,
                 color="#4393c3", edgecolor="white", linewidth=0.3, ax=ax)
    ax.axvline(10, color="#d73027", linewidth=2, linestyle="--",
               label="10 km access threshold")
    ax.axvline(med, color="#1a9641", linewidth=1.8, linestyle="-.",
               label=f"National median = {med:.1f} km")
    ax.set_xlim(0, cap)
    ax.set_xlabel(
        "Median distance from populated centers to nearest active IPRESS (km)")
    ax.set_ylabel("Number of districts")
    ax.set_title(
        f"Q2 — Distribution of Spatial Access: Median Distance to Nearest Active IPRESS\n"
        f"{above10:,} of {len(data):,} districts ({pct_10:.1f}%) exceed the 10 km threshold",
        fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.text(0.97, 0.96, f"x-axis capped at p97 = {cap:.0f} km",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
    fig.tight_layout()
    return _save(fig, "fig_q2_distribution.png")


def fig_q2_rankings(df: gpd.GeoDataFrame, n: int = 10) -> str:
    """
    Q2 — Vertical bar ranking: top-n (worst) and bottom-n (best) districts
    by dist_mediana_km.
    """
    valid = df.dropna(subset=["dist_mediana_km"]).copy()
    valid["lbl"] = (valid["distrito"].str.title()
                    + "\n" + valid["departamen"].str.title())

    top    = valid.nlargest(n,  "dist_mediana_km").sort_values(
        "dist_mediana_km", ascending=False)
    bottom = valid.nsmallest(n, "dist_mediana_km").sort_values("dist_mediana_km")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    _vertical_bar_panel(
        ax1, top, "dist_mediana_km", "lbl", "#d73027",
        f"Top {n} — Worst Spatial Access\n(Highest median distance to active IPRESS)",
        "Median distance (km)")

    _vertical_bar_panel(
        ax2, bottom, "dist_mediana_km", "lbl", "#1a9641",
        f"Bottom {n} — Best Spatial Access\n(Lowest median distance to active IPRESS)",
        "Median distance (km)")

    fig.suptitle(
        "Q2 — District Rankings by Spatial Access to Emergency-Active IPRESS",
        fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, "fig_q2_rankings.png")


def fig4_q3_ranking(df: gpd.GeoDataFrame, n: int = 15) -> str:
    """
    Q3 — Top-n best and bottom-n worst districts by baseline access index.
    Component score markers show which pillar drives each district's ranking.
    """
    valid = df.dropna(subset=["indice_baseline",
                               "score_disponibilidad_b",
                               "score_actividad_b",
                               "score_acceso_b"]).copy()
    valid["label"] = (valid["distrito"].str.title()
                      + "\n(" + valid["departamen"].str.title() + ")")

    top    = valid.nlargest(n,  "indice_baseline").sort_values("indice_baseline")
    bottom = valid.nsmallest(n, "indice_baseline").sort_values(
        "indice_baseline", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(19, 9))

    def _panel(ax, data, bar_color, title, title_color):
        ax.barh(data["label"], data["indice_baseline"],
                color=bar_color, edgecolor="white",
                height=0.55, alpha=0.75, zorder=2)
        ax.scatter(data["score_disponibilidad_b"], data["label"],
                   color=COMP_COLORS["Facility availability"],
                   s=60, zorder=5, marker="o", label="Facility availability")
        ax.scatter(data["score_actividad_b"], data["label"],
                   color=COMP_COLORS["Emergency activity"],
                   s=60, zorder=5, marker="D", label="Emergency activity")
        ax.scatter(data["score_acceso_b"], data["label"],
                   color=COMP_COLORS["Spatial access"],
                   s=60, zorder=5, marker="^", label="Spatial access")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Score (0 – 1)", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold", color=title_color)
        ax.legend(title="Component score", fontsize=8, loc="lower right",
                  markerscale=1.1)
        ax.grid(axis="x", linewidth=0.5, alpha=0.4)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=8)

    _panel(axes[0], top,    "#4575b4",
           f"Top {n} Best-Served Districts",    "#4575b4")
    _panel(axes[1], bottom, "#d73027",
           f"Bottom {n} Worst-Served Districts", "#d73027")

    fig.suptitle(
        "Q3 — District Ranking by Baseline Emergency Healthcare Access Index\n"
        "Bars = composite index (0–1); markers show individual component percentile scores",
        fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return _save(fig, "fig4_q3_ranking.png")


def fig5_q4_sensitivity(df: gpd.GeoDataFrame) -> str:
    """
    Q4 — Two-panel: scatter baseline vs alternative + quintile transition heatmap.
    """
    valid = df.dropna(subset=["indice_baseline", "indice_alternativo",
                               "quintil_baseline", "quintil_alternativo"]).copy()
    valid["cambio_abs"] = (
        valid["quintil_alternativo"].astype(float) -
        valid["quintil_baseline"].astype(float)
    ).abs()

    trans = pd.crosstab(
        valid["quintil_baseline"].astype(int),
        valid["quintil_alternativo"].astype(int),
        rownames=["Baseline"],
        colnames=["Alternative"],
    ).reindex(index=range(1, 6), columns=range(1, 6), fill_value=0)
    trans_pct = trans.div(trans.sum(axis=1), axis=0).fillna(0) * 100

    annot = np.empty((5, 5), dtype=object)
    for i in range(5):
        for j in range(5):
            n_   = trans.iloc[i, j]
            pct_ = trans_pct.iloc[i, j]
            annot[i, j] = f"{pct_:.1f}%\n(n={n_})"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    color_map   = {0.0: "#4dac26", 1.0: "#f1b300", 2.0: "#d73027"}
    dot_colors  = valid["cambio_abs"].map(color_map).fillna("#d73027")
    ax1.scatter(valid["indice_baseline"], valid["indice_alternativo"],
                c=dot_colors, alpha=0.50, s=18, edgecolors="none", zorder=3)
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1.2, alpha=0.45,
             label="Perfect agreement")
    legend_elems = [
        mpatches.Patch(facecolor="#4dac26", label="No quintile shift"),
        mpatches.Patch(facecolor="#f1b300", label="1 quintile shift"),
        mpatches.Patch(facecolor="#d73027", label="≥2 quintile shifts"),
        plt.Line2D([0], [0], color="k", linestyle="--",
                   linewidth=1, label="Perfect agreement"),
    ]
    ax1.legend(handles=legend_elems, fontsize=8.5, loc="upper left")
    ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
    ax1.set_xlabel("Baseline index (Component 3 = dist_mediana_km)", fontsize=10)
    ax1.set_ylabel("Alternative index (Component 3 = pct_ccpp_10km)", fontsize=10)
    ax1.set_title("Index Agreement Scatter\nColor = quintile reclassification magnitude",
                  fontsize=11, fontweight="bold")

    qlabels = ["Very low\n(1)", "Low\n(2)", "Medium\n(3)",
               "High\n(4)", "Very high\n(5)"]
    sns.heatmap(
        trans_pct, annot=annot, fmt="",
        cmap="Blues", linewidths=0.5,
        cbar_kws={"label": "Row % (share of baseline quintile)", "shrink": 0.8},
        ax=ax2, vmin=0, vmax=100,
    )
    ax2.set_xticklabels(qlabels, rotation=0, fontsize=8)
    ax2.set_yticklabels(qlabels, rotation=0, fontsize=8)
    ax2.set_xlabel("Alternative Quintile", fontsize=10)
    ax2.set_ylabel("Baseline Quintile", fontsize=10)
    ax2.set_title("Quintile Transition Matrix\n"
                  "Row % = share of baseline quintile reclassified by alternative",
                  fontsize=11, fontweight="bold")

    fig.suptitle(
        "Q4 — Methodological Sensitivity: Baseline (dist_mediana_km) vs "
        "Alternative (pct_ccpp_10km) Specification",
        fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return _save(fig, "fig5_q4_sensitivity.png")


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 5 — Static maps (GeoPandas)
# ═══════════════════════════════════════════════════════════════════════════════

def map1_q1_dual(df: gpd.GeoDataFrame) -> str:
    """Q1 — Two-panel choropleth with dark district borders (improved visibility)."""
    gdf = df.copy()
    cap_v = gdf["ipress_per_10k"].quantile(0.95)
    cap_a = gdf["atenciones_per_10k"].quantile(0.95)
    gdf["avail_cap"] = gdf["ipress_per_10k"].clip(upper=cap_v)
    gdf["activ_cap"] = gdf["atenciones_per_10k"].clip(upper=cap_a)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 14))

    gdf.plot(column="avail_cap", cmap="Blues", linewidth=0.25,
             edgecolor="#444444", legend=True, ax=ax1,
             legend_kwds={"label": f"IPRESS per 10k (p95 cap = {cap_v:.1f})",
                          "shrink": 0.45},
             missing_kwds={"color": "lightgrey", "label": "No data"})
    _map_off(ax1, "Facility Availability per 10,000 Inhabitants\n(Component 1 — Q1)")

    gdf.plot(column="activ_cap", cmap="Purples", linewidth=0.25,
             edgecolor="#444444", legend=True, ax=ax2,
             legend_kwds={"label": f"Emergency consults per 10k (p95 cap = {cap_a:.0f})",
                          "shrink": 0.45},
             missing_kwds={"color": "lightgrey", "label": "No data"})
    _map_off(ax2, "Emergency Activity Intensity per 10,000 Inhabitants\n(Component 2 — Q1)")

    fig.suptitle("Q1 — Territorial Availability of Emergency Healthcare — Peru, 2024",
                 fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout()
    return _save(fig, "map1_q1_dual.png")


def map2_q2_dist_mediana(df: gpd.GeoDataFrame) -> str:
    """Q2 — Choropleth of dist_mediana_km (darker = farther = worse access)."""
    gdf   = df.copy()
    cap   = gdf["dist_mediana_km"].quantile(0.97)
    gdf["dist_cap"] = gdf["dist_mediana_km"].clip(upper=cap)

    fig, ax = plt.subplots(figsize=(10, 14))
    gdf.plot(column="dist_cap", cmap="YlOrRd", linewidth=0.1, edgecolor="white",
             legend=True, ax=ax,
             legend_kwds={"label": f"Median distance (km, p97 cap = {cap:.0f} km)",
                          "shrink": 0.45},
             missing_kwds={"color": "lightgrey", "label": "No data"})
    _map_off(ax, "Q2 — Spatial Access: Median Distance from Populated Centers\n"
                  "to Nearest Active IPRESS (km)  |  Darker = farther = worse")
    fig.tight_layout()
    return _save(fig, "map2_q2_dist_mediana.png")


def map3_q3_baseline_quintile(df: gpd.GeoDataFrame) -> str:
    """Q3 — Categorical choropleth of baseline access quintile (5 classes)."""
    fig, ax = plt.subplots(figsize=(10, 14))
    _plot_quintile_choropleth(df, "etiqueta_baseline", ax)
    _map_off(ax, "Q3 — Emergency Healthcare Access Index — Baseline Quintile\n"
                  "Peru, 2024  |  Red = Very low access → Blue = Very high access")
    ax.legend(handles=_quintile_patches(),
              loc="lower left", fontsize=9,
              title="Access quintile", title_fontsize=9, framealpha=0.85)
    fig.tight_layout()
    return _save(fig, "map3_q3_baseline_quintile.png")


def map4_q4_comparison(df: gpd.GeoDataFrame) -> str:
    """Q4 — Two-panel: baseline quintile (left) vs quintile shift magnitude (right)."""
    gdf = df.copy()
    gdf["cambio_abs"] = (
        gdf["quintil_alternativo"].astype(float) -
        gdf["quintil_baseline"].astype(float)
    ).abs()

    shift_colors = {0.0: "#4dac26", 1.0: "#f1b300", 2.0: "#d73027"}
    shift_order  = [0.0, 1.0, 2.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 14))

    _plot_quintile_choropleth(gdf, "etiqueta_baseline", ax1)
    _map_off(ax1, "Baseline Quintile Classification\n(Component 3 = dist_mediana_km)")
    ax1.legend(handles=_quintile_patches(),
               loc="lower left", fontsize=8,
               title="Baseline quintile", title_fontsize=8, framealpha=0.85)

    for val, color in zip(shift_order, [shift_colors[k] for k in shift_order]):
        sub = gdf[gdf["cambio_abs"] == val]
        if len(sub):
            sub.plot(ax=ax2, color=color, linewidth=0.1, edgecolor="white")
    gdf[gdf["cambio_abs"].isna()].plot(
        ax=ax2, color="lightgrey", linewidth=0.1, edgecolor="white")

    _map_off(ax2, "Quintile Shift: Baseline → Alternative Specification\n"
                  "(pct_ccpp_10km replaces dist_mediana_km as Component 3)")
    shift_patches = [
        mpatches.Patch(facecolor="#4dac26", label="Stable (0 quintile shift)"),
        mpatches.Patch(facecolor="#f1b300", label="Moderate (1 quintile shift)"),
        mpatches.Patch(facecolor="#d73027", label="Large (≥2 quintile shifts)"),
        mpatches.Patch(facecolor="lightgrey", label="No data"),
    ]
    ax2.legend(handles=shift_patches, loc="lower left", fontsize=8,
               title="Quintile shift", title_fontsize=8, framealpha=0.85)

    fig.suptitle(
        "Q4 — Methodological Sensitivity: Impact of Changing the Spatial Access Metric",
        fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout()
    return _save(fig, "map4_q4_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 5 — Interactive maps (Folium)
# ═══════════════════════════════════════════════════════════════════════════════

def folium_q3_indice(df: gpd.GeoDataFrame) -> str:
    """Q3 — Interactive choropleth of indice_baseline with all component values."""
    gdf = df[["ubigeo", "distrito", "departamen", "poblacion",
              "indice_baseline", "etiqueta_baseline",
              "ipress_per_10k", "atenciones_per_10k", "dist_mediana_km",
              "score_disponibilidad_b", "score_actividad_b", "score_acceso_b",
              "geometry"]].copy()
    gdf = gdf.to_crs("EPSG:4326")
    for col in ["indice_baseline", "ipress_per_10k", "dist_mediana_km",
                "score_disponibilidad_b", "score_actividad_b", "score_acceso_b"]:
        gdf[col] = gdf[col].round(3)
    gdf["atenciones_per_10k"] = gdf["atenciones_per_10k"].round(0)

    m = _folium_base()
    folium.Choropleth(
        geo_data=gdf.__geo_interface__,
        data=gdf,
        columns=["ubigeo", "indice_baseline"],
        key_on="feature.properties.ubigeo",
        fill_color="RdYlBu",
        fill_opacity=0.75,
        line_opacity=0.1,
        legend_name="Baseline Emergency Access Index (Q3)",
        nan_fill_color="lightgrey",
    ).add_to(m)
    folium.GeoJson(
        gdf.__geo_interface__,
        style_function=lambda f: {"fillOpacity": 0, "weight": 0},
        tooltip=GeoJsonTooltip(
            fields=["distrito", "departamen", "poblacion",
                    "indice_baseline", "etiqueta_baseline",
                    "score_disponibilidad_b", "score_actividad_b", "score_acceso_b",
                    "ipress_per_10k", "atenciones_per_10k", "dist_mediana_km"],
            aliases=["District:", "Department:", "Population:",
                     "Baseline index:", "Classification:",
                     "Component 1 (availability):", "Component 2 (activity):",
                     "Component 3 (spatial access):",
                     "IPRESS per 10k:", "Consults per 10k:", "Median dist (km):"],
            localize=True,
        ),
    ).add_to(m)

    path = os.path.join(OUTPUT_DIR, "mapa_q3_indice_baseline.html")
    m.save(path)
    print(f"  Saved → {path}")
    return path


def folium_q4_comparison(df: gpd.GeoDataFrame) -> str:
    """
    Q4 — Two-layer choropleth with layer control mirroring map4_q4_comparison.png:
         Layer 1 (default on):  Baseline quintile — 5 classes, same palette as static map.
         Layer 2 (toggle):      Quintile shift magnitude — green/yellow/red.
    Each district tooltip shows both specifications and the shift size.
    """
    gdf = df[["ubigeo", "distrito", "departamen",
              "indice_baseline",    "etiqueta_baseline",
              "indice_alternativo", "etiqueta_alternativo",
              "quintil_baseline",   "quintil_alternativo",
              "geometry"]].copy()
    gdf = gdf.to_crs("EPSG:4326")
    gdf = _slim_geo(gdf)
    gdf["cambio_abs"] = (
        gdf["quintil_alternativo"].astype(float) -
        gdf["quintil_baseline"].astype(float)
    ).abs().round(0)
    for col in ["indice_baseline", "indice_alternativo"]:
        gdf[col] = gdf[col].round(3)

    geo = gdf.__geo_interface__
    m   = _folium_base()

    # ── Layer 1: Baseline quintile (left panel of static map) ─────────────────
    _pal = PALETTE_QUINTILE

    def _style_baseline(feature):
        label = feature["properties"].get("etiqueta_baseline")
        return {
            "fillColor":   _pal.get(label, "#d3d3d3"),
            "fillOpacity": 0.75,
            "weight":      0.3,
            "color":       "white",
        }

    fg_baseline = folium.FeatureGroup(
        name="Baseline Quintile  (Component 3 = dist_mediana_km)", show=True
    )
    folium.GeoJson(
        geo,
        style_function=_style_baseline,
        tooltip=GeoJsonTooltip(
            fields=["distrito", "departamen",
                    "etiqueta_baseline",    "indice_baseline",
                    "etiqueta_alternativo", "indice_alternativo",
                    "cambio_abs"],
            aliases=["District:", "Department:",
                     "Baseline class:", "Baseline index:",
                     "Alternative class:", "Alternative index:",
                     "Quintile shift (|Δ|):"],
            localize=True,
        ),
    ).add_to(fg_baseline)
    fg_baseline.add_to(m)

    # ── Layer 2: Quintile shift (right panel of static map) ───────────────────
    _shift_pal = {0.0: "#4dac26", 1.0: "#f1b300", 2.0: "#d73027"}

    def _style_shift(feature):
        val = feature["properties"].get("cambio_abs")
        try:
            key = min(float(val), 2.0)
        except (TypeError, ValueError):
            key = None
        return {
            "fillColor":   _shift_pal.get(key, "#d3d3d3"),
            "fillOpacity": 0.75,
            "weight":      0.3,
            "color":       "white",
        }

    fg_shift = folium.FeatureGroup(
        name="Quintile Shift: Baseline → Alternative", show=False
    )
    folium.GeoJson(
        geo,
        style_function=_style_shift,
        tooltip=GeoJsonTooltip(
            fields=["distrito", "departamen",
                    "quintil_baseline",    "etiqueta_baseline",
                    "quintil_alternativo", "etiqueta_alternativo",
                    "cambio_abs"],
            aliases=["District:", "Department:",
                     "Baseline quintile:", "Baseline class:",
                     "Alternative quintile:", "Alternative class:",
                     "Quintile shift (|Δ|):"],
            localize=True,
        ),
    ).add_to(fg_shift)
    fg_shift.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    _add_title(
        m,
        "Q4 — Methodological Sensitivity: Baseline Quintile vs. Alternative Specification"
        " (pct_ccpp_10km replaces dist_mediana_km)",
    )
    _html_legend(m, "Baseline Quintile", [
        (_pal["Very low"],  "Very low  (Q1)"),
        (_pal["Low"],       "Low  (Q2)"),
        (_pal["Medium"],    "Medium  (Q3)"),
        (_pal["High"],      "High  (Q4)"),
        (_pal["Very high"], "Very high  (Q5)"),
        ("#d3d3d3",         "No data"),
    ])

    path = os.path.join(OUTPUT_DIR, "mapa_q4_dual_comparison.html")
    m.save(path)
    print(f"  Saved → {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_visualization_pipeline(
    processed_path: str = "data/processed",
    output_path:    str = "output/figures",
) -> dict:
    """
    Generate all charts and maps for Tasks 4 and 5.

    Static charts  (Task 4)            — output/figures/fig*.png
    Static maps    (Task 5, GeoPandas) — output/figures/map*.png
    Interactive    (Task 5, Folium)    — output/figures/mapa_*.html
    """
    global OUTPUT_DIR
    OUTPUT_DIR = output_path
    os.makedirs(output_path, exist_ok=True)

    print("Loading data...")
    metrics = gpd.read_file(f"{processed_path}/distrito_metrics.geojson")
    metrics["ubigeo"] = metrics["ubigeo"].astype(str).str.zfill(6)

    paths = {}

    print("\n── Q1 charts ──")
    paths["fig1"] = fig1_q1_distributions(metrics)
    paths["fig2"] = fig2_q1_rankings(metrics)

    print("\n── Q2 charts ──")
    paths["fig_q2_dist"]     = fig_q2_distribution(metrics)
    paths["fig_q2_rankings"] = fig_q2_rankings(metrics)

    print("\n── Q3 chart ──")
    paths["fig4"] = fig4_q3_ranking(metrics)

    print("\n── Q4 chart ──")
    paths["fig5"] = fig5_q4_sensitivity(metrics)

    print("\n── Static maps (Task 5 — GeoPandas) ──")
    paths["map1"] = map1_q1_dual(metrics)
    paths["map2"] = map2_q2_dist_mediana(metrics)
    paths["map3"] = map3_q3_baseline_quintile(metrics)
    paths["map4"] = map4_q4_comparison(metrics)

    print("\n── Interactive maps (Task 5 — Folium) ──")
    paths["folium_q3"] = folium_q3_indice(metrics)
    paths["folium_q4"] = folium_q4_comparison(metrics)

    print(f"\n{'─'*52}")
    print(f"All outputs saved → {output_path}/")
    print(f"  Static charts : 6  (fig1, fig2, fig_q2_dist, fig_q2_rankings, fig4, fig5)")
    print(f"  Static maps   : 4  (map1–map4)")
    print(f"  Folium maps   : 2  (mapa_q3, mapa_q4)")
    return paths


if __name__ == "__main__":
    run_visualization_pipeline()
