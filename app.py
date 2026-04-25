"""
app.py — Emergency Healthcare Access in Peru
Streamlit application — 4 tabs as required by the assignment.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import geopandas as gpd
import folium
from folium.features import GeoJsonTooltip

from visualization import _folium_base, _slim_geo, _add_title, PALETTE_QUINTILE

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emergency Healthcare Access — Peru",
    page_icon="🏥",
    layout="wide",
)

st.title("Emergency Healthcare Access Inequality in Peru")

# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data
def load_metrics():
    df = pd.read_csv("output/tables/distrito_metrics.csv")
    df["ubigeo"] = df["ubigeo"].astype(str).str.zfill(6)
    return df


@st.cache_data
def load_comparison():
    df = pd.read_csv("output/tables/comparacion_baseline_alternativo.csv")
    df["ubigeo"] = df["ubigeo"].astype(str).str.zfill(6)
    return df


@st.cache_resource
def load_geodata():
    gdf = gpd.read_file("data/processed/distrito_metrics.geojson")
    gdf["ubigeo"] = gdf["ubigeo"].astype(str).str.zfill(6)
    return gdf


@st.cache_resource
def build_q3_map_html():
    """Build Q3 Folium map inline with simplified geometry (avoids the 131 MB file)."""
    gdf = load_geodata()
    cols = [
        "ubigeo", "distrito", "departamen", "poblacion",
        "indice_baseline", "etiqueta_baseline",
        "ipress_per_10k", "atenciones_per_10k", "dist_mediana_km",
        "score_disponibilidad_b", "score_actividad_b", "score_acceso_b",
        "geometry",
    ]
    gdf = gdf[cols].copy().to_crs("EPSG:4326")
    gdf = _slim_geo(gdf)
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
        legend_name="Baseline Emergency Access Index",
        nan_fill_color="lightgrey",
    ).add_to(m)
    folium.GeoJson(
        gdf.__geo_interface__,
        style_function=lambda f: {"fillOpacity": 0, "weight": 0},
        tooltip=GeoJsonTooltip(
            fields=[
                "distrito", "departamen", "poblacion",
                "indice_baseline", "etiqueta_baseline",
                "score_disponibilidad_b", "score_actividad_b", "score_acceso_b",
                "ipress_per_10k", "atenciones_per_10k", "dist_mediana_km",
            ],
            aliases=[
                "District:", "Department:", "Population:",
                "Baseline index:", "Classification:",
                "Availability score:", "Activity score:", "Spatial access score:",
                "IPRESS per 10k:", "Consults per 10k:", "Median dist (km):",
            ],
            localize=True,
        ),
    ).add_to(m)
    _add_title(m, "Q3 — Baseline Emergency Healthcare Access Index by District")
    return m._repr_html_()


@st.cache_data
def load_q4_html():
    with open("output/figures/mapa_q4_dual_comparison.html", "r", encoding="utf-8") as f:
        return f.read()


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Data & Methodology",
    "📊 Static Analysis",
    "🗺️ Geospatial Results",
    "🔍 Interactive Exploration",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Data & Methodology
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Data & Methodology")

    # ── Problem statement ──────────────────────────────────────────────────────
    st.subheader("Problem Statement")
    st.markdown("""
    Emergency healthcare access in Peru is highly unequal across its 1,873 districts.
    Many districts — particularly in rural Amazonia and the highlands — have few or no
    health facilities with emergency activity, and their populated centers lie tens of
    kilometres from the nearest active facility.

    This project answers: **Which districts in Peru appear relatively better or worse
    served in emergency healthcare access, and what evidence supports that conclusion?**

    We build a composite district-level index combining three dimensions of access:
    - **Facility availability** — active health facilities per 10,000 inhabitants
    - **Emergency activity intensity** — emergency consultations per 10,000 inhabitants
    - **Spatial access** — distance from populated centers to the nearest active facility
    """)

    # ── Data sources ───────────────────────────────────────────────────────────
    st.subheader("Data Sources")
    st.dataframe(
        pd.DataFrame({
            "Dataset": [
                "IPRESS Health Facilities",
                "Emergency Care Production by IPRESS",
                "Populated Centers (Centros Poblados)",
                "District Boundaries (DISTRITOS.shp)",
            ],
            "Source": ["MINSA", "MINSA", "INEI", "INEI"],
            "Raw records": [
                "~13,000 facilities",
                "~6,000 facility-period records",
                "~100,000 populated centers",
                "1,874 districts",
            ],
            "Role in analysis": [
                "Facility locations and per-capita availability (Component 1)",
                "Emergency consultations per district (Component 2)",
                "Origin points for spatial access calculations (Component 3)",
                "Unit of analysis for all district-level aggregations",
            ],
        }),
        use_container_width=True,
        hide_index=True,
    )

    # ── Cleaning summary ───────────────────────────────────────────────────────
    st.subheader("Cleaning Summary")
    st.markdown("""
    **IPRESS facilities**
    - Removed facilities with missing or invalid coordinates (outside Peru's geographic bounding box)
    - Standardised facility identifiers (`codigo_renaes`) for joining to emergency activity data
    - Facilities without a district match after spatial join were excluded

    **Emergency care production**
    - Aggregated to facility level (sum of `total_atenciones` and `total_atendidos`)
    - Districts reporting zero consultations were **retained** — this is a meaningful signal,
      not missing data; **75.6% of districts report zero emergency consultations**

    **Populated centers**
    - Removed records with missing or zero coordinates
    - Spatial join to district boundaries using point-in-polygon
      (reprojected to EPSG:32718 for distance calculations)

    **District boundaries**
    - Source CRS: EPSG:4326 (WGS 84) — retained for display
    - Reprojected to EPSG:32718 (UTM Zone 18S) for all metric distance computations
    - 1 district dropped due to missing population data → **final sample: 1,873 districts**
    """)

    # ── Methodological decisions ───────────────────────────────────────────────
    st.subheader("Methodological Decisions — Index Construction")
    st.markdown("""
    **Index formula (both specifications)**
    ```
    index = (score_availability + score_activity + score_access) / 3
    ```
    All components are normalised using **percentile rank**, which maps each district's raw
    value to its position in the national distribution. This is robust to Peru's extreme
    outliers (Lima districts vs. remote Amazonian districts).

    **Equal weights** are used because:
    - There is no strong theoretical basis to weight one pillar over another
    - The three components are weakly correlated (PCA PC1 explains ~37% of variance),
      meaning they capture genuinely different dimensions of access
    - Equal-weight composite indices are standard in the literature (e.g., UNDP Human Development Index)

    **Two specifications for Q4 — Methodological Sensitivity**
    """)
    st.table(
        pd.DataFrame({
            "": ["Metric used", "Direction", "Rationale"],
            "Baseline": [
                "dist_mediana_km — median km from populated centers to nearest active IPRESS",
                "Lower = better (inverted before scoring)",
                "Pure physical proximity — captures how far communities are from any emergency-active facility",
            ],
            "Alternative": [
                "pct_ccpp_10km — % of populated centers within 10 km of nearest active IPRESS",
                "Higher = better (used directly)",
                "Coverage metric — captures how many communities are actually within acceptable reach",
            ],
        }).set_index("")
    )

    # ── Limitations ────────────────────────────────────────────────────────────
    st.subheader("Limitations")
    st.markdown("""
    - **Zero-activity districts**: 75.6% of districts report zero emergency consultations.
      This may reflect data gaps as much as true absence of activity, deflating Component 2
      for a large share of the country.
    - **NaN imputation**: Districts with no populated centers or no nearby active IPRESS have
      `dist_mediana_km` imputed with the national median (26 km) — a conservative assumption.
    - **Snapshot in time**: The analysis reflects one reporting period; seasonal or annual
      variation is not captured.
    - **Distance definition**: Spatial access is measured to the nearest active IPRESS (any
      emergency activity reported), not to a facility specifically equipped for the patient's
      emergency type.
    - **Population data**: District population figures may not precisely match the period of
      the health activity data.
    """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Static Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Static Analysis")

    # ── Q1 ─────────────────────────────────────────────────────────────────────
    st.subheader("Q1 — Territorial Availability")
    st.markdown(
        "_Which districts appear to have lower or higher availability of health "
        "facilities and emergency care activity?_"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.image("output/figures/fig1_q1_distributions.png", use_column_width=True)
        st.caption(
            "**Distribution of Q1 metrics.** The left panel shows that most districts "
            "fall below the WHO minimum of 2 IPRESS per 10,000 inhabitants. The right "
            "panel reveals a sharp spike at zero consultations — over 75% of districts "
            "report no emergency activity. A histogram with KDE was chosen because the "
            "shape of the distribution (skew, threshold crossing, zero spike) is the "
            "central evidence for Q1. A bar chart of means would have hidden the "
            "zero-activity phenomenon entirely."
        )
    with col2:
        st.image("output/figures/fig2_q1_rankings.png", use_column_width=True)
        st.caption(
            "**Top-10 district rankings** by facility availability (left) and emergency "
            "activity (right). Ranking bar charts are used rather than a scatter plot "
            "because Q1 asks which districts are better or worse — a named ranking "
            "directly answers that. The two panels show that top performers differ "
            "across metrics: high facility count does not guarantee high activity."
        )

    st.divider()

    # ── Q2 ─────────────────────────────────────────────────────────────────────
    st.subheader("Q2 — Settlement Access")
    st.markdown(
        "_Which districts seem to have populated centers with weaker spatial access "
        "to emergency-related health services?_"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.image("output/figures/fig_q2_distribution.png", use_column_width=True)
        st.caption(
            "**Distribution of median district distance** from populated centers to the "
            "nearest active IPRESS. The 10 km threshold line shows what fraction of "
            "districts have acceptable proximity. A histogram with KDE was chosen over "
            "a boxplot because the tail shape — how many districts extend beyond 50 or "
            "100 km — is the key spatial evidence. The national median line adds "
            "geographic context."
        )
    with col2:
        st.image("output/figures/fig_q2_rankings.png", use_column_width=True)
        st.caption(
            "**Worst and best districts by spatial access.** The side-by-side ranking "
            "directly names the districts with the most and least isolated populated "
            "centers. Vertical bar rankings are used rather than a choropleth because "
            "they allow precise value comparison and district identification at the "
            "individual level, which a national-scale map cannot provide."
        )

    st.divider()

    # ── Q3 ─────────────────────────────────────────────────────────────────────
    st.subheader("Q3 — District Comparison")
    st.markdown(
        "_Which districts appear most underserved and which appear best served when "
        "combining all three dimensions?_"
    )
    st.image("output/figures/fig4_q3_ranking.png", use_column_width=True)
    st.caption(
        "**Top-15 and bottom-15 districts by baseline composite index.** Horizontal bars "
        "show the overall score; overlaid scatter markers show each component score "
        "individually, revealing which pillar drives a district's ranking. This combined "
        "view was chosen over a simple sorted table because it shows both the overall "
        "ranking and the decomposition simultaneously — critical for justifying the "
        "classification. A radar chart was considered but becomes illegible with 15+ districts."
    )

    st.divider()

    # ── Q4 ─────────────────────────────────────────────────────────────────────
    st.subheader("Q4 — Methodological Sensitivity")
    st.markdown(
        "_How much do the district results change if the spatial access definition changes?_"
    )
    st.image("output/figures/fig5_q4_sensitivity.png", use_column_width=True)
    st.caption(
        "**Left: index agreement scatter.** Each point is a district coloured by quintile "
        "shift magnitude (green = stable, yellow = 1 quintile, red = ≥2 quintiles). Most "
        "districts cluster near the diagonal, indicating broad agreement between the two "
        "specifications. **Right: quintile transition matrix.** Row percentages show what "
        "fraction of each baseline quintile is reclassified by the alternative. The "
        "scatter + heatmap combination was chosen because the scatter reveals overall "
        "correlation while the matrix reveals where disagreement concentrates — a single "
        "chart type could not show both simultaneously."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Geospatial Results
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Geospatial Results")

    # ── Static maps ────────────────────────────────────────────────────────────
    st.subheader("Q1 — Territorial Availability")
    st.image("output/figures/map1_q1_dual.png", use_column_width=True)
    st.caption(
        "Facility availability per 10,000 inhabitants (left) and emergency activity "
        "intensity (right) by district. Darker = higher values. Grey = no data. "
        "Urban coastal districts and Lima show the highest values; the Amazon basin "
        "and southern highlands are the most underserved."
    )

    st.divider()

    st.subheader("Q2 — Spatial Access")
    st.image("output/figures/map2_q2_dist_mediana.png", use_column_width=True)
    st.caption(
        "Median distance from populated centers to the nearest active IPRESS (km). "
        "Darker orange/red = more isolated districts. The Amazon basin shows the "
        "largest distances, with several districts exceeding 100 km."
    )

    st.divider()

    st.subheader("Q3 — Baseline Emergency Access Index")
    st.image("output/figures/map3_q3_baseline_quintile.png", use_column_width=True)
    st.caption(
        "Quintile classification of the baseline composite index. "
        "Red = Very low access (Q1). Blue = Very high access (Q5). "
        "Low-access districts concentrate in Loreto, Ucayali, and parts of the "
        "southern highlands."
    )

    st.divider()

    st.subheader("Q4 — Methodological Sensitivity")
    st.image("output/figures/map4_q4_comparison.png", use_column_width=True)
    st.caption(
        "Left: baseline quintile classification (dist_mediana_km as Component 3). "
        "Right: quintile shift magnitude when switching to the alternative (pct_ccpp_10km). "
        "Green = stable. Yellow = 1 quintile shift. Red = ≥2 quintile shifts. "
        "Most districts remain stable; large shifts are geographically concentrated."
    )

    st.divider()

    # ── District-level comparison table ────────────────────────────────────────
    st.subheader("District-Level Comparison Table")

    df_metrics = load_metrics()

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        dept_sel = st.selectbox(
            "Department",
            ["All"] + sorted(df_metrics["departamen"].dropna().unique().tolist()),
            key="tab3_dept",
        )
    with col_f2:
        q_sel = st.selectbox(
            "Baseline quintile",
            ["All", "Very low", "Low", "Medium", "High", "Very high"],
            key="tab3_q",
        )
    with col_f3:
        view_sel = st.radio(
            "Show",
            ["All districts", "Top 20 (best)", "Bottom 20 (worst)"],
            horizontal=True,
            key="tab3_view",
        )

    mask = pd.Series(True, index=df_metrics.index)
    if dept_sel != "All":
        mask &= df_metrics["departamen"] == dept_sel
    if q_sel != "All":
        mask &= df_metrics["etiqueta_baseline"] == q_sel

    display = df_metrics.loc[
        mask,
        [
            "distrito", "departamen", "poblacion",
            "ipress_per_10k", "atenciones_per_10k", "dist_mediana_km",
            "indice_baseline", "etiqueta_baseline",
            "indice_alternativo", "etiqueta_alternativo",
        ],
    ].copy().sort_values("indice_baseline", ascending=False)

    display.columns = [
        "District", "Department", "Population",
        "IPRESS / 10k", "Consults / 10k", "Median dist (km)",
        "Baseline index", "Baseline class",
        "Alt. index", "Alt. class",
    ]

    if view_sel == "Top 20 (best)":
        display = display.head(20)
    elif view_sel == "Bottom 20 (worst)":
        display = display.tail(20).sort_values("Baseline index")

    st.dataframe(
        display.style.format({
            "Population":       "{:,.0f}",
            "IPRESS / 10k":     "{:.2f}",
            "Consults / 10k":   "{:,.0f}",
            "Median dist (km)": "{:.1f}",
            "Baseline index":   "{:.3f}",
            "Alt. index":       "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        f"Showing {len(display):,} districts · sorted by baseline index (descending)."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Interactive Exploration
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Interactive Exploration")

    # ── Q3 Folium map ──────────────────────────────────────────────────────────
    st.subheader("Q3 — Baseline Access Index (Interactive Choropleth)")
    st.markdown(
        "Hover over any district to see its baseline index, quintile classification, "
        "and all three component scores."
    )
    with st.spinner("Building interactive map…"):
        q3_html = build_q3_map_html()
    components.html(q3_html, height=550, scrolling=False)

    st.divider()

    # ── Q4 Folium map ──────────────────────────────────────────────────────────
    st.subheader("Q4 — Baseline vs Alternative Specification (Interactive)")
    st.markdown(
        "Use the layer control (top-right corner) to toggle between the **baseline "
        "quintile** layer and the **quintile-shift** layer. Hover for per-district details."
    )
    components.html(load_q4_html(), height=550, scrolling=False)

    st.divider()

    # ── Multi-district comparison view ─────────────────────────────────────────
    st.subheader("District Comparison View")
    st.markdown("Select up to 10 districts to compare their metrics side by side.")

    df_metrics = load_metrics()
    district_labels = (
        df_metrics["distrito"].str.title()
        + " — "
        + df_metrics["departamen"].str.title()
    )
    label_to_idx = dict(zip(district_labels, df_metrics.index))
    sorted_labels = sorted(label_to_idx.keys())

    selected_labels = st.multiselect(
        "Choose districts",
        options=sorted_labels,
        default=sorted_labels[:3],
        max_selections=10,
        key="tab4_multiselect",
    )

    if selected_labels:
        sel_idx = [label_to_idx[lbl] for lbl in selected_labels]
        comp_view = df_metrics.loc[
            sel_idx,
            [
                "distrito", "departamen",
                "ipress_per_10k", "atenciones_per_10k", "dist_mediana_km",
                "score_disponibilidad_b", "score_actividad_b", "score_acceso_b",
                "indice_baseline", "etiqueta_baseline",
                "indice_alternativo", "etiqueta_alternativo",
            ],
        ].copy()
        comp_view.columns = [
            "District", "Department",
            "IPRESS / 10k", "Consults / 10k", "Median dist (km)",
            "Availability score", "Activity score", "Spatial access score",
            "Baseline index", "Baseline class",
            "Alt. index", "Alt. class",
        ]
        st.dataframe(
            comp_view.style.format({
                "IPRESS / 10k":         "{:.2f}",
                "Consults / 10k":       "{:,.0f}",
                "Median dist (km)":     "{:.1f}",
                "Availability score":   "{:.3f}",
                "Activity score":       "{:.3f}",
                "Spatial access score": "{:.3f}",
                "Baseline index":       "{:.3f}",
                "Alt. index":           "{:.3f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Select at least one district above.")

    st.divider()

    # ── Baseline vs alternative comparison ─────────────────────────────────────
    st.subheader("Baseline vs Alternative — Full Quintile Shift Comparison")
    st.markdown("""
    The table below shows how each district's quintile classification changes between
    the **baseline** (`dist_mediana_km`) and **alternative** (`pct_ccpp_10km`) specification.
    Rows are colour-coded by shift magnitude.
    """)

    df_comp = load_comparison()

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        dept_comp = st.selectbox(
            "Filter by department",
            ["All"] + sorted(df_comp["departamen"].dropna().unique().tolist()),
            key="tab4_dept",
        )
    with col_c2:
        shift_filter = st.selectbox(
            "Filter by quintile shift",
            ["All shifts", "Stable (0)", "Moderate (1 quintile)", "Large (≥2 quintiles)"],
            key="tab4_shift",
        )

    comp_mask = pd.Series(True, index=df_comp.index)
    if dept_comp != "All":
        comp_mask &= df_comp["departamen"] == dept_comp
    if shift_filter == "Stable (0)":
        comp_mask &= df_comp["cambio_abs"] == 0
    elif shift_filter == "Moderate (1 quintile)":
        comp_mask &= df_comp["cambio_abs"] == 1
    elif shift_filter == "Large (≥2 quintiles)":
        comp_mask &= df_comp["cambio_abs"] >= 2

    comp_display = df_comp.loc[
        comp_mask,
        [
            "distrito", "departamen",
            "indice_baseline", "etiqueta_baseline",
            "indice_alternativo", "etiqueta_alternativo",
            "cambio_abs",
        ],
    ].copy().sort_values("cambio_abs", ascending=False)

    comp_display.columns = [
        "District", "Department",
        "Baseline index", "Baseline class",
        "Alt. index", "Alt. class",
        "Quintile shift |Δ|",
    ]

    def _shift_color(val):
        if val == 0:
            return "background-color: #c7e9c0"
        elif val == 1:
            return "background-color: #fff3cd"
        return "background-color: #f5c6cb"

    st.dataframe(
        comp_display.style
        .format({
            "Baseline index":    "{:.3f}",
            "Alt. index":        "{:.3f}",
            "Quintile shift |Δ|": "{:.0f}",
        })
        .map(_shift_color, subset=["Quintile shift |Δ|"]),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        f"Showing {len(comp_display):,} districts. "
        "Green = stable (0 shift) · Yellow = 1 quintile shift · Red = ≥2 quintile shifts."
    )
