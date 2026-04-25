"""
metrics.py
District-level emergency healthcare access index for Peru.

Index components (3 pillars)
───────────────────────────────────────────────────────────────────────────────
1. Facility availability
     ipress_per_10k = (n_establecimientos / poblacion) × 10,000
     Higher = better. WHO minimum threshold: 2 per 10,000 inhabitants.

2. Emergency activity intensity
     atenciones_per_10k = (total_atenciones / poblacion) × 10,000
     Higher = better.

3. Spatial access to emergency services  ← differs between specifications
     BASELINE    : dist_mediana_km — median distance from populated centers
                   to the nearest active IPRESS (any reported activity).
                   Lower = better → inverted before aggregation.
     ALTERNATIVE : pct_ccpp_10km — % of a district's populated centers
                   within 10 km of the nearest active IPRESS.
                   Higher = better → used directly.

Aggregation methodology (identical for both specifications)
───────────────────────────────────────────────────────────────────────────────
All three components are normalized to [0, 1] using percentile rank —
robust to the extreme outliers present in Peru (Lima vs. rural Amazonas).
The final index is the simple average of the three normalized scores:

    index = (score_availability + score_activity + score_access) / 3

Equal weights are used because:
  · No strong theoretical basis to weight one pillar over another.
  · The three components are weakly correlated (PCA PC1 explains only ~37%
    of variance), meaning they capture genuinely different dimensions of
    access — unequal weights would require domain justification.
  · Equal-weight composite indices are standard in the literature
    (e.g., UNDP Human Development Index).

Baseline vs. Alternative — Methodological Sensitivity (Question 4)
───────────────────────────────────────────────────────────────────────────────
The two specifications are identical except for Component 3:
  · Baseline uses dist_mediana_km: pure physical proximity to any facility
    with emergency activity.
  · Alternative uses pct_ccpp_10km: share of populated centers with
    "acceptable" access (≤10 km), a coverage metric that better reflects
    how many communities in a district are actually within reach.

Comparing the quintile rankings produced by each specification directly
answers Question 4: districts whose classification changes substantially
are sensitive to the choice of spatial access metric.

Quintile labels (1 = worst access, 5 = best access)
    1 → Very low   2 → Low   3 → Medium   4 → High   5 → Very high

Outputs
───────
data/processed/distrito_metrics.geojson
output/tables/distrito_metrics.csv
output/tables/comparacion_baseline_alternativo.csv
───────────────────────────────────────────────────────────────────────────────
"""

import os

import numpy as np
import pandas as pd
import geopandas as gpd

WHO_THRESHOLD = 2.0     # minimum IPRESS per 10,000 inhabitants (WHO)

QUINTILE_LABELS = {
    1: "Very low",
    2: "Low",
    3: "Medium",
    4: "High",
    5: "Very high",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _percentile_rank(series: pd.Series) -> pd.Series:
    """Normalize a series to [0, 1] by percentile rank, preserving NaN."""
    return series.rank(method="average", na_option="keep", pct=True)


def _quintile(series: pd.Series) -> pd.Series:
    """Assign quintile 1–5 from a continuous score series."""
    return pd.qcut(
        series.rank(method="first", na_option="keep"),
        q=5,
        labels=[1, 2, 3, 4, 5],
    ).astype("Int64")


# ── Component building ────────────────────────────────────────────────────────

def build_components(summary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Compute the three raw indicators used in both index specifications.

    Edge-case handling
    ──────────────────
    · Districts with missing population → per-capita metrics = NaN.
    · Districts with 0 facilities → pct_con_emergencia = NaN (not used).
    · Districts with NaN dist_mediana_km (no CCPP or no active IPRESS with
      coordinates) → imputed with national median so no district is lost.
    · Districts with NaN pct_ccpp_10km → imputed with 0 (no CCPP within 10 km
      is the conservative assumption for districts with no CCPP data).
    """
    df = summary.copy()

    # ── Component 1: facility availability ──
    df["ipress_per_10k"] = np.where(
        df["poblacion"] > 0,
        (df["n_establecimientos"] / df["poblacion"]) * 10_000,
        np.nan,
    ).round(4)

    df["bajo_umbral_oms"] = df["ipress_per_10k"] < WHO_THRESHOLD

    # ── Component 2: emergency activity ──
    df["atenciones_per_10k"] = np.where(
        df["poblacion"] > 0,
        (df["total_atenciones"] / df["poblacion"]) * 10_000,
        np.nan,
    ).round(4)

    # ── Component 3a: dist_mediana_km (baseline) ──
    # Impute NaN with national median (districts with no CCPP or no active IPRESS)
    mediana_nacional = df["dist_mediana_km"].median()
    n_imp = df["dist_mediana_km"].isna().sum()
    if n_imp > 0:
        print(f"  [METRICS] dist_mediana_km: {n_imp} NaN imputed "
              f"with national median ({mediana_nacional:.1f} km)")
    df["dist_mediana_km_imp"] = df["dist_mediana_km"].fillna(mediana_nacional)

    # ── Component 3b: pct_ccpp_10km (alternative) ──
    # Districts with no CCPP data → 0 (conservative: no coverage)
    n_imp2 = df["pct_ccpp_10km"].isna().sum()
    if n_imp2 > 0:
        print(f"  [METRICS] pct_ccpp_10km: {n_imp2} NaN imputed with 0")
    df["pct_ccpp_10km_imp"] = df["pct_ccpp_10km"].fillna(0)

    return df


# ── Baseline index ────────────────────────────────────────────────────────────

def build_baseline_index(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Baseline access index.

    Component 3 = dist_mediana_km (inverted: 1 − rank so that shorter
    distance → higher score, consistent with the direction of the other two).

        index_baseline = (rank(ipress_per_10k)
                        + rank(atenciones_per_10k)
                        + (1 − rank(dist_mediana_km))) / 3
    """
    df = df.copy()

    s1 = _percentile_rank(df["ipress_per_10k"])
    s2 = _percentile_rank(df["atenciones_per_10k"])
    s3 = 1 - _percentile_rank(df["dist_mediana_km_imp"])   # inverted distance

    df["score_disponibilidad_b"] = s1.round(4)
    df["score_actividad_b"]      = s2.round(4)
    df["score_acceso_b"]         = s3.round(4)

    df["indice_baseline"]  = ((s1 + s2 + s3) / 3).round(4)
    df["quintil_baseline"] = _quintile(df["indice_baseline"])
    df["etiqueta_baseline"] = df["quintil_baseline"].map(QUINTILE_LABELS)

    return df


# ── Alternative index ─────────────────────────────────────────────────────────

def build_alternative_index(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Alternative access index.

    Component 3 = pct_ccpp_10km — share of populated centers within 10 km
    of the nearest active IPRESS. Higher = better; no inversion needed.

        index_alternativo = (rank(ipress_per_10k)
                           + rank(atenciones_per_10k)
                           + rank(pct_ccpp_10km)) / 3
    """
    df = df.copy()

    s1 = _percentile_rank(df["ipress_per_10k"])
    s2 = _percentile_rank(df["atenciones_per_10k"])
    s3 = _percentile_rank(df["pct_ccpp_10km_imp"])          # no inversion needed

    df["score_disponibilidad_a"] = s1.round(4)
    df["score_actividad_a"]      = s2.round(4)
    df["score_acceso_a"]         = s3.round(4)

    df["indice_alternativo"]  = ((s1 + s2 + s3) / 3).round(4)
    df["quintil_alternativo"] = _quintile(df["indice_alternativo"])
    df["etiqueta_alternativo"] = df["quintil_alternativo"].map(QUINTILE_LABELS)

    return df


# ── Comparison: methodological sensitivity ────────────────────────────────────

def build_comparison(df: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Compare baseline vs. alternative quintile rankings.

    The quintile shift measures how sensitive a district's access
    classification is to the choice of spatial access metric.
    Districts shifting ≥2 quintiles are the most sensitive cases
    and anchor the discussion of Question 4.
    """
    cols = [
        "ubigeo", "distrito", "departamen", "poblacion",
        "ipress_per_10k", "atenciones_per_10k",
        "dist_mediana_km", "pct_ccpp_10km",
        "indice_baseline",    "quintil_baseline",    "etiqueta_baseline",
        "indice_alternativo", "quintil_alternativo", "etiqueta_alternativo",
    ]
    comp = df[cols].copy()

    comp["cambio_quintil"] = (
        comp["quintil_alternativo"].astype(float) -
        comp["quintil_baseline"].astype(float)
    )
    comp["cambio_abs"] = comp["cambio_quintil"].abs()

    print(f"\n  [COMPARISON] No quintile change : "
          f"{(comp['cambio_abs'] == 0).sum():>5,}")
    print(f"  [COMPARISON] Change 1 quintile  : "
          f"{(comp['cambio_abs'] == 1).sum():>5,}")
    print(f"  [COMPARISON] Change ≥2 quintiles: "
          f"{(comp['cambio_abs'] >= 2).sum():>5,}")

    return comp.sort_values("cambio_abs", ascending=False)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_metrics_pipeline(
    processed_path: str = "data/processed",
    output_path:    str = "output/tables",
) -> dict:
    """
    Run the full district metrics pipeline and save outputs.

    Outputs
    ───────
    data/processed/distrito_metrics.geojson
    output/tables/distrito_metrics.csv
    output/tables/comparacion_baseline_alternativo.csv
    """
    os.makedirs(output_path, exist_ok=True)

    print("Loading district summary...")
    summary = gpd.read_file(f"{processed_path}/distrito_summary.geojson")
    summary["ubigeo"] = summary["ubigeo"].astype(str).str.zfill(6)

    print("Computing components...")
    df = build_components(summary)

    print("Building BASELINE index (dist_mediana_km)...")
    df = build_baseline_index(df)

    print("Building ALTERNATIVE index (pct_ccpp_10km)...")
    df = build_alternative_index(df)

    print("Comparing specifications...")
    comparacion = build_comparison(df)

    # ── Save ──
    print("\nSaving outputs...")
    df.to_file(f"{processed_path}/distrito_metrics.geojson", driver="GeoJSON")
    df.drop(columns="geometry").to_csv(
        f"{output_path}/distrito_metrics.csv", index=False)
    comparacion.to_csv(
        f"{output_path}/comparacion_baseline_alternativo.csv", index=False)

    # ── Summary report ──
    print(f"\n{'─'*55}")
    print(f"Districts analyzed           : {len(df):>6,}")
    print(f"Below WHO threshold (<2/10k) : {df['bajo_umbral_oms'].sum():>6,}"
          f"  ({100*df['bajo_umbral_oms'].mean():.1f}%)")
    print(f"Median national distance     : {df['dist_mediana_km'].median():.1f} km")
    print(f"Median pct_ccpp_10km         : {df['pct_ccpp_10km'].median():.1f}%")
    print()
    print("Baseline distribution:")
    for label in QUINTILE_LABELS.values():
        n = (df["etiqueta_baseline"] == label).sum()
        print(f"  {label:<12}: {n:>5,}")
    print()
    print("Alternative distribution:")
    for label in QUINTILE_LABELS.values():
        n = (df["etiqueta_alternativo"] == label).sum()
        print(f"  {label:<12}: {n:>5,}")
    print(f"\nOutputs saved → {processed_path}/ and {output_path}/")

    return {
        "distrito_metrics": df,
        "comparacion":      comparacion,
    }


if __name__ == "__main__":
    run_metrics_pipeline()
