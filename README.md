# Emergency Healthcare Access Inequality in Peru

A geospatial analytics pipeline that evaluates emergency healthcare access inequality across Peru's 1,873 districts by combining health facility data, emergency care activity, populated center locations, and district boundaries into a composite district-level access index.

---

## What does the project do?

This project builds a complete end-to-end pipeline that:

1. Loads and cleans four public datasets from MINSA and INEI
2. Performs spatial joins to assign facilities and populated centers to districts
3. Computes a composite emergency healthcare access index for each district
4. Produces static charts, static maps, and interactive Folium maps
5. Presents all results through a 4-tab Streamlit application

---

## Main Analytical Goal

**Which districts in Peru appear relatively better or worse served in emergency healthcare access, and what evidence supports that conclusion?**

The project answers four analytical questions:

- **Q1 — Territorial Availability**: Which districts have lower or higher facility presence and emergency activity per capita?
- **Q2 — Settlement Access**: Which districts have populated centers with weaker spatial access to emergency-active facilities?
- **Q3 — District Comparison**: Which districts are most underserved or best served when combining all three dimensions?
- **Q4 — Methodological Sensitivity**: How much do district rankings change when the spatial access metric is redefined?

---

## Datasets Used

| Dataset | Source | Role |
|---|---|---|
| IPRESS Health Facilities (MINSA) | MINSA | Facility locations and per-capita availability (Component 1) |
| Emergency Care Production by IPRESS | MINSA | Emergency consultations per district (Component 2) |
| Populated Centers — Centros Poblados | INEI | Origin points for spatial access calculation (Component 3) |
| District Boundaries — DISTRITOS.shp | INEI | Unit of analysis for all district-level aggregations |

All raw files are stored in `data/raw/`. Cleaned outputs are saved to `data/processed/`.

---

## Data Cleaning

**IPRESS facilities**
- Removed facilities with missing or invalid coordinates (outside Peru's geographic bounding box)
- Standardised facility identifiers (`codigo_renaes`) for joining to emergency activity data
- Facilities without a district match after spatial join were excluded

**Emergency care production**
- Aggregated to facility level (sum of `total_atenciones` and `total_atendidos`)
- Districts with zero reported consultations were **retained** as a meaningful signal
- **75.6% of districts report zero emergency consultations** — the most critical data quality finding in the dataset

**Populated centers**
- Removed records with missing or zero coordinates
- Spatial join to district boundaries using point-in-polygon (CRS: EPSG:32718 for distance calculations)

**District boundaries**
- Source CRS: EPSG:4326 (WGS 84) — used for display and mapping
- Reprojected to EPSG:32718 (UTM Zone 18S) for all distance-based metric calculations
- 1 district dropped due to missing population data → **final sample: 1,873 districts**

---

## District-Level Metrics

### Index formula

```
index = (score_availability + score_activity + score_access) / 3
```

All three components are normalised using **percentile rank**, which maps each district's raw value to its position in the national distribution. This approach is robust to Peru's extreme outliers (Lima vs. remote Amazonian districts).

### Components

| Component | Metric | Direction |
|---|---|---|
| 1 — Facility availability | `ipress_per_10k` — IPRESS per 10,000 inhabitants | Higher = better |
| 2 — Emergency activity | `atenciones_per_10k` — emergency consultations per 10,000 inhabitants | Higher = better |
| 3 — Spatial access (baseline) | `dist_mediana_km` — median km from populated centers to nearest active IPRESS | Lower = better (inverted) |
| 3 — Spatial access (alternative) | `pct_ccpp_10km` — % of populated centers within 10 km of nearest active IPRESS | Higher = better |

### Equal weights

Equal weights are used because:
- No strong theoretical basis to weight one pillar over another
- The three components are weakly correlated (PCA PC1 explains ~37% of variance), meaning they capture genuinely different dimensions of access
- Equal-weight composite indices are standard in the literature (e.g., UNDP Human Development Index)

### Two specifications (Q4)

- **Baseline**: Component 3 = `dist_mediana_km` (pure physical proximity)
- **Alternative**: Component 3 = `pct_ccpp_10km` (coverage metric — share of communities within acceptable reach)

Districts are assigned to quintiles 1–5 (Very low → Very high access) for each specification. Comparing the two quintile rankings directly answers Q4.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/aaccasanih-wq/emergency_access_peru.git
cd emergency_access_peru
```

### 2. Create and activate the conda environment

```bash
conda create -n emergency_peru python=3.10
conda activate emergency_peru
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run the Processing Pipeline

Run each module in order from the project root:

```bash
# Step 1 — Load and clean raw data
python src/data_loader.py

# Step 2 — Build geospatial objects and spatial joins
python src/geospatial.py

# Step 3 — Compute district-level metrics and index
python src/metrics.py

# Step 4 & 5 — Generate all static charts, maps, and interactive Folium maps
python src/visualization.py
```

Processed outputs are saved to `data/processed/`.
Charts and maps are saved to `output/figures/`.
District-level tables are saved to `output/tables/`.

---

## How to Run the Streamlit App

```bash
conda activate emergency_peru
streamlit run app.py
```

The app opens at `http://localhost:8501` and contains four tabs:

| Tab | Content |
|---|---|
| Data & Methodology | Problem statement, data sources, cleaning summary, index methodology, limitations |
| Static Analysis | All charts (Q1–Q4) with interpretations and visual justifications |
| Geospatial Results | Static maps and filterable district comparison table |
| Interactive Exploration | Folium maps, multi-district comparison, baseline vs alternative quintile shift table |

---

## Main Findings

- **Severe facility shortage**: The majority of Peru's districts fall below the WHO minimum threshold of 2 IPRESS per 10,000 inhabitants. Small, remote districts in the Amazon basin and southern highlands are the most underserved.

- **Widespread inactivity**: 75.6% of districts report zero emergency consultations in the dataset. This is the starkest indicator of unequal access and likely reflects both genuine absence of activity and data gaps in reporting.

- **Spatial isolation**: The national median distance from populated centers to the nearest active IPRESS is approximately 26 km. Districts in Loreto, Ucayali, and parts of Cusco and Puno show median distances exceeding 100 km.

- **Composite ranking**: When combining all three dimensions, the worst-served districts concentrate in the Amazon lowlands and remote highland provinces. The best-served districts are predominantly urban or peri-urban, with Lima-adjacent districts consistently ranking in the top quintile.

- **Methodological robustness (Q4)**: Switching from `dist_mediana_km` to `pct_ccpp_10km` as Component 3 leaves 73% of districts in the same quintile. Only 10 districts shift by ≥2 quintiles, indicating the composite ranking is broadly stable across specifications. The districts most sensitive to this change tend to have many small, scattered populated centers — where the two metrics measure meaningfully different things.

---

## Limitations

- **Zero-activity districts**: 75.6% of districts report zero emergency consultations. This may reflect data reporting gaps as much as true absence of activity, which deflates Component 2 for a large share of the country.
- **NaN imputation**: Districts with no populated centers or no nearby active IPRESS have `dist_mediana_km` imputed with the national median (26 km) — a conservative assumption that may understate isolation in some cases.
- **Snapshot in time**: The analysis reflects one reporting period; seasonal or annual variation is not captured.
- **Distance definition**: Spatial access is measured to the nearest active IPRESS (any emergency activity reported), not to a facility specifically equipped for the patient's emergency type.
- **Population data**: District population figures may not precisely match the period of the health activity data.

---

## Repository Structure

```
emergency_access_peru/
│
├── app.py                    # Streamlit application (4 tabs)
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
│
├── src/
│   ├── data_loader.py        # Load and clean raw datasets
│   ├── cleaning.py           # Cleaning and preprocessing helpers
│   ├── geospatial.py         # Spatial joins and distance calculations
│   ├── metrics.py            # District-level index construction
│   ├── visualization.py      # Static charts, maps, and Folium maps
│   └── utils.py              # Utility functions
│
├── data/
│   ├── raw/                  # Original downloaded files
│   └── processed/            # Cleaned and analysis-ready outputs
│
├── output/
│   ├── figures/              # Static PNGs and interactive HTML maps
│   └── tables/               # District-level CSV tables
│
└── video/
    └── link.txt              # Link to explanatory video
```
