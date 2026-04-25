"""
geospatial.py
Integración espacial de los 4 datasets del proyecto.

Operaciones principales
───────────────────────────────────────────────────────────────────────────────
1. Construir GeoDataFrame puntual de IPRESS (solo establecimientos con
   coordenadas válidas, ~38 % del total).
2. Merge IPRESS + EMERGENCIAS (left join por codigo_unico):
   todos los establecimientos quedan enriquecidos con actividad de emergencia
   cuando existe; NaN si no reportaron.
3. Asignación de IPRESS a distritos por UBIGEO (autoritativo, cubre el 100 %).
4. Asignación de Centros Poblados a distritos por spatial join (punto-en-polígono).
5. Cálculo de distancia de cada Centro Poblado al IPRESS activo más cercano
   (se reproyecta a UTM 18S — EPSG:32718 — para obtener metros reales).
6. Resumen distrital con todos los indicadores base para metrics.py.

CRS
───
- Almacenamiento y joins espaciales : EPSG:4326  (WGS-84, grados)
- Cálculos de distancia             : EPSG:32718 (WGS-84 / UTM zona 18S, metros)
  UTM 18S es la zona que cubre la mayor parte del territorio peruano y es el
  estándar habitual para análisis nacionales de Perú.

Salidas (data/processed/)
────────────────────────
ipress_geo.geojson        : capa puntual de IPRESS con coordenadas
ipress_distrito.csv       : IPRESS + emergencias + ubigeo distrital
ccpp_distrito.geojson     : centros poblados con distrito asignado y distancia
distrito_summary.geojson  : resumen por distrito (base para metrics.py)
───────────────────────────────────────────────────────────────────────────────
"""

import os

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# CRS
CRS_GEO  = "EPSG:4326"    # geográfico — almacenamiento y joins
CRS_PROJ = "EPSG:32718"   # UTM zona 18S — distancias en metros


# ── Carga de datos procesados ─────────────────────────────────────────────────

def load_processed_data(processed_path: str = "data/processed") -> dict:
    """
    Carga los datasets ya limpios desde data/processed/.
    Fuerza ubigeo a string de 6 dígitos en los CSV (pandas los lee como int64).
    """
    ipress      = pd.read_csv(f"{processed_path}/ipress_clean.csv")
    emergencias = pd.read_csv(f"{processed_path}/emergencias_clean.csv")
    poblacion   = pd.read_csv(f"{processed_path}/poblacion_clean.csv")

    # Restaurar formato string de 6 dígitos perdido al guardar como CSV
    ipress["ubigeo"]      = ipress["ubigeo"].astype(str).str.zfill(6)
    emergencias["ubigeo"] = emergencias["ubigeo"].astype(str).str.zfill(6)
    poblacion["ubigeo"]   = poblacion["ubigeo"].astype(str).str.zfill(6)

    return {
        "ipress":      ipress,
        "emergencias": emergencias,
        "distritos":   gpd.read_file(f"{processed_path}/distritos_clean.geojson"),
        "ccpp":        gpd.read_file(f"{processed_path}/ccpp_clean.geojson"),
        "poblacion":   poblacion,
    }


# ── 1. GeoDataFrame puntual de IPRESS ─────────────────────────────────────────

def build_ipress_gdf(ipress_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Crea una capa puntual con los establecimientos que tienen coordenadas válidas.
    Los ~62 % sin coordenadas se excluyen de esta capa pero se conservan en
    ipress_distrito (asignación por UBIGEO).

    Retorna GeoDataFrame en EPSG:4326.
    """
    con_coord = ipress_df[ipress_df["tiene_coordenadas"]].copy()
    geometry  = [Point(lon, lat)
                 for lon, lat in zip(con_coord["longitud"], con_coord["latitud"])]
    gdf = gpd.GeoDataFrame(con_coord, geometry=geometry, crs=CRS_GEO)
    return gdf


# ── 2. Merge IPRESS + EMERGENCIAS ─────────────────────────────────────────────

def merge_ipress_emergencias(
    ipress_df:      pd.DataFrame,
    emergencias_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left join de IPRESS con EMERGENCIAS por codigo_unico.

    - Todos los 20,774 establecimientos quedan en el resultado.
    - Los 1,167 con actividad de emergencia reciben total_atenciones,
      total_atendidos y meses_reporte; el resto queda en NaN → se rellena
      con 0 para los indicadores numéricos.
    - La columna 'tiene_emergencia' indica si el establecimiento reportó
      actividad de emergencia en 2024.
    """
    cols_emerg = [
        "codigo_unico", "sector", "razon_soc",
        "total_atenciones", "total_atendidos", "meses_reporte",
    ]
    merged = ipress_df.merge(
        emergencias_df[cols_emerg],
        on="codigo_unico",
        how="left",
    )

    merged["tiene_emergencia"] = merged["total_atenciones"].notna()
    merged["total_atenciones"] = merged["total_atenciones"].fillna(0).astype(int)
    merged["total_atendidos"]  = merged["total_atendidos"].fillna(0).astype(int)
    merged["meses_reporte"]    = merged["meses_reporte"].fillna(0).astype(int)

    return merged


# ── 3. Asignación de IPRESS a distritos (por UBIGEO) ──────────────────────────

def assign_ipress_to_districts(
    ipress_emerg_df: pd.DataFrame,
    distritos_gdf:   gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Une IPRESS+EMERGENCIAS con los distritos usando UBIGEO como clave.
    Estrategia: UBIGEO es autoritativo (asignado por SUSALUD) y cubre el 100 %
    de los establecimientos, a diferencia del spatial join que solo aplicaría
    al 38 % con coordenadas.

    Los 2 UBIGEOs de IPRESS sin correspondencia en el shapefile distrital
    (<0.01 % del total) quedan con campos distritales en NaN.
    """
    dist_cols = ["ubigeo", "departamen", "provincia", "distrito"]
    df = ipress_emerg_df.merge(
        distritos_gdf[dist_cols].rename(columns={
            "departamen": "dep_distrito",
            "provincia":  "prov_distrito",
            "distrito":   "nom_distrito",
        }),
        on="ubigeo",
        how="left",
    )
    return df


# ── 4. Asignación de Centros Poblados a distritos (spatial join) ───────────────

def assign_ccpp_to_districts(
    ccpp_gdf:      gpd.GeoDataFrame,
    distritos_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Asigna cada centro poblado a su distrito mediante un spatial join
    punto-en-polígono (predicate='within').
    Ambas capas deben estar en EPSG:4326.

    Los centros poblados que no caen dentro de ningún polígono distrital
    (bordes o pequeños gaps de geometría) quedan con ubigeo NaN y se
    excluyen del resumen distrital.
    """
    dist_cols = distritos_gdf[["ubigeo", "departamen", "provincia",
                                "distrito", "geometry"]].copy()

    joined = gpd.sjoin(
        ccpp_gdf,
        dist_cols,
        how="left",
        predicate="within",
    ).drop(columns="index_right")

    n_sin = joined["ubigeo"].isna().sum()
    if n_sin > 0:
        print(f"  [CCPP] {n_sin:,} puntos sin distrito asignado (bordes/gaps)")

    return joined


# ── 5. Distancia de cada CCPP al IPRESS activo más cercano ────────────────────

def compute_nearest_facility_distance(
    ccpp_gdf:        gpd.GeoDataFrame,
    ipress_emerg_df: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """
    Para cada centro poblado calcula la distancia en km al IPRESS con actividad
    de emergencia más cercano.

    Se usan únicamente los establecimientos con emergencia reportada y con
    coordenadas válidas, ya que el objetivo es medir acceso a atención real.
    Se reproyecta a EPSG:32718 (UTM 18S) para obtener distancias en metros.

    Retorna ccpp_gdf con columna adicional 'dist_ipress_km'.
    """
    ipress_activo = ipress_emerg_df[
        ipress_emerg_df["tiene_emergencia"] &
        ipress_emerg_df["tiene_coordenadas"]
    ].copy()

    if len(ipress_activo) == 0:
        ccpp_gdf = ccpp_gdf.copy()
        ccpp_gdf["dist_ipress_km"] = np.nan
        return ccpp_gdf

    geom_ipress = [Point(lon, lat)
                   for lon, lat in zip(ipress_activo["longitud"],
                                       ipress_activo["latitud"])]
    ipress_gdf  = gpd.GeoDataFrame(ipress_activo, geometry=geom_ipress,
                                   crs=CRS_GEO).to_crs(CRS_PROJ)

    ccpp_proj = ccpp_gdf.to_crs(CRS_PROJ).copy()

    nearest = gpd.sjoin_nearest(
        ccpp_proj[["geometry"]],
        ipress_gdf[["geometry"]],
        how="left",
        distance_col="dist_m",
    )

    ccpp_gdf = ccpp_gdf.copy()
    ccpp_gdf["dist_ipress_km"] = (nearest["dist_m"].values / 1000).round(3)

    return ccpp_gdf


# ── 6. Resumen distrital ──────────────────────────────────────────────────────

def build_district_summary(
    ipress_distrito_df: pd.DataFrame,
    ccpp_distrito_gdf:  gpd.GeoDataFrame,
    distritos_gdf:      gpd.GeoDataFrame,
    poblacion_df:       pd.DataFrame = None,
) -> gpd.GeoDataFrame:
    """
    Construye un GeoDataFrame a nivel distrital con todos los indicadores base
    necesarios para metrics.py.

    Indicadores incluidos
    ─────────────────────
    n_establecimientos  : total IPRESS en funcionamiento en el distrito
    n_con_emergencia    : IPRESS con actividad de emergencia reportada en 2024
    total_atenciones    : suma anual de atenciones de emergencia
    total_atendidos     : suma anual de pacientes atendidos
    n_ccpp              : número de centros poblados en el distrito
    dist_media_km       : distancia media (CCPP → IPRESS activo más cercano)
    dist_mediana_km     : distancia mediana (CCPP → IPRESS activo más cercano)
    pct_ccpp_10km       : % de CCPP del distrito a ≤10 km del IPRESS activo más
                          cercano — métrica de acceso espacial para especificación
                          alternativa del índice (baseline usa dist_mediana_km)
    """
    # Agregados de IPRESS por distrito
    agg_ipress = (
        ipress_distrito_df
        .groupby("ubigeo", as_index=False)
        .agg(
            n_establecimientos=("codigo_unico",     "count"),
            n_con_emergencia  =("tiene_emergencia", "sum"),
            total_atenciones  =("total_atenciones", "sum"),
            total_atendidos   =("total_atendidos",  "sum"),
        )
    )
    agg_ipress["n_con_emergencia"] = agg_ipress["n_con_emergencia"].astype(int)

    # Agregados de CCPP por distrito
    ccpp_con_dist = ccpp_distrito_gdf.dropna(subset=["ubigeo"])

    # Indicador de acceso: CCPP a ≤10 km del IPRESS activo más cercano
    ccpp_con_dist = ccpp_con_dist.copy()
    ccpp_con_dist["dentro_10km"] = (ccpp_con_dist["dist_ipress_km"] <= 10).astype(int)

    agg_ccpp = (
        ccpp_con_dist
        .groupby("ubigeo", as_index=False)
        .agg(
            n_ccpp          =("nom_poblad",    "count"),
            dist_media_km   =("dist_ipress_km", "mean"),
            dist_mediana_km =("dist_ipress_km", "median"),
            n_ccpp_10km     =("dentro_10km",   "sum"),
        )
    )
    agg_ccpp["dist_media_km"]   = agg_ccpp["dist_media_km"].round(3)
    agg_ccpp["dist_mediana_km"] = agg_ccpp["dist_mediana_km"].round(3)
    agg_ccpp["pct_ccpp_10km"]   = (
        agg_ccpp["n_ccpp_10km"] / agg_ccpp["n_ccpp"] * 100
    ).round(2)
    agg_ccpp = agg_ccpp.drop(columns="n_ccpp_10km")

    # Unir al shapefile distrital (todos los 1,873 distritos)
    summary = distritos_gdf.merge(agg_ipress, on="ubigeo", how="left")
    summary = summary.merge(agg_ccpp,  on="ubigeo", how="left")

    # Rellenar distritos sin registros con 0
    for col in ["n_establecimientos", "n_con_emergencia",
                "total_atenciones", "total_atendidos", "n_ccpp"]:
        summary[col] = summary[col].fillna(0).astype(int)

    # Merge población censal si se provee
    if poblacion_df is not None:
        summary = summary.merge(
            poblacion_df[["ubigeo", "poblacion", "superficie_km2"]],
            on="ubigeo",
            how="left",
        )

    return summary


# ── Pipeline principal ────────────────────────────────────────────────────────

def run_geospatial_pipeline(
    processed_path: str = "data/processed",
    output_path:    str = "data/processed",
) -> dict:
    """
    Ejecuta la integración geoespacial completa y guarda los resultados.

    Salidas
    ───────
    data/processed/ipress_geo.geojson       capa puntual IPRESS
    data/processed/ipress_distrito.csv      IPRESS + emergencias + distrito
    data/processed/ccpp_distrito.geojson    CCPP + distrito + distancia
    data/processed/distrito_summary.geojson resumen distrital
    """
    os.makedirs(output_path, exist_ok=True)

    print("Cargando datos procesados...")
    data           = load_processed_data(processed_path)
    ipress_df      = data["ipress"]
    emergencias_df = data["emergencias"]
    distritos_gdf  = data["distritos"]
    ccpp_gdf       = data["ccpp"]
    poblacion_df   = data["poblacion"]

    print("Construyendo GeoDataFrame puntual de IPRESS...")
    ipress_geo = build_ipress_gdf(ipress_df)

    print("Mergeando IPRESS con EMERGENCIAS...")
    ipress_emerg = merge_ipress_emergencias(ipress_df, emergencias_df)

    print("Asignando establecimientos a distritos (UBIGEO)...")
    ipress_distrito = assign_ipress_to_districts(ipress_emerg, distritos_gdf)

    print("Asignando centros poblados a distritos (spatial join)...")
    ccpp_distrito = assign_ccpp_to_districts(ccpp_gdf, distritos_gdf)

    print("Calculando distancia CCPP → IPRESS activo más cercano...")
    ccpp_distrito = compute_nearest_facility_distance(ccpp_distrito, ipress_emerg)

    print("Construyendo resumen distrital...")
    distrito_summary = build_district_summary(
        ipress_distrito, ccpp_distrito, distritos_gdf, poblacion_df
    )

    print("Guardando outputs...")
    ipress_geo.to_file(
        f"{output_path}/ipress_geo.geojson", driver="GeoJSON")
    ipress_distrito.to_csv(
        f"{output_path}/ipress_distrito.csv", index=False)
    ccpp_distrito.to_file(
        f"{output_path}/ccpp_distrito.geojson", driver="GeoJSON")
    distrito_summary.to_file(
        f"{output_path}/distrito_summary.geojson", driver="GeoJSON")

    print(f"\n{'─'*55}")
    print(f"IPRESS puntual          : {len(ipress_geo):>7,} establecimientos")
    print(f"IPRESS + emergencias    : {len(ipress_distrito):>7,} establecimientos")
    print(f"  → con actividad emerg : {ipress_distrito['tiene_emergencia'].sum():>7,}")
    print(f"CCPP con distrito       : {ccpp_distrito['ubigeo'].notna().sum():>7,}")
    print(f"Distritos en resumen    : {len(distrito_summary):>7,}")
    print(f"  → con ≥1 IPRESS       : {(distrito_summary['n_establecimientos'] > 0).sum():>7,}")
    print(f"  → con actividad emerg : {(distrito_summary['n_con_emergencia'] > 0).sum():>7,}")
    print(f"Outputs guardados en → {output_path}/")

    return {
        "ipress_geo":       ipress_geo,
        "ipress_distrito":  ipress_distrito,
        "ccpp_distrito":    ccpp_distrito,
        "distrito_summary": distrito_summary,
    }


if __name__ == "__main__":
    run_geospatial_pipeline()
