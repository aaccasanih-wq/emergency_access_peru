"""
cleaning.py
Limpieza y estandarización de los 4 datasets del proyecto.

Decisiones de limpieza documentadas:
──────────────────────────────────────────────────────────────────────────────
IPRESS
  · Solo establecimientos EN FUNCIONAMIENTO (se descartan 19 inactivos).
  · UBIGEO estandarizado a string de 6 dígitos con zfill.
  · Columnas NORTE/ESTE renombradas a longitud/latitud: a pesar del nombre,
    NORTE contiene longitud (rango −81 a −68) y ESTE contiene latitud
    (rango −18 a 0), en línea con las coordenadas geográficas de Perú.
  · Valores 0.0 en coordenadas reemplazados por NaN (entradas inválidas).
  · Se añade bandera 'tiene_coordenadas' para uso en la capa puntual.
  · Los 61.8 % de establecimientos sin coordenadas se conservan: el UBIGEO
    permite su asignación distrital sin necesidad de lat/lon.
  · 26 duplicados por codigo_unico eliminados.
  · Se retienen 13 columnas de 33 originales.

EMERGENCIAS
  · UBIGEO estandarizado a string de 6 dígitos con zfill.
  · 36,377 filas (14.55 %) con códigos NE_0001/NE_0002 eliminadas antes de
    agregar. Convertirlas a 0 haría que 3,126 de 4,293 establecimientos
    (72.8 %) aparecieran con cero atenciones, distorsionando el índice.
  · 'co_ipress' renombrado a 'codigo_unico' para unificar la clave de merge.
  · Agregación anual por establecimiento (de 213,623 a 1,167 filas).

DISTRITOS
  · UBIGEO creado desde 'iddist' con zfill(6).
  · CRS forzado a EPSG:4326 (WGS-84).
  · Geometrías inválidas corregidas con buffer(0).
  · Se retienen 8 columnas de 12 originales.

CENTROS POBLADOS
  · Geometrías inválidas/vacías descartadas (ninguna encontrada).
  · CRS forzado a EPSG:4326 (WGS-84).
  · Puntos fuera del rango geográfico de Perú eliminados (ninguno encontrado).
    Límites usados: lat [−18.35, 0.00] | lon [−81.33, −68.65].
  · Se retienen 8 columnas de 14 originales.
──────────────────────────────────────────────────────────────────────────────
"""

import os
import unicodedata

import numpy as np
import pandas as pd
import geopandas as gpd

from src.data_loader import (
    load_ipress,
    load_emergencias,
    load_distritos,
    load_centros_poblados,
    load_poblacion,
)

# Límites geográficos de Perú (WGS-84)
PERU_LAT = (-18.35,  0.00)
PERU_LON = (-81.33, -68.65)


# ── Utilidades ────────────────────────────────────────────────────────────────

def _limpiar_columnas(df):
    """Estandariza nombres de columnas: minúsculas, sin tildes, guiones bajos."""
    def _quitar_tildes(s):
        return "".join(
            c for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn"
        )
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .map(_quitar_tildes)
        .str.replace(" ", "_", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace(".", "",  regex=False)
        .str.replace("(", "",  regex=False)
        .str.replace(")", "",  regex=False)
    )
    return df


# ── IPRESS ────────────────────────────────────────────────────────────────────

def clean_ipress(ipress: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el dataset de establecimientos de salud IPRESS.

    Columnas retenidas
    ------------------
    codigo_unico, nombre_del_establecimiento, clasificacion, categoria,
    departamento, provincia, distrito, ubigeo, condicion, institucion,
    longitud, latitud, tiene_coordenadas
    """
    df = _limpiar_columnas(ipress.copy())

    # Solo establecimientos operativos
    df = df[df["condicion"] == "EN FUNCIONAMIENTO"].copy()

    # UBIGEO a string de 6 dígitos
    df["ubigeo"] = df["ubigeo"].astype(str).str.zfill(6)

    # Selección de columnas (incluye norte/este para análisis espacial puntual)
    cols = [
        "codigo_unico", "nombre_del_establecimiento", "clasificacion",
        "categoria", "departamento", "provincia", "distrito", "ubigeo",
        "condicion", "institucion",
        "norte",   # contiene longitud — ver nota en módulo docstring
        "este",    # contiene latitud  — ver nota en módulo docstring
    ]
    df = df[cols].copy()

    # Renombrar coordenadas para reflejar su contenido real
    df = df.rename(columns={"norte": "longitud", "este": "latitud"})

    # Valores 0.0 no son coordenadas válidas en Perú → NaN
    df["longitud"] = df["longitud"].replace(0.0, np.nan)
    df["latitud"]  = df["latitud"].replace(0.0, np.nan)

    # Bandera de coordenada válida dentro de los límites de Perú
    lon_ok = df["longitud"].notna() & df["longitud"].between(*PERU_LON)
    lat_ok = df["latitud"].notna()  & df["latitud"].between(*PERU_LAT)
    df["tiene_coordenadas"] = lon_ok & lat_ok

    # Eliminar duplicados por código único
    df = df.drop_duplicates(subset="codigo_unico", keep="first").copy()

    return df


# ── EMERGENCIAS ───────────────────────────────────────────────────────────────

def clean_emergencias(emergencias: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el dataset de producción asistencial en emergencia 2024.

    Columnas retenidas
    ------------------
    codigo_unico, ubigeo, departamento, provincia, distrito, sector,
    categoria, razon_soc, total_atenciones, total_atendidos, meses_reporte
    """
    df = _limpiar_columnas(emergencias.copy())

    # UBIGEO a string de 6 dígitos
    df["ubigeo"] = df["ubigeo"].astype(str).str.zfill(6)

    # Eliminar filas con códigos NE (No Especificado) antes de agregar.
    # Convertirlas a 0 produciría 3,126 establecimientos con cero atenciones
    # artificiales, distorsionando cualquier índice de actividad.
    ne_mask = df["nro_total_atenciones"].astype(str).str.startswith("NE")
    df = df[~ne_mask].copy()

    # Conversión segura a entero (sin NE, no quedan no-numéricos)
    df["nro_total_atenciones"] = df["nro_total_atenciones"].astype(int)
    df["nro_total_atendidos"]  = df["nro_total_atendidos"].astype(int)

    # Unificar nombre de clave de merge con IPRESS
    df = df.rename(columns={"co_ipress": "codigo_unico"})

    # Agregación anual por establecimiento
    df = (
        df.groupby(
            ["codigo_unico", "ubigeo", "departamento", "provincia",
             "distrito", "sector", "categoria", "razon_soc"],
            as_index=False,
        )
        .agg(
            total_atenciones=("nro_total_atenciones", "sum"),
            total_atendidos =("nro_total_atendidos",  "sum"),
            meses_reporte   =("mes",                  "nunique"),
        )
    )

    return df


# ── DISTRITOS ─────────────────────────────────────────────────────────────────

def clean_distritos(distritos: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Limpia el shapefile de límites distritales.

    Columnas retenidas
    ------------------
    ubigeo, iddpto, departamen, idprov, provincia, iddist, distrito, geometry
    """
    gdf = _limpiar_columnas(distritos.copy())

    # UBIGEO desde iddist
    gdf["ubigeo"] = gdf["iddist"].astype(str).str.zfill(6)

    # CRS estándar WGS-84
    gdf = gdf.to_crs(epsg=4326)

    # Corregir geometrías inválidas
    if not gdf.geometry.is_valid.all():
        gdf["geometry"] = gdf["geometry"].buffer(0)

    cols = ["ubigeo", "iddpto", "departamen", "idprov", "provincia",
            "iddist", "distrito", "geometry"]
    return gdf[cols].copy()


# ── CENTROS POBLADOS ──────────────────────────────────────────────────────────

def clean_ccpp(ccpp: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Limpia el shapefile de centros poblados.

    Columnas retenidas
    ------------------
    nom_poblad, codigo, cat_poblad, dist, prov, dep, categoria, geometry
    """
    gdf = _limpiar_columnas(ccpp.copy())

    # Descartar geometrías inválidas o vacías
    gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty].copy()

    # CRS estándar WGS-84
    gdf = gdf.to_crs(epsg=4326)

    # Descartar puntos fuera de los límites geográficos de Perú
    gx = gdf.geometry.x
    gy = gdf.geometry.y
    in_bounds = (
        gx.between(*PERU_LON) &
        gy.between(*PERU_LAT)
    )
    gdf = gdf[in_bounds].copy()

    cols = ["nom_poblad", "codigo", "cat_poblad", "dist", "prov", "dep",
            "categoria", "geometry"]
    return gdf[cols].copy()


# ── POBLACIÓN DISTRITAL ───────────────────────────────────────────────────────

def clean_poblacion(poblacion: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el dataset de población distrital (Censo INEI 2017).

    Columnas retenidas
    ------------------
    ubigeo, poblacion, superficie_km2
    """
    df = poblacion.copy()

    # UBIGEO a string de 6 dígitos
    df["ubigeo"] = df["Ubigeo"].astype(str).str.zfill(6)

    # Población: eliminar comas y convertir a entero
    df["poblacion"] = df["Poblacion"].astype(str).str.replace(",", "", regex=False).astype(int)

    # Superficie en km²
    df["superficie_km2"] = df["Superficie"].astype(str).str.replace(",", "", regex=False).astype(float).round(2)

    cols = ["ubigeo", "poblacion", "superficie_km2"]
    return df[cols].copy()


# ── Pipeline principal ────────────────────────────────────────────────────────

def run_cleaning_pipeline(output_path: str = "data/processed") -> dict:
    """
    Ejecuta la limpieza completa y guarda los datasets procesados.

    Salidas
    -------
    data/processed/ipress_clean.csv
    data/processed/emergencias_clean.csv
    data/processed/distritos_clean.geojson
    data/processed/ccpp_clean.geojson
    """
    os.makedirs(output_path, exist_ok=True)

    print("Cargando datasets...")
    ipress      = load_ipress()
    emergencias = load_emergencias()
    distritos   = load_distritos()
    ccpp        = load_centros_poblados()
    poblacion   = load_poblacion()

    print("Limpiando datasets...")
    ipress_clean      = clean_ipress(ipress)
    emergencias_clean = clean_emergencias(emergencias)
    distritos_clean   = clean_distritos(distritos)
    ccpp_clean        = clean_ccpp(ccpp)
    poblacion_clean   = clean_poblacion(poblacion)

    print("Guardando outputs...")
    ipress_clean.to_csv(
        f"{output_path}/ipress_clean.csv", index=False)
    emergencias_clean.to_csv(
        f"{output_path}/emergencias_clean.csv", index=False)
    distritos_clean.to_file(
        f"{output_path}/distritos_clean.geojson", driver="GeoJSON")
    ccpp_clean.to_file(
        f"{output_path}/ccpp_clean.geojson", driver="GeoJSON")
    poblacion_clean.to_csv(
        f"{output_path}/poblacion_clean.csv", index=False)

    print(f"\n{'─'*50}")
    print(f"IPRESS      : {len(ipress_clean):>7,} establecimientos")
    print(f"EMERGENCIAS : {len(emergencias_clean):>7,} establecimientos con actividad")
    print(f"DISTRITOS   : {len(distritos_clean):>7,} distritos")
    print(f"CCPP        : {len(ccpp_clean):>7,} centros poblados")
    print(f"POBLACIÓN   : {len(poblacion_clean):>7,} distritos con datos censales")
    print(f"Outputs guardados en → {output_path}/")

    return {
        "ipress":      ipress_clean,
        "emergencias": emergencias_clean,
        "distritos":   distritos_clean,
        "ccpp":        ccpp_clean,
        "poblacion":   poblacion_clean,
    }


if __name__ == "__main__":
    run_cleaning_pipeline()
