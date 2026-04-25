"""
data_loader.py
Funciones para cargar los 4 datasets del proyecto.
"""

import pandas as pd
import geopandas as gpd

# ── Rutas a los archivos raw ──────────────────────────────────────────────────
IPRESS_PATH         = "data/raw/IPRESS.csv"
EMERGENCIAS_PATH    = "data/raw/emergencias_2024.csv"
DISTRITOS_PATH      = "data/raw/distritos/DISTRITOS.shp"
CCPP_PATH           = "data/raw/centros_poblados/CCPP_IGN100K.shp"
POBLACION_PATH      = "data/raw/poblacion_distritos_2017.csv"


def load_ipress() -> pd.DataFrame:
    """
    Carga el dataset de establecimientos de salud IPRESS.
    Fuente: SUSALUD - Plataforma Nacional de Datos Abiertos
    Retorna un DataFrame con todos los establecimientos de salud del Perú.
    
    """
    # Encoding latin-1 verificado: utf-8 falla, latin-1/iso-8859-1/cp1252 funcionan
    # Se usa latin-1 por ser el más estándar para archivos del gobierno peruano
    df = pd.read_csv(IPRESS_PATH, encoding="latin-1", low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"[IPRESS] {df.shape[0]:,} filas | {df.shape[1]} columnas")
    return df


def load_emergencias() -> pd.DataFrame:
    """
    Carga el dataset de producción asistencial en emergencia por IPRESS (2024).
    Fuente: SUSALUD - datos.susalud.gob.pe
    Retorna un DataFrame con atenciones de emergencia por establecimiento y mes.
    """
    # Encoding latin-1 verificado: utf-8 falla, latin-1/iso-8859-1/cp1252 funcionan
    # Se usa latin-1 por ser el más estándar para archivos del gobierno peruano
    df = pd.read_csv(EMERGENCIAS_PATH, encoding="latin-1",
                     sep=";", low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"[EMERGENCIAS] {df.shape[0]:,} filas | {df.shape[1]} columnas")
    return df


def load_distritos() -> gpd.GeoDataFrame:
    """
    Carga el shapefile de límites distritales del Perú.
    Fuente: Repositorio del curso - d2cml-ai/Data-Science-Python
    Retorna un GeoDataFrame con los polígonos de los 1,873 distritos.
    CRS original: EPSG:4326 (WGS84)
    """
    gdf = gpd.read_file(DISTRITOS_PATH)
    print(f"[DISTRITOS] {gdf.shape[0]:,} filas | CRS: {gdf.crs}")
    return gdf


def load_centros_poblados() -> gpd.GeoDataFrame:
    """
    Carga el shapefile de centros poblados del Perú.
    Fuente: Instituto Geográfico Nacional (IGN) - datosabiertos.gob.pe
    Retorna un GeoDataFrame con 136,587 centros poblados.
    CRS original: EPSG:4326 (WGS84)
    """
    gdf = gpd.read_file(CCPP_PATH)
    print(f"[CENTROS POBLADOS] {gdf.shape[0]:,} filas | CRS: {gdf.crs}")
    return gdf


def load_poblacion() -> pd.DataFrame:
    """
    Carga el dataset de población por distrito (Censo INEI 2017).
    Fuente: geodir/ubigeo-peru — basado en datos del INEI
    Retorna un DataFrame con población y superficie por distrito.
    """
    df = pd.read_csv(POBLACION_PATH)
    print(f"[POBLACION] {df.shape[0]:,} filas | {df.shape[1]} columnas")
    return df


if __name__ == "__main__":
    print("=== Cargando datasets ===\n")
    ipress       = load_ipress()
    emergencias  = load_emergencias()
    distritos    = load_distritos()
    ccpp         = load_centros_poblados()
    poblacion    = load_poblacion()
    print("\n=== Todos los datasets cargados correctamente ===")