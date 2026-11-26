"""
Lógica del producto Ganado Protegido (Hackathon IA PUCP 2026).
Implementa los umbrales con fuente (Senamhi-FAO, Fosberg, Nixon-Smith / FAO-56,
ETCCDI y Pulgar Vidal) separados de la interfaz visual en Streamlit.
"""
import os
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

APP = Path(__file__).resolve().parent
RAIZ = APP.parent
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from gfs_forecast import LATS as GFS_LATS, LONS as GFS_LONS, get_gfs_forecast, make_windows

# Sectores ganaderos altoandinos representativos dentro del dominio WRF (70x35)
SECTORES_PUNA = {
    "Cátac / Pastoruri (Recuay)": (-9.80, -77.45),
    "Olleros / Quebrada Honda (Huaraz)": (-9.55, -77.48),
    "Carhuaz / Cordillera Blanca": (-9.28, -77.55),
    "Yungay / Llanganuco": (-9.12, -77.62),
    "Chavín / Conchucos Sur": (-9.58, -77.18),
}


# ---------------------------------------------------------------------------
# 1. ARQUITECTURA Y CARGA ÚNICA DEL MODELO
# ---------------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self, w=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(14, w, 3, padding=1), nn.GELU(),
            nn.Conv2d(w, w, 3, padding=1), nn.GELU(),
        )
        self.static = nn.Parameter(torch.zeros(1, 8, 70, 35))
        self.dec = nn.Sequential(
            nn.Conv2d(w + 9, w, 3, padding=1), nn.GELU(),
            nn.Conv2d(w, w, 3, padding=1), nn.GELU(),
            nn.Conv2d(w, 4, 1),
        )

    def forward(self, x, h, grid, dem):
        B = x.shape[0]
        x = x.permute(0, 1, 4, 2, 3).reshape(B, 12, 17, 13)
        ang = 2 * np.pi * h / 24
        hh = torch.stack([torch.sin(ang), torch.cos(ang)], 1)[..., None, None].expand(B, 2, 17, 13)
        f = self.enc(torch.cat([x, hh], 1))
        f = F.grid_sample(f, grid.expand(B, -1, -1, -1), align_corners=True)
        f = torch.cat([f, dem.expand(B, -1, -1, -1), self.static.expand(B, -1, -1, -1)], 1)
        return self.dec(f).permute(0, 2, 3, 1)


def cargar_recursos(ruta_static=None, ruta_modelo=None, ruta_stats=None):
    """Carga una sola vez pesos, estadísticas de normalización y grilla estática."""
    ruta_static = Path(ruta_static or (RAIZ / "data" / "static.npz"))
    ruta_modelo = Path(ruta_modelo or (RAIZ / "models" / "modelo.pt"))
    ruta_stats = Path(ruta_stats or (RAIZ / "models" / "stats.npz"))
    ruta_chill = RAIZ / "models" / "chill_percentiles.npz"

    if not ruta_static.exists():
        raise FileNotFoundError(f"No se encontró {ruta_static}. Descarga la data de Kaggle en data/.")
    if not ruta_modelo.exists():
        raise FileNotFoundError(f"No se encontró {ruta_modelo}.")
    if not ruta_stats.exists():
        raise FileNotFoundError(f"No se encontró {ruta_stats}.")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    static = dict(np.load(ruta_static))
    stats = np.load(ruta_stats)

    glat = static.get("gfs_lats", GFS_LATS)
    glon = static.get("gfs_lons", GFS_LONS)
    wlat, wlon = static["wrf_lats"], static["wrf_lons"]
    LAT, LON = np.meshgrid(wlat, wlon, indexing="ij")
    gy = (LAT - glat[0]) / (glat[-1] - glat[0]) * 2 - 1
    gx = (LON - glon[0]) / (glon[-1] - glon[0]) * 2 - 1
    grid_t = torch.tensor(np.stack([gx, gy], -1)[None], dtype=torch.float32, device=dev)

    dem = static["dem"].astype(np.float32)
    dem_t = torch.tensor((dem - dem.mean()) / dem.std(), dtype=torch.float32, device=dev)[None, None]

    modelo = Net().to(dev)
    modelo.load_state_dict(torch.load(ruta_modelo, map_location=dev, weights_only=True))
    modelo.eval()

    chill_pct = dict(np.load(ruta_chill)) if ruta_chill.exists() else None

    return {
        "dev": dev,
        "static": static,
        "mu_x": stats["mu_x"],
        "sd_x": stats["sd_x"],
        "clim_t": torch.tensor(stats["clim_all"], dtype=torch.float32, device=dev),
        "asd_t": torch.tensor(stats["anom_sd"], dtype=torch.float32, device=dev),
        "grid_t": grid_t,
        "dem_t": dem_t,
        "modelo": modelo,
        "chill_pct": chill_pct,
    }


def predecir(Xw, times, recursos):
    """Ejecuta el downscaling a 3 km usando los recursos ya cargados en memoria."""
    dev = recursos["dev"]
    Xn = torch.tensor((Xw - recursos["mu_x"]) / recursos["sd_x"], dtype=torch.float32, device=dev)
    horas = torch.tensor([int(str(t)[-2:]) for t in times], dtype=torch.long, device=dev)

    with torch.no_grad():
        pred_norm = recursos["modelo"](Xn, horas.float(), recursos["grid_t"], recursos["dem_t"])
        Y = (recursos["clim_t"][horas] + pred_norm * recursos["asd_t"]).cpu().numpy().astype(np.float32)
    return Y


# ---------------------------------------------------------------------------
# 2. DESCARGA GFS + PRECIPITACIÓN AUXILIAR (OPEN-METEO) CON CACHÉ
# ---------------------------------------------------------------------------
def obtener_precipitacion_sectores(forecast_hours=168, past_days=15, cache_max_age_h=3.0):
    """
    Descarga precipitación horaria GFS de Open-Meteo solo para los 5 sectores ganaderos
    (5 llamadas en vez de 221) para calcular lluvia 24 h previas y días secos (< 1 mm).
    Devuelve dict: {sector_nombre: pd.Series(precip_mm, index=DatetimeIndex_UTC)}.
    """
    cache_file = APP / "precip_sectores_cache.npz"
    backup_file = APP / "precip_sectores_respaldo.npz"

    if cache_file.exists():
        age_h = (time.time() - cache_file.stat().st_mtime) / 3600.0
        if age_h < cache_max_age_h:
            d = np.load(cache_file, allow_pickle=True)
            idx = pd.to_datetime(d["times_utc"])
            return {k: pd.Series(d[k], index=idx) for k in SECTORES_PUNA if k in d}

    nombres = list(SECTORES_PUNA.keys())
    lats = [str(SECTORES_PUNA[n][0]) for n in nombres]
    lons = [str(SECTORES_PUNA[n][1]) for n in nombres]

    params = {
        "latitude": ",".join(lats),
        "longitude": ",".join(lons),
        "hourly": "precipitation",
        "models": "gfs_global",
        "past_days": past_days,
        "forecast_hours": forecast_hours,
        "timezone": "GMT",
    }
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
        if not isinstance(js, list):
            js = [js]
        times_utc = pd.to_datetime(js[0]["hourly"]["time"])
        payload = {"times_utc": times_utc.astype(str).to_numpy()}
        resultado = {}
        for nombre, item in zip(nombres, js):
            arr = np.nan_to_num(np.array(item["hourly"]["precipitation"], dtype=np.float32), nan=0.0)
            payload[nombre] = arr
            resultado[nombre] = pd.Series(arr, index=times_utc)
        np.savez_compressed(cache_file, **payload)
        if not backup_file.exists():
            shutil.copy2(cache_file, backup_file)
        return resultado
    except Exception:
        for alt in (cache_file, backup_file):
            if alt.exists():
                d = np.load(alt, allow_pickle=True)
                idx = pd.to_datetime(d["times_utc"])
                return {k: pd.Series(d[k], index=idx) for k in SECTORES_PUNA if k in d}
        return {}


def obtener_gfs_seguro(forecast_hours=168, start_date=None, end_date=None):
    """
    Descarga GFS desde Open-Meteo (pronóstico de hasta 7 días o fechas pasadas exactas).
    Incluye 2 horas previas en modo histórico para construir la ventana [t-2, t-1, t].
    """
    cache_path = str(APP / "gfs_cache.npz")
    backup_path = str(APP / "gfs_respaldo_demo.npz")

    try:
        if start_date:
            dt_ini = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=1)
            X_raw, times_raw = get_gfs_forecast(
                start_date=dt_ini.strftime("%Y-%m-%d"),
                end_date=end_date or start_date,
            )
            # Filtrar desde las 22:00 UTC del día previo (para tener t-2, t-1 al iniciar el día en Lima)
            inicio_utc = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(hours=3)).strftime("%Y%m%d%H")
            fin_utc = (datetime.strptime(end_date or start_date, "%Y-%m-%d") + timedelta(days=1, hours=5)).strftime("%Y%m%d%H")
            mask = (times_raw >= inicio_utc) & (times_raw <= fin_utc)
            if mask.sum() >= 3:
                X, times = X_raw[mask], times_raw[mask]
            else:
                X, times = X_raw, times_raw
            fuente = f"Histórico Open-Meteo ({start_date} a {end_date or start_date})"
        else:
            # Si cambia el horizonte pedido y el caché tiene otro tamaño, refrescar
            if os.path.exists(cache_path):
                d_chk = np.load(cache_path)
                if abs(len(d_chk["times"]) - (forecast_hours + 2)) > 4:
                    os.remove(cache_path)
            X, times = get_gfs_forecast(forecast_hours=forecast_hours, cache_path=cache_path)
            if os.path.exists(cache_path) and not os.path.exists(backup_path):
                shutil.copy2(cache_path, backup_path)
            fuente = f"Open-Meteo en vivo / caché ({forecast_hours} h)"
        return X, times, fuente
    except Exception as err:
        for ruta_alt in (cache_path, backup_path):
            if os.path.exists(ruta_alt):
                d = np.load(ruta_alt)
                return d["X"], d["times"], f"Respaldo local ({Path(ruta_alt).name}) tras error de API"
        raise RuntimeError(f"Error en Open-Meteo y sin caché de respaldo disponible: {err}")


# ---------------------------------------------------------------------------
# 3. FILTRO DE PUNA, UNIDADES Y VENTANAS HORARIAS (LIMA = UTC - 5)
# ---------------------------------------------------------------------------
def filtrar_puna(dem, alt_min=3800.0, alt_max=4800.0):
    """
    Máscara booleana (70, 35) para la zona altoandina.
    Referencia: Pulgar Vidal (2014) — Suni 3500-4000 m, Puna 4000-4800 m.
    Nota: el DEM es la altura media de cada celda de 3 km (SRTM promediado).
    """
    return (dem >= alt_min) & (dem <= alt_max)


def a_datetime_utc(t_utc):
    return datetime.strptime(str(t_utc), "%Y%m%d%H")


def a_datetime_lima(t_utc):
    return a_datetime_utc(t_utc) - timedelta(hours=5)


def a_hora_lima(t_utc):
    return a_datetime_lima(t_utc).strftime("%d/%m %H:00")


def ubicar_celda_sector(sector_nombre, static, alt_min=3800.0, alt_max=4800.0):
    """Ubica la celda (i, j) más cercana al sector estrictamente dentro de [alt_min, alt_max]."""
    lat_obj, lon_obj = SECTORES_PUNA[sector_nombre]
    LAT, LON = np.meshgrid(static["wrf_lats"], static["wrf_lons"], indexing="ij")
    dist2 = (LAT - lat_obj) ** 2 + (LON - lon_obj) ** 2
    mascara = filtrar_puna(static["dem"], alt_min=alt_min, alt_max=alt_max)
    if mascara.any():
        dist2 = np.where(mascara, dist2, np.inf)
    i, j = np.unravel_index(np.argmin(dist2), dist2.shape)
    return int(i), int(j)


def clasificar_severidad_helada(tmin_c):
    """Clasificación oficial de severidad de helada (Da Motta 1961, en Senamhi-FAO 2010)."""
    if tmin_c > 0.0:
        return "Sin helada"
    if -2.0 < tmin_c <= 0.0:
        return "Helada suave (-2 a 0 °C)"
    if -4.0 < tmin_c <= -2.0:
        return "Helada moderada (-4 a -2 °C)"
    if -6.0 < tmin_c <= -4.0:
        return "Helada fuerte (-6 a -4 °C)"
    if -8.0 < tmin_c <= -6.0:
        return "Helada muy fuerte (-8 a -6 °C)"
    if -10.0 < tmin_c <= -8.0:
        return "Helada severa (-10 a -8 °C)"
    return "Helada muy severa (≤ -10 °C)"


def calcular_indice_enfriamiento(temp_c, viento_10m_ms, lluvia_24h_mm=0.0):
    """
    Índice de enfriamiento de Nixon-Smith (McCaskill et al., 2010; Weeks et al., 2015)
    con ajuste logarítmico de viento a 0,4 m según FAO-56 (Allen et al., 1998):
      v_0.4 = 0.47 * v_10
      C = (11.7 + 3.1 * sqrt(v_0.4)) * (40 - T) + 481.0 + 418.0 * (1 - exp(-0.04 * R))
    Devuelve C en kJ/(m²·h).
    """
    v_04 = np.maximum(0.0, 0.47 * np.asarray(viento_10m_ms, dtype=np.float32))
    t_arr = np.asarray(temp_c, dtype=np.float32)
    r_arr = np.maximum(0.0, np.asarray(lluvia_24h_mm, dtype=np.float32))
    return (11.7 + 3.1 * np.sqrt(v_04)) * (40.0 - t_arr) + 481.0 + 418.0 * (1.0 - np.exp(-0.04 * r_arr))


def analizar_heladas(Y, times, static, alt_min=3800.0, alt_max=4800.0, umbral_c=0.0):
    """Calcula temperatura mínima en el periodo y zonas con helada dentro de la puna."""
    T_c = Y[..., 1] - 273.15
    tmin = T_c.min(axis=0)
    hora_min = np.asarray(times)[T_c.argmin(axis=0)]
    helada = tmin <= umbral_c

    mascara = filtrar_puna(static["dem"], alt_min=alt_min, alt_max=alt_max)
    if mascara.any():
        tmin_puna = float(tmin[mascara].min())
        pct_helada_puna = float(100.0 * helada[mascara].mean())
        idx_rel = np.argmin(np.where(mascara, tmin, np.inf))
        i, j = np.unravel_index(idx_rel, tmin.shape)
    else:
        tmin_puna = float(tmin.min())
        pct_helada_puna = float(100.0 * helada.mean())
        i, j = np.unravel_index(tmin.argmin(), tmin.shape)

    return {
        "tmin": tmin,
        "helada": helada,
        "mascara_puna": mascara,
        "tmin_dominio": float(tmin.min()),
        "tmin_puna": tmin_puna,
        "pct_helada_puna": pct_helada_puna,
        "punto_frio": {
            "lat": float(static["wrf_lats"][i]),
            "lon": float(static["wrf_lons"][j]),
            "alt": float(static["dem"][i, j]),
            "temp": float(tmin[i, j]),
            "hora_lima": a_hora_lima(hora_min[i, j]),
        },
        "periodo_lima": f"{a_hora_lima(times[0])} a {a_hora_lima(times[-1])} (hora Lima)",
    }


# ---------------------------------------------------------------------------
# 4. DECISIÓN 1: COBERTIZO ESTA NOCHE (18:00 - 07:00 LIMA / 23 - 12 UTC)
# ---------------------------------------------------------------------------
def _indices_proxima_noche(times):
    """
    Selecciona únicamente las horas de la PRÓXIMA noche (18:00 a 07:00 Lima, es decir 23 a 12 UTC),
    deteniéndose al terminar ese amanecer (07:00 Lima) para no mezclar noches futuras.
    """
    dts_lima = [a_datetime_lima(t) for t in times]
    idx_noche = []
    en_noche = False
    for k, dt in enumerate(dts_lima):
        es_hora_nocturna = (dt.hour >= 18) or (dt.hour <= 7)
        if es_hora_nocturna:
            en_noche = True
            idx_noche.append(k)
            if dt.hour == 7 and len(idx_noche) > 1:
                break
        elif en_noche:
            break
    return idx_noche if idx_noche else list(range(min(14, len(times))))


def evaluar_cobertizo_sector(
    Y,
    times,
    static,
    sector_nombre,
    alt_min=3800.0,
    alt_max=4800.0,
    mae_nocturno=0.0,
    precip_serie_utc=None,
    chill_pct=None,
):
    """
    Evalúa solo la PRÓXIMA noche (18:00-07:00 Lima; 23-12 UTC):
    - Rojo (oficial): T_min <= 0 °C (Senamhi-FAO, 2010) + etiqueta Da Motta (1961).
    - Ámbar (adaptado): 0 < T_min <= margen, donde margen = max(2.0 °C, mae_nocturno).
    - Viento entra vía índice de enfriamiento en seco (R=0) comparado contra percentiles
      p90/p97 locales (111 noches del entrenamiento, 18 feb a 30 jun 2024).
    - La lluvia acumulada de 24 h previas se reporta aparte como agravante informativo.
    """
    i, j = ubicar_celda_sector(sector_nombre, static, alt_min=alt_min, alt_max=alt_max)
    idx_noche = _indices_proxima_noche(times)
    margen_ambar = max(2.0, float(mae_nocturno))

    T_noche = Y[idx_noche, i, j, 1] - 273.15
    vel_noche = np.sqrt(Y[idx_noche, i, j, 2] ** 2 + Y[idx_noche, i, j, 3] ** 2)

    # Lluvia acumulada de 24 h previas para cada hora nocturna (como agravante)
    lluvia_24h = np.zeros(len(idx_noche), dtype=np.float32)
    if precip_serie_utc is not None and len(precip_serie_utc) > 0:
        for pos, k in enumerate(idx_noche):
            dt_u = pd.Timestamp(a_datetime_utc(times[k]))
            ventana_24h = precip_serie_utc.loc[dt_u - pd.Timedelta(hours=23):dt_u]
            lluvia_24h[pos] = float(ventana_24h.sum()) if len(ventana_24h) > 0 else 0.0

    lluvia_24h_max = float(lluvia_24h.max())
    chill_seco_max = float(calcular_indice_enfriamiento(T_noche, vel_noche, 0.0).max())
    chill_con_lluvia_max = float(calcular_indice_enfriamiento(T_noche, vel_noche, lluvia_24h).max())

    tmin_noche = float(T_noche.min())
    vmax_noche = float(vel_noche.max())
    hora_critica = a_hora_lima(times[idx_noche[int(T_noche.argmin())]])
    ventana_texto = f"{a_hora_lima(times[idx_noche[0]])} a {a_hora_lima(times[idx_noche[-1]])}"
    severidad = clasificar_severidad_helada(tmin_noche)

    # Evaluación por percentiles p90/p97 locales del índice en seco (R = 0)
    nivel_chill = "Sin percentiles locales"
    semaforo_chill = "VERDE"
    if chill_pct is not None and "p90" in chill_pct and "p97" in chill_pct:
        p90 = float(chill_pct["p90"][i, j])
        p97 = float(chill_pct["p97"][i, j])
        if chill_seco_max > p97:
            semaforo_chill = "ROJO"
            nivel_chill = f"> p97 local ({p97:.0f})"
        elif chill_seco_max >= p90:
            semaforo_chill = "ÁMBAR"
            nivel_chill = f"p90–p97 local ({p90:.0f}–{p97:.0f})"
        else:
            nivel_chill = f"< p90 local ({p90:.0f})"

    if tmin_noche <= 0.0 or semaforo_chill == "ROJO":
        semaforo = "ROJO"
        accion = f"Encierro obligatorio en cobertizo ({severidad}). Priorizar crías y madres preñadas/lactantes."
    elif tmin_noche <= margen_ambar or semaforo_chill == "ÁMBAR":
        semaforo = "ÁMBAR"
        accion = f"Encierro preventivo recomendado: T° ≤ {margen_ambar:.1f} °C en caseta puede acercarse a 0 °C junto al suelo."
    else:
        semaforo = "VERDE"
        accion = "Sin helada prevista esta noche (T° > margen de seguridad)."

    return {
        "sector": sector_nombre,
        "altitud_m": float(static["dem"][i, j]),
        "ventana_noche": ventana_texto,
        "tmin_noche": tmin_noche,
        "vmax_noche": vmax_noche,
        "chill_max": chill_seco_max,
        "chill_seco_max": chill_seco_max,
        "chill_con_lluvia_max": chill_con_lluvia_max,
        "lluvia_24h_max": lluvia_24h_max,
        "nivel_chill": nivel_chill,
        "severidad": severidad,
        "margen_ambar": margen_ambar,
        "hora_critica": hora_critica,
        "semaforo": semaforo,
        "accion": accion,
        "evidencia": "Oficial ≤ 0 °C (rojo) / Adaptado margen 2 °C (ámbar) · Senamhi-FAO (2010); Chill adaptado McCaskill et al. (2010)",
    }

# ---------------------------------------------------------------------------
# 5. DECISIÓN 2: PASTOREO MAÑANA (07:00 - 17:00 LIMA / 12 - 22 UTC)
# ---------------------------------------------------------------------------
def _indices_manana_dia(times):
    """
    Devuelve los índices de la madrugada de mañana (00:00-07:00 Lima) y del día de mañana (07:00-17:00 Lima).
    """
    dts_lima = [a_datetime_lima(t) for t in times]
    fecha_base = dts_lima[0].date()
    fecha_manana = fecha_base + timedelta(days=1) if dts_lima[0].hour >= 7 else fecha_base

    idx_madrugada = [k for k, dt in enumerate(dts_lima) if dt.date() == fecha_manana and 0 <= dt.hour <= 7]
    idx_dia = [k for k, dt in enumerate(dts_lima) if dt.date() == fecha_manana and 7 <= dt.hour <= 17]

    if not idx_dia:
        idx_dia = [k for k, dt in enumerate(dts_lima) if 7 <= dt.hour <= 17] or list(range(len(times)))
    if not idx_madrugada:
        idx_madrugada = [k for k, dt in enumerate(dts_lima) if 0 <= dt.hour <= 7] or idx_dia
    return idx_madrugada, idx_dia, fecha_manana.strftime("%d/%m")


def recomendar_pastoreo_sector(Y, times, static, sector_nombre, alt_min=3800.0, alt_max=4800.0, radio_celdas=2):
    """
    Compara zonas cercanas (ventana 5x5) ESTRICTAMENTE dentro de [alt_min, alt_max] para mañana (07:00-17:00 Lima):
    - Regla 1 (Clements et al., 2003): no asume que bajar siempre es más templado (evalúa T real del modelo).
    - Regla 2 (Provisional): prefiere la zona que NO tuvo helada (≤ 0 °C) en la madrugada y,
      entre las que no la tuvieron, la de menor índice de enfriamiento diurno medio.
    """
    i0, j0 = ubicar_celda_sector(sector_nombre, static, alt_min=alt_min, alt_max=alt_max)
    idx_madrugada, idx_dia, fecha_str = _indices_manana_dia(times)

    i_ini, i_fin = max(0, i0 - radio_celdas), min(70, i0 + radio_celdas + 1)
    j_ini, j_fin = max(0, j0 - radio_celdas), min(35, j0 + radio_celdas + 1)

    dem_sub = static["dem"][i_ini:i_fin, j_ini:j_fin]
    mascara_puna_sub = filtrar_puna(dem_sub, alt_min=alt_min, alt_max=alt_max)
    alt_base = float(static["dem"][i0, j0])

    T_mad = Y[idx_madrugada, i_ini:i_fin, j_ini:j_fin, 1] - 273.15
    tuvo_helada_mad = T_mad.min(axis=0) <= 0.0

    T_dia = Y[idx_dia, i_ini:i_fin, j_ini:j_fin, 1] - 273.15
    V_dia = np.sqrt(
        Y[idx_dia, i_ini:i_fin, j_ini:j_fin, 2] ** 2
        + Y[idx_dia, i_ini:i_fin, j_ini:j_fin, 3] ** 2
    )
    chill_dia_medio = calcular_indice_enfriamiento(T_dia, V_dia, 0.0).mean(axis=0)

    # Evaluar zonas altas vs bajas estrictamente dentro de [alt_min, alt_max]
    arriba_mask = mascara_puna_sub & (dem_sub > alt_base + 50.0)
    abajo_mask = mascara_puna_sub & (dem_sub < alt_base - 50.0)

    def _evaluar_grupo(mask):
        if not mask.any():
            return None
        sin_helada = mask & (~tuvo_helada_mad)
        pool = sin_helada if sin_helada.any() else mask
        mejor_idx = np.argmin(np.where(pool, chill_dia_medio, np.inf))
        r, c = np.unravel_index(mejor_idx, dem_sub.shape)
        return {
            "sin_helada": bool(sin_helada.any()),
            "chill": float(chill_dia_medio[r, c]),
            "alt": float(dem_sub[r, c]),
            "t_dia": float(T_dia[:, r, c].mean()),
        }

    cand_arriba = _evaluar_grupo(arriba_mask)
    cand_abajo = _evaluar_grupo(abajo_mask)

    if cand_arriba and cand_abajo:
        # Prioridad 1: evitar zona con helada en madrugada; Prioridad 2: menor índice de enfriamiento diurno
        score_arriba = (0 if cand_arriba["sin_helada"] else 1, cand_arriba["chill"])
        score_abajo = (0 if cand_abajo["sin_helada"] else 1, cand_abajo["chill"])
        if score_arriba <= score_abajo:
            decision = "Pastorea en ladera alta"
            elegido = cand_arriba
            razon = "sin helada de madrugada y menor enfriamiento diurno" if cand_arriba["sin_helada"] else "menor índice de enfriamiento diurno"
        else:
            decision = "Baja el ganado en la puna"
            elegido = cand_abajo
            razon = "evita helada de madrugada en cumbres y tiene menor enfriamiento diurno" if cand_abajo["sin_helada"] else "menor índice de enfriamiento diurno"
    elif cand_arriba:
        decision = "Pastorea en ladera alta"
        elegido = cand_arriba
        razon = "única franja disponible dentro del rango de puna seleccionado"
    elif cand_abajo:
        decision = "Baja el ganado en la puna"
        elegido = cand_abajo
        razon = "única franja disponible dentro del rango de puna seleccionado"
    else:
        decision = "Mantén pastoreo en sector base"
        elegido = {"alt": alt_base, "chill": float(chill_dia_medio.mean()), "sin_helada": not bool(tuvo_helada_mad.any()), "t_dia": float(T_dia.mean())}
        razon = "sin gradiente altitudinal significativo en celdas vecinas dentro del rango"

    return {
        "sector": sector_nombre,
        "fecha_manana": fecha_str,
        "alt_base": alt_base,
        "alt_sugerida": elegido["alt"],
        "decision": decision,
        "motivo": f"Para mañana {fecha_str} (07:00–17:00): ir hacia ~{elegido['alt']:.0f} m ({razon}; T° media diurna {elegido['t_dia']:.1f} °C).",
        "aviso_resolucion": "Cautela: celdas de 3 km (resolución efectiva WRF ~20 km / 7Δx; Skamarock, 2004); quebradas estrechas pueden variar.",
        "evidencia": "Provisional · Reglas físicas de inversión y enfriamiento (Clements et al., 2003; Pulgar Vidal, 2014)",
    }


# ---------------------------------------------------------------------------
# 6. COMPARACIÓN GFS (27 KM) VS. MODELO IA (3 KM) ORDENADA POR FECHA REAL
# ---------------------------------------------------------------------------
def comparar_gfs_vs_ia(X, Y, times, static, sector_nombre, alt_min=3800.0, alt_max=4800.0):
    """
    Compara T (°C) y viento (m/s) del GFS crudo vs. el modelo a 3 km.
    Usa DatetimeIndex real en hora de Lima ordenado cronológicamente para evitar
    que Streamlit ordene los días alfabéticamente.
    """
    i, j = ubicar_celda_sector(sector_nombre, static, alt_min=alt_min, alt_max=alt_max)
    lat_celda = float(static["wrf_lats"][i])
    lon_celda = float(static["wrf_lons"][j])

    glat = static.get("gfs_lats", GFS_LATS)
    glon = static.get("gfs_lons", GFS_LONS)
    gi = int(np.argmin(np.abs(glat - lat_celda)))
    gj = int(np.argmin(np.abs(glon - lon_celda)))

    t_gfs = X[:, gi, gj, 1] - 273.15
    v_gfs = np.sqrt(X[:, gi, gj, 2] ** 2 + X[:, gi, gj, 3] ** 2)

    t_ia = Y[:, i, j, 1] - 273.15
    v_ia = np.sqrt(Y[:, i, j, 2] ** 2 + Y[:, i, j, 3] ** 2)

    dts_lima = pd.DatetimeIndex([a_datetime_lima(t) for t in times])
    orden = np.argsort(dts_lima)

    dts_ord = dts_lima[orden]
    t_gfs_ord = t_gfs[orden]
    t_ia_ord = t_ia[orden]
    v_gfs_ord = v_gfs[orden]
    v_ia_ord = v_ia[orden]

    return {
        "sector": sector_nombre,
        "alt_ia_m": float(static["dem"][i, j]),
        "fechas_lima": dts_ord,
        "t_gfs": t_gfs_ord,
        "t_ia": t_ia_ord,
        "v_gfs": v_gfs_ord,
        "v_ia": v_ia_ord,
        "delta_tmin": float(t_ia_ord.min() - t_gfs_ord.min()),
        "delta_vmax": float(v_ia_ord.max() - v_gfs_ord.max()),
    }


# ---------------------------------------------------------------------------
# 7. DECISIÓN 3: RIESGO DE QUEMA (13:00 LIMA / 18 UTC + DÍAS SECOS + FOSBERG)
# ---------------------------------------------------------------------------
def calcular_indice_fosberg(temp_c, hr_pct, viento_10m_ms):
    """
    Índice Meteorológico de Incendios de Fosberg (FFWI; Fosberg, 1978; NOAA SPC):
    - T en °F = T_C * 9/5 + 32
    - h = HR recortada a [0, 100] %
    - U = viento a 10 m convertido a mph (m/s / 0.44704)
    - Tres tramos sin huecos: h < 10, 10 <= h <= 50, h > 50
    - Resultado recortado a [0, 100].
    """
    t_f = float(temp_c) * 9.0 / 5.0 + 32.0
    h = float(np.clip(hr_pct, 0.0, 100.0))
    u_mph = max(0.0, float(viento_10m_ms) / 0.44704)

    if h < 10.0:
        m = 0.03229 + 0.281073 * h - 0.000578 * h * t_f
    elif h <= 50.0:
        m = 2.22749 + 0.160107 * h - 0.014784 * t_f
    else:
        m = 21.0606 + 0.005565 * (h ** 2) - 0.00035 * h * t_f - 0.483199 * h

    m_ratio = m / 30.0
    eta = 1.0 - 2.0 * m_ratio + 1.5 * (m_ratio ** 2) - 0.5 * (m_ratio ** 3)
    ffwi = (eta * np.sqrt(1.0 + u_mph ** 2)) / 0.3002
    return float(np.clip(ffwi, 0.0, 100.0))


def contar_dias_secos_seguidos(precip_serie_utc, fecha_ref_lima):
    """
    Cuenta días secos seguidos (< 1 mm en día calendario de Lima; ETCCDI)
    hasta la fecha de referencia.
    """
    if precip_serie_utc is None or len(precip_serie_utc) == 0:
        return 0
    serie_lima = precip_serie_utc.copy()
    serie_lima.index = serie_lima.index - pd.Timedelta(hours=5)
    lluvia_diaria = serie_lima.resample("D").sum()
    dias_previos = lluvia_diaria[lluvia_diaria.index.date <= fecha_ref_lima].sort_index(ascending=False)

    seguidos = 0
    for mm in dias_previos:
        if float(mm) < 1.0:
            seguidos += 1
        else:
            break
    return seguidos


def evaluar_riesgo_quema_sector(Y, times, static, sector_nombre, alt_min=3800.0, alt_max=4800.0, precip_serie_utc=None):
    """
    Evalúa riesgo de propagación de quema usando los valores de las 13:00 Lima (18:00 UTC):
    - Verde: HR_13 >= 40 % y < 10 días sin lluvia (< 1 mm).
    - Ámbar (adaptado Senamhi PR-DMA-006): HR_13 < 40 % o >= 10 días secos seguidos.
    - Rojo (adaptado Fosberg 1978 / NOAA SPC): HR_13 < 40 % y FFWI >= 50.
    """
    i, j = ubicar_celda_sector(sector_nombre, static, alt_min=alt_min, alt_max=alt_max)
    dts_lima = [a_datetime_lima(t) for t in times]

    # Buscar las 13:00 Lima (18:00 UTC) más cercana al día objetivo (mañana si existe, si no hoy)
    idx_13 = [k for k, dt in enumerate(dts_lima) if dt.hour == 13]
    if idx_13:
        k_obj = idx_13[1] if len(idx_13) > 1 and dts_lima[0].hour >= 13 else idx_13[0]
    else:
        k_obj = int(np.argmin([abs(dt.hour - 13) for dt in dts_lima]))

    dt_eval_lima = dts_lima[k_obj]
    hr_13 = float(np.clip(Y[k_obj, i, j, 0], 0.0, 100.0))
    t_13 = float(Y[k_obj, i, j, 1] - 273.15)
    v_13 = float(np.sqrt(Y[k_obj, i, j, 2] ** 2 + Y[k_obj, i, j, 3] ** 2))

    ffwi = calcular_indice_fosberg(t_13, hr_13, v_13)
    dias_secos = contar_dias_secos_seguidos(precip_serie_utc, dt_eval_lima.date())

    if hr_13 < 40.0 and ffwi >= 50.0:
        nivel = "ROJO"
        mensaje = "Peligro extremo de propagación: HR < 40 % y Fosberg ≥ 50 a las 13:00. Prohibido quemar pastos."
    elif hr_13 < 40.0 or dias_secos >= 10:
        nivel = "ÁMBAR"
        motivos = []
        if hr_13 < 40.0:
            motivos.append(f"HR 13:00 baja ({hr_13:.0f} % < 40 %)")
        if dias_secos >= 10:
            motivos.append(f"{dias_secos} días secos seguidos (< 1 mm)")
        mensaje = f"Precaución por {', '.join(motivos)}. Evitar quemas agrícolas o pecuarias."
    else:
        nivel = "VERDE"
        mensaje = f"HR 13:00 ({hr_13:.0f} % ≥ 40 %) y {dias_secos} días secos (< 10); sin alerta atmosférica de propagación."

    return {
        "sector": sector_nombre,
        "hora_eval": dt_eval_lima.strftime("%d/%m %H:00"),
        "nivel": nivel,
        "ffwi": round(ffwi, 1),
        "hr_13": hr_13,
        "t_13": t_13,
        "v_13": v_13,
        "dias_secos": dias_secos,
        "mensaje": mensaje,
        "evidencia": "Adaptado · HR < 40 % y días secos (Senamhi PR-DMA-006; ETCCDI) + FFWI ≥ 50 (Fosberg, 1978)",
    }


def crear_figura_mapa(tmin, helada, static, titulo):
    """Genera la figura de matplotlib (fila 0 = sur) lista para st.pyplot()."""
    lats, lons = static["wrf_lats"], static["wrf_lons"]
    ext = [lons.min(), lons.max(), lats.min(), lats.max()]
    fig, ax = plt.subplots(1, 2, figsize=(10, 6), constrained_layout=True)

    im = ax[0].imshow(tmin, origin="lower", extent=ext, cmap="coolwarm")
    ax[0].set_title("Temperatura mínima (°C)")
    fig.colorbar(im, ax=ax[0], shrink=0.75)

    ax[1].imshow(static["dem"], origin="lower", extent=ext, cmap="gray")
    if helada.any():
        ax[1].contourf(
            np.linspace(ext[0], ext[1], tmin.shape[1]),
            np.linspace(ext[2], ext[3], tmin.shape[0]),
            helada.astype(float),
            levels=[0.5, 1.5],
            colors=["#4aa3ff"],
            alpha=0.6,
        )
    ax[1].set_title("Zonas con helada (azul) sobre el relieve")

    for a in ax:
        a.set_xlabel("Longitud")
        a.set_ylabel("Latitud")
    fig.suptitle(titulo)
    return fig