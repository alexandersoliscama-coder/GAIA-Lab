import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

APP = Path(__file__).resolve().parent
RAIZ = APP.parent
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from gfs_forecast import LATS as GFS_LATS, LONS as GFS_LONS, get_gfs_forecast, make_windows

# Sectores ganaderos
SECTORES_PUNA = {
    "Cátac / Pastoruri (Recuay)": (-9.80, -77.45),
    "Olleros / Quebrada Honda (Huaraz)": (-9.55, -77.48),
    "Carhuaz / Cordillera Blanca": (-9.28, -77.55),
    "Yungay / Llanganuco": (-9.12, -77.62),
    "Chavín / Conchucos Sur": (-9.58, -77.18),
}

#Arquitectura

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

#Descarga GFS

def obtener_gfs_seguro(forecast_hours=168, start_date=None, end_date=None):
    """
    Descarga GFS desde Open-Meteo (pronóstico de hasta 7 días o fechas pasadas).
    Si Open-Meteo falla o limita peticiones, recurre automáticamente al caché local.
    """
    cache_path = str(APP / "gfs_cache.npz")
    backup_path = str(APP / "gfs_respaldo_demo.npz")

    try:
        if start_date:
            X, times = get_gfs_forecast(start_date=start_date, end_date=end_date or start_date)
            fuente = f"Histórico Open-Meteo ({start_date} a {end_date or start_date})"
        else:
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

#Filtro de punta y utilidades horarias

def filtrar_puna(dem, alt_min=3800.0, alt_max=4800.0):
    """Máscara booleana (70, 35) para aislar la zona altoandina de pastoreo."""
    return (dem >= alt_min) & (dem <= alt_max)


def a_datetime_lima(t_utc):
    return datetime.strptime(str(t_utc), "%Y%m%d%H") - timedelta(hours=5)


def a_hora_lima(t_utc):
    return a_datetime_lima(t_utc).strftime("%a %d/%m %H:00")


def ubicar_celda_sector(sector_nombre, static, alt_min=3800.0, alt_max=4800.0):
    """Ubica la celda (i, j) más cercana al sector dentro del rango de altitud de puna."""
    lat_obj, lon_obj = SECTORES_PUNA[sector_nombre]
    LAT, LON = np.meshgrid(static["wrf_lats"], static["wrf_lons"], indexing="ij")
    dist2 = (LAT - lat_obj) ** 2 + (LON - lon_obj) ** 2
    mascara = filtrar_puna(static["dem"], alt_min=alt_min, alt_max=alt_max)
    if mascara.any():
        dist2 = np.where(mascara, dist2, np.inf)
    i, j = np.unravel_index(np.argmin(dist2), dist2.shape)
    return int(i), int(j)


def analizar_heladas(Y, times, static, alt_min=3800.0, alt_max=4800.0, umbral_c=0.0):
    """Calcula temperatura mínima, zonas con helada y estadísticas dentro de la puna."""
    T_c = Y[..., 1] - 273.15
    tmin = T_c.min(axis=0)
    hora_min = np.asarray(times)[T_c.argmin(axis=0)]
    helada = tmin < umbral_c

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
        "periodo_lima": f"{a_hora_lima(times[0])} a {a_hora_lima(times[-1])} (hora de Lima)",
    }

# Cobertiso nocturno

def evaluar_cobertizo_sector(
    Y,
    times,
    static,
    sector_nombre,
    alt_min=3800.0,
    alt_max=4800.0,
    umbral_t_ambar=2.0,
    umbral_t_rojo=0.0,
    umbral_viento_ambar=6.0,
    umbral_viento_rojo=10.0,
):
    """
    Evalúa T mínima y viento máximo durante horas nocturnas (18:00 a 06:00 Lima)
    en el sector elegido y devuelve un semáforo con umbrales parametrizables.
    """
    i, j = ubicar_celda_sector(sector_nombre, static, alt_min=alt_min, alt_max=alt_max)
    horas_lima = [a_datetime_lima(t).hour for t in times]
    idx_noche = [k for k, h in enumerate(horas_lima) if h >= 18 or h <= 6]
    if not idx_noche:
        idx_noche = list(range(len(times)))

    T_noche = Y[idx_noche, i, j, 1] - 273.15
    U_noche = Y[idx_noche, i, j, 2]
    V_noche = Y[idx_noche, i, j, 3]
    vel_noche = np.sqrt(U_noche**2 + V_noche**2)

    tmin_noche = float(T_noche.min())
    vmax_noche = float(vel_noche.max())
    hora_critica = a_hora_lima(times[idx_noche[int(T_noche.argmin())]])

    if tmin_noche <= umbral_t_rojo or vmax_noche >= umbral_viento_rojo:
        semaforo = "ROJO"
        accion = "Encierro obligatorio en cobertizo (priorizar crías y madres preñadas)."
    elif tmin_noche <= umbral_t_ambar or vmax_noche >= umbral_viento_ambar:
        semaforo = "ÁMBAR"
        accion = "Encierro preventivo recomendado; vigilar descenso térmico nocturno."
    else:
        semaforo = "VERDE"
        accion = "Sin riesgo nocturno severo; dormir en corral abierto o cobertizo ventilado."

    return {
        "sector": sector_nombre,
        "celda": (i, j),
        "altitud_m": float(static["dem"][i, j]),
        "tmin_noche": tmin_noche,
        "vmax_noche": vmax_noche,
        "hora_critica": hora_critica,
        "semaforo": semaforo,
        "accion": accion,
    }

# Pastoreo por gradiente

def recomendar_pastoreo_sector(Y, times, static, sector_nombre, alt_min=3800.0, alt_max=4800.0, radio_celdas=2):
    """
    Compara celdas cercanas (ventana 5x5 ~ 15 km) de distinta altura durante el día
    (07:00 a 16:00 Lima) para sugerir 'Pastorea arriba' o 'Baja el ganado'.
    """
    i0, j0 = ubicar_celda_sector(sector_nombre, static, alt_min=alt_min, alt_max=alt_max)
    horas_lima = [a_datetime_lima(t).hour for t in times]
    idx_dia = [k for k, h in enumerate(horas_lima) if 7 <= h <= 16]
    if not idx_dia:
        idx_dia = list(range(len(times)))

    i_ini, i_fin = max(0, i0 - radio_celdas), min(70, i0 + radio_celdas + 1)
    j_ini, j_fin = max(0, j0 - radio_celdas), min(35, j0 + radio_celdas + 1)

    dem_sub = static["dem"][i_ini:i_fin, j_ini:j_fin]
    alt_base = float(static["dem"][i0, j0])

    T_dia = (Y[idx_dia, i_ini:i_fin, j_ini:j_fin, 1] - 273.15).mean(axis=0)
    V_dia = np.sqrt(
        Y[idx_dia, i_ini:i_fin, j_ini:j_fin, 2] ** 2
        + Y[idx_dia, i_ini:i_fin, j_ini:j_fin, 3] ** 2
    ).mean(axis=0)

    # Índice de confort de pastoreo: penaliza frío (< 5 °C) y viento fuerte (> 5 m/s)
    confort = T_dia - 0.6 * V_dia

    arriba_mask = (dem_sub > alt_base + 80.0) & (dem_sub <= alt_max)
    abajo_mask = (dem_sub < alt_base - 80.0) & (dem_sub >= alt_min - 300.0)

    confort_arriba = float(confort[arriba_mask].max()) if arriba_mask.any() else -np.inf
    confort_abajo = float(confort[abajo_mask].max()) if abajo_mask.any() else -np.inf

    if arriba_mask.any() and confort_arriba >= confort_abajo and float(T_dia[arriba_mask].max()) >= 4.0 and float(V_dia[arriba_mask].min()) <= 6.5:
        decision = "Pastorea arriba"
        alt_sugerida = float(dem_sub[arriba_mask][np.argmax(confort[arriba_mask])])
        motivo = f"Laderas altas (~{alt_sugerida:.0f} m) mantienen buen confort térmico y viento moderado durante el día."
    else:
        decision = "Baja el ganado"
        alt_sugerida = float(dem_sub[abajo_mask][np.argmax(confort[abajo_mask])]) if abajo_mask.any() else alt_base
        motivo = f"Menor exposición al viento y mayor temperatura hacia zonas bajas (~{alt_sugerida:.0f} m)."

    return {
        "sector": sector_nombre,
        "alt_base": alt_base,
        "alt_sugerida": alt_sugerida,
        "decision": decision,
        "motivo": motivo,
        "t_dia_media": float(T_dia.mean()),
        "v_dia_media": float(V_dia.mean()),
    }

# Comparacion GFS 27 KM vs Modelo IA

def comparar_gfs_vs_ia(X, Y, times, static, sector_nombre, alt_min=3800.0, alt_max=4800.0):
    """Compara la serie temporal de T (°C) y viento (m/s) del GFS crudo vs. el modelo a 3 km."""
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

    horas_etiqueta = [a_hora_lima(t) for t in times]

    return {
        "sector": sector_nombre,
        "alt_ia_m": float(static["dem"][i, j]),
        "horas_lima": horas_etiqueta,
        "t_gfs": t_gfs,
        "t_ia": t_ia,
        "v_gfs": v_gfs,
        "v_ia": v_ia,
        "delta_tmin": float(t_ia.min() - t_gfs.min()),
        "delta_vmax": float(v_ia.max() - v_gfs.max()),
    }

# Riesgo de propagacion de fuego

def evaluar_riesgo_quema_sector(Y, times, static, sector_nombre, alt_min=3800.0, alt_max=4800.0):
    """
    Evalúa el riesgo de que una quema de pastos se salga de control usando
    humedad relativa mínima (RH), temperatura máxima (T) y viento máximo diurno.
    """
    i, j = ubicar_celda_sector(sector_nombre, static, alt_min=alt_min, alt_max=alt_max)
    horas_lima = [a_datetime_lima(t).hour for t in times]
    idx_dia = [k for k, h in enumerate(horas_lima) if 9 <= h <= 17]
    if not idx_dia:
        idx_dia = list(range(len(times)))

    rh_min = float(Y[idx_dia, i, j, 0].min())
    tmax = float((Y[idx_dia, i, j, 1] - 273.15).max())
    vmax = float(np.sqrt(Y[idx_dia, i, j, 2] ** 2 + Y[idx_dia, i, j, 3] ** 2).max())

    # Índice simplificado de peligro de propagación (Chandler / Angström adaptado)
    indice = (tmax / 10.0) + (vmax / 3.0) + max(0.0, (60.0 - rh_min) / 15.0)

    if rh_min < 35.0 and vmax >= 7.0 or indice >= 5.0:
        nivel = "EXTREMO"
        mensaje = "Peligro extremo de propagación: aire muy seco y ráfagas de viento. Prohibido realizar quemas."
    elif rh_min < 45.0 and vmax >= 5.0 or indice >= 3.5:
        nivel = "ALTO"
        mensaje = "Riesgo alto de descontrol del fuego por baja humedad y viento en ladera. Evitar cualquier quema."
    elif indice >= 2.2:
        nivel = "MODERADO"
        mensaje = "Condiciones secas moderadas; mantener vigilancia y cortafuegos."
    else:
        nivel = "BAJO"
        mensaje = "Humedad ambiental alta o viento débil; baja probabilidad de propagación rápida."

    return {
        "sector": sector_nombre,
        "nivel": nivel,
        "indice": round(float(indice), 2),
        "rh_min": rh_min,
        "tmax": tmax,
        "vmax": vmax,
        "mensaje": mensaje,
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