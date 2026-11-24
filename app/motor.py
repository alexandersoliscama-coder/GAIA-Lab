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

#Descarga de data
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


# ---------------------------------------------------------------------------
# 3. FILTRO DE PUNA Y MÉTRICAS DE HELADAS
# ---------------------------------------------------------------------------
def filtrar_puna(dem, alt_min=3800.0, alt_max=4800.0):
    """Máscara booleana (70, 35) para aislar la zona altoandina de pastoreo."""
    return (dem >= alt_min) & (dem <= alt_max)


def a_hora_lima(t_utc):
    return (datetime.strptime(str(t_utc), "%Y%m%d%H") - timedelta(hours=5)).strftime("%a %d/%m %H:00")


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