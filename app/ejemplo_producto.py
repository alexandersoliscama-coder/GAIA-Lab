"""
Hackathon IA PUCP 2026 · Producto de alerta de heladas con modelo entrenado.

    GFS actual (Open-Meteo)  ->  Net (models/modelo.pt)  ->  alerta de heladas (70x35)
"""
import argparse
import os
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gfs_forecast import LATS as GFS_LATS, LONS as GFS_LONS, get_gfs_forecast, make_windows

VARS = ["RH", "T", "U", "V"]
APP = Path(__file__).resolve().parent          # carpeta app/
RAIZ = APP.parent                                # raíz del repositorio


# ---------------------------------------------------------------------------
# 1. ARQUITECTURA E INFERENCIA DE TU MODELO
# ---------------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self, w=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(14, w, 3, padding=1), nn.GELU(),
            nn.Conv2d(w, w, 3, padding=1), nn.GELU()
        )
        self.static = nn.Parameter(torch.zeros(1, 8, 70, 35))
        self.dec = nn.Sequential(
            nn.Conv2d(w + 9, w, 3, padding=1), nn.GELU(),
            nn.Conv2d(w, w, 3, padding=1), nn.GELU(),
            nn.Conv2d(w, 4, 1)
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


def predecir(Xw, times, static, ruta_modelo=None, ruta_stats=None):
    """
    Entrada:
      Xw     : (N, 3, 17, 13, 4) ventanas [t-2, t-1, t] del GFS en unidades físicas.
      times  : (N,) timestamps UTC ('YYYYMMDDHH') correspondientes a cada ventana.
      static : contenido de static.npz (dem, wrf_lats, wrf_lons, gfs_lats, gfs_lons).
    Salida:
      Y      : (N, 70, 35, 4) mapas a 3 km en unidades físicas (fila 0 = sur).
    """
    ruta_modelo = Path(ruta_modelo or (RAIZ / "models" / "modelo.pt"))
    ruta_stats = Path(ruta_stats or (RAIZ / "models" / "stats.npz"))

    if not ruta_modelo.exists():
        raise SystemExit(f"No encuentro {ruta_modelo}. Ejecuta train/starter_notebook.ipynb primero.")
    if not ruta_stats.exists():
        raise SystemExit(f"No encuentro {ruta_stats}. Guarda stats.npz desde train/starter_notebook.ipynb.")

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # Cargar estadísticas de normalización del entrenamiento
    stats = np.load(ruta_stats)
    mu_x = stats["mu_x"]
    sd_x = stats["sd_x"]
    clim_t = torch.tensor(stats["clim_all"], dtype=torch.float32, device=dev)
    asd_t = torch.tensor(stats["anom_sd"], dtype=torch.float32, device=dev)

    # Preparar grilla espacial (GRID) y relieve normalizado (DEM)
    glat = static.get("gfs_lats", GFS_LATS)
    glon = static.get("gfs_lons", GFS_LONS)
    wlat, wlon = static["wrf_lats"], static["wrf_lons"]
    LAT, LON = np.meshgrid(wlat, wlon, indexing="ij")
    gy = (LAT - glat[0]) / (glat[-1] - glat[0]) * 2 - 1
    gx = (LON - glon[0]) / (glon[-1] - glon[0]) * 2 - 1
    grid_t = torch.tensor(np.stack([gx, gy], -1)[None], dtype=torch.float32, device=dev)

    dem = static["dem"].astype(np.float32)
    dem_t = torch.tensor((dem - dem.mean()) / dem.std(), dtype=torch.float32, device=dev)[None, None]

    # Cargar pesos del modelo entrenado
    modelo = Net().to(dev)
    modelo.load_state_dict(torch.load(ruta_modelo, map_location=dev, weights_only=True))
    modelo.eval()

    # Normalizar entrada, inferir y desnormalizar salida
    Xn = torch.tensor((Xw - mu_x) / sd_x, dtype=torch.float32, device=dev)
    horas = torch.tensor([int(str(t)[-2:]) for t in times], dtype=torch.long, device=dev)

    with torch.no_grad():
        pred_norm = modelo(Xn, horas.float(), grid_t, dem_t)
        Y = (clim_t[horas] + pred_norm * asd_t).cpu().numpy().astype(np.float32)

    return Y


# ---------------------------------------------------------------------------
# 2. LO QUE EL PRODUCTO HACE CON LA SALIDA DEL MODELO
# ---------------------------------------------------------------------------
def alerta_heladas(Y, times, umbral_c=0.0):
    """Para cada píxel: ¿habrá helada en el periodo? y la hora (Lima) de la mínima."""
    T_c = Y[..., 1] - 273.15                               # (N, 70, 35)
    tmin = T_c.min(axis=0)
    hora_min = np.asarray(times)[T_c.argmin(axis=0)]
    helada = tmin < umbral_c
    return tmin, hora_min, helada


def a_hora_lima(t_utc):
    return (datetime.strptime(str(t_utc), "%Y%m%d%H") - timedelta(hours=5)).strftime("%a %d/%m %H:00")


def guardar_mapa(tmin, helada, static, titulo, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lats, lons = static["wrf_lats"], static["wrf_lons"]
    ext = [lons.min(), lons.max(), lats.min(), lats.max()]
    fig, ax = plt.subplots(1, 2, figsize=(9, 8), constrained_layout=True)
    im = ax[0].imshow(tmin, origin="lower", extent=ext, cmap="coolwarm")
    ax[0].set_title("Temperatura mínima (°C)")
    fig.colorbar(im, ax=ax[0], shrink=0.7)
    ax[1].imshow(static["dem"], origin="lower", extent=ext, cmap="gray")
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
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Producto de alerta de heladas con IA")
    ap.add_argument("--static", default=str(RAIZ / "data" / "static.npz"), help="ruta a static.npz")
    ap.add_argument("--modelo", default=str(RAIZ / "models" / "modelo.pt"), help="ruta a modelo.pt")
    ap.add_argument("--stats", default=str(RAIZ / "models" / "stats.npz"), help="ruta a stats.npz")
    ap.add_argument("--horas", type=int, default=24, help="horas de pronóstico hacia adelante")
    ap.add_argument("--desde", help="fecha pasada YYYY-MM-DD (opcional)")
    ap.add_argument("--hasta", help="fecha pasada YYYY-MM-DD (opcional)")
    a = ap.parse_args()

    if not os.path.exists(a.static):
        raise SystemExit(f"No encuentro {a.static}. Baja la data de Kaggle a la carpeta data/ (ver README).")

    static = dict(np.load(a.static))
    if a.desde:
        X, times = get_gfs_forecast(start_date=a.desde, end_date=a.hasta or a.desde)
    else:
        X, times = get_gfs_forecast(forecast_hours=a.horas, cache_path=str(APP / "gfs_cache.npz"))

    Xw, times = make_windows(X, times)
    Y = predecir(Xw, times, static, ruta_modelo=a.modelo, ruta_stats=a.stats)
    assert Y.shape[1:] == (70, 35, 4), f"predecir() debe devolver (N, 70, 35, 4), devolvió {Y.shape}"

    tmin, hora_min, helada = alerta_heladas(Y, times)
    os.makedirs(APP / "salidas", exist_ok=True)
    ruta = str(APP / "salidas" / "alerta_heladas.png")
    periodo = f"{a_hora_lima(times[0])} a {a_hora_lima(times[-1])} (hora de Lima)"
    guardar_mapa(tmin, helada, static, f"Alerta de heladas (Modelo IA) · {periodo}", ruta)

    print(f"Periodo: {periodo}")
    print(f"Temperatura mínima en el dominio: {tmin.min():.1f} °C")
    print(f"Zonas con helada: {100 * helada.mean():.0f}% del territorio")
    if helada.any():
        i, j = np.unravel_index(tmin.argmin(), tmin.shape)
        print(
            f"Punto más frío: lat {static['wrf_lats'][i]:.2f}, lon {static['wrf_lons'][j]:.2f}, "
            f"{static['dem'][i, j]:.0f} m, {tmin[i, j]:.1f} °C el {a_hora_lima(hora_min[i, j])}"
        )
    print("Mapa guardado en", ruta)


if __name__ == "__main__":
    main()