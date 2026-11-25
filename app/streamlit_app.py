"""
Interfaz funcional en Streamlit para Ganado Protegido.
Solo invoca las funciones lógicas definidas en app/motor.py.
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

APP = Path(__file__).resolve().parent
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from gfs_forecast import make_windows
from motor import (
    SECTORES_PUNA,
    analizar_heladas,
    cargar_recursos,
    comparar_gfs_vs_ia,
    crear_figura_mapa,
    evaluar_cobertizo_sector,
    evaluar_riesgo_quema_sector,
    obtener_gfs_seguro,
    obtener_precipitacion_sectores,
    predecir,
    recomendar_pastoreo_sector,
)

st.set_page_config(page_title="Ganado Protegido · GAIA Lab", layout="wide")


@st.cache_resource
def obtener_modelo_en_cache():
    return cargar_recursos()


st.title("Ganado Protegido · Pronóstico Agroclimático a 3 km")

recursos = obtener_modelo_en_cache()

with st.sidebar:
    st.header("Parámetros de consulta")
    sector = st.selectbox("Sector ganadero", options=list(SECTORES_PUNA.keys()))
    modo = st.radio("Fuente temporal", ["Pronóstico actual", "Fechas pasadas (histórico)"])

    if modo == "Pronóstico actual":
        horas = st.selectbox(
            "Horizonte de pronóstico",
            options=[48, 72, 168],
            index=2,
            format_func=lambda h: f"{h} horas ({h // 24} días)",
        )
        f_ini, f_fin = None, None
    else:
        horas = None
        hoy = date.today()
        f_ini = st.date_input("Desde", value=hoy - timedelta(days=5), max_value=hoy - timedelta(days=1))
        f_fin = st.date_input("Hasta", value=hoy - timedelta(days=3), min_value=f_ini, max_value=hoy - timedelta(days=1))

    st.subheader("Franja altitudinal (Pulgar Vidal, 2014)")
    alt_min = st.number_input("Altura mínima (m)", 2500, 5000, 3800, 100)
    alt_max = st.number_input("Altura máxima (m)", 3000, 5800, 4800, 100)
    st.caption("Suni: 3500–4000 m | Puna: 4000–4800 m. El DEM es la altura media de cada celda de 3 km.")

    st.subheader("Calibración local del modelo")
    mae_nocturno = st.number_input(
        "MAE nocturno del modelo (°C)",
        min_value=0.0,
        max_value=6.0,
        value=0.0,
        step=0.1,
        help="Margen ámbar = máx(2.0 °C, MAE nocturno) según Senamhi-FAO (2010).",
    )

if st.button("Ejecutar pronóstico y decisiones", type="primary"):
    with st.spinner("Procesando GFS y downscaling a 3 km..."):
        X, times, fuente = obtener_gfs_seguro(
            forecast_hours=horas or 48,
            start_date=str(f_ini) if f_ini else None,
            end_date=str(f_fin) if f_fin else None,
        )
        precip_dict = obtener_precipitacion_sectores(forecast_hours=horas or 48)
        precip_sector = precip_dict.get(sector)

        Xw, times_w = make_windows(X, times)
        Y = predecir(Xw, times_w, recursos)

        res_heladas = analizar_heladas(Y, times_w, recursos["static"], alt_min=alt_min, alt_max=alt_max)
        dec1 = evaluar_cobertizo_sector(
            Y,
            times_w,
            recursos["static"],
            sector,
            alt_min=alt_min,
            alt_max=alt_max,
            mae_nocturno=mae_nocturno,
            precip_serie_utc=precip_sector,
            chill_pct=recursos.get("chill_pct"),
        )
        dec2 = recomendar_pastoreo_sector(Y, times_w, recursos["static"], sector, alt_min=alt_min, alt_max=alt_max)
        dec3 = evaluar_riesgo_quema_sector(
            Y,
            times_w,
            recursos["static"],
            sector,
            alt_min=alt_min,
            alt_max=alt_max,
            precip_serie_utc=precip_sector,
        )
        comp = comparar_gfs_vs_ia(X, Y, times_w, recursos["static"], sector, alt_min=alt_min, alt_max=alt_max)

    st.info(f"Fuente: {fuente} | Periodo analizado: {res_heladas['periodo_lima']}")

    # Tarjetas de las 3 decisiones del criador con cita de evidencia
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("1️⃣ Cobertizo esta noche")
        st.metric(
            f"Semáforo: {dec1['semaforo']}",
            f"{dec1['tmin_noche']:.1f} °C",
            delta=f"Enfriamiento máx: {dec1['chill_max']:.0f} kJ/m²·h",
            delta_color="off",
        )
        st.write(dec1["accion"])
        st.caption(f"Ventana: {dec1['ventana_noche']} | Hora mínima: {dec1['hora_critica']} ({dec1['altitud_m']:.0f} m)")
        st.caption(f"Evidencia: {dec1['evidencia']}")

    with c2:
        st.subheader("2️⃣ Pastoreo mañana")
        st.metric(dec2["decision"], f"~{dec2['alt_sugerida']:.0f} m s. n. m.")
        st.write(dec2["motivo"])
        st.caption(dec2["aviso_resolucion"])
        st.caption(f"Evidencia: {dec2['evidencia']}")

    with c3:
        st.subheader("3️⃣ Riesgo de quema (13:00)")
        st.metric(
            f"Semáforo: {dec3['nivel']}",
            f"HR {dec3['hr_13']:.0f}% | FFWI {dec3['ffwi']:.1f}",
            delta=f"{dec3['dias_secos']} días secos seguidos (< 1 mm)",
            delta_color="off",
        )
        st.write(dec3["mensaje"])
        st.caption(f"Evaluado a las {dec3['hora_eval']} (hora Lima) | Viento 10 m: {dec3['v_13']:.1f} m/s")
        st.caption(f"Evidencia: {dec3['evidencia']}")

    st.divider()

    # Comparación GFS (27 km) vs Modelo IA (3 km) ordenada cronológicamente
    st.subheader(f"Comparación GFS (27 km) vs. Modelo IA (3 km) en {sector}")
    g1, g2 = st.columns(2)
    g1.metric("Diferencia T° mínima (IA - GFS)", f"{comp['delta_tmin']:+.1f} °C")
    g2.metric("Diferencia Viento máx (IA - GFS)", f"{comp['delta_vmax']:+.1f} m/s")

    df_comp = pd.DataFrame(
        {"GFS 27 km (°C)": comp["t_gfs"], "Modelo IA 3 km (°C)": comp["t_ia"]},
        index=comp["fechas_lima"],
    )
    df_comp.index.name = "Fecha y hora (Lima)"
    st.line_chart(df_comp)

    # Mapa espacial a 3 km
    fig = crear_figura_mapa(
        res_heladas["tmin"],
        res_heladas["helada"],
        recursos["static"],
        f"Temperatura mínima a 3 km · {res_heladas['periodo_lima']}",
    )
    st.pyplot(fig)
    plt.close(fig)