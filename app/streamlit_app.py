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
        horas = st.selectbox("Horizonte", [48, 72, 168], index=2, format_func=lambda h: f"{h} h ({h // 24} días)")
        f_ini, f_fin = None, None
    else:
        horas = 48
        hoy = date.today()
        f_ini = st.date_input("Desde", value=hoy - timedelta(days=5))
        f_fin = st.date_input("Hasta", value=hoy - timedelta(days=3))

    st.subheader("Rango altitudinal de puna")
    alt_min = st.number_input("Altura mínima (m)", 2500, 5000, 3800, 100)
    alt_max = st.number_input("Altura máxima (m)", 3000, 5800, 4800, 100)

    st.subheader("Umbrales cobertizo (provisionales)")
    st.caption("Umbrales provisionales; se actualizarán con fuentes revisadas en el siguiente PR.")
    t_ambar = st.number_input("T° Ámbar (°C)", -5.0, 10.0, 2.0, 0.5)
    t_rojo = st.number_input("T° Rojo (°C)", -10.0, 5.0, 0.0, 0.5)
    v_ambar = st.number_input("Viento Ámbar (m/s)", 1.0, 20.0, 6.0, 0.5)
    v_rojo = st.number_input("Viento Rojo (m/s)", 2.0, 25.0, 10.0, 0.5)

if st.button("Ejecutar pronóstico y decisiones", type="primary"):
    with st.spinner("Procesando GFS y downscaling a 3 km..."):
        X, times, fuente = obtener_gfs_seguro(
            forecast_hours=horas,
            start_date=str(f_ini) if f_ini else None,
            end_date=str(f_fin) if f_fin else None,
        )
        Xw, times_w = make_windows(X, times)
        Y = predecir(Xw, times_w, recursos)

        res_heladas = analizar_heladas(Y, times_w, recursos["static"], alt_min=alt_min, alt_max=alt_max)
        dec1 = evaluar_cobertizo_sector(
            Y, times_w, recursos["static"], sector,
            alt_min=alt_min, alt_max=alt_max,
            umbral_t_ambar=t_ambar, umbral_t_rojo=t_rojo,
            umbral_viento_ambar=v_ambar, umbral_viento_rojo=v_rojo,
        )
        dec2 = recomendar_pastoreo_sector(Y, times_w, recursos["static"], sector, alt_min=alt_min, alt_max=alt_max)
        dec3 = evaluar_riesgo_quema_sector(Y, times_w, recursos["static"], sector, alt_min=alt_min, alt_max=alt_max)
        comp = comparar_gfs_vs_ia(X, Y, times_w, recursos["static"], sector, alt_min=alt_min, alt_max=alt_max)

    st.caption(f"Fuente: {fuente} | Periodo: {res_heladas['periodo_lima']}")

    # Tarjetas de las 3 decisiones del criador
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("1️⃣ Cobertizo nocturno")
        st.metric(f"Semáforo: {dec1['semaforo']}", f"{dec1['tmin_noche']:.1f} °C / {dec1['vmax_noche']:.1f} m/s")
        st.write(dec1["accion"])
        st.caption(f"Hora crítica: {dec1['hora_critica']} ({dec1['altitud_m']:.0f} m)")
    with c2:
        st.subheader("2️⃣ Pastoreo mañana")
        st.metric(dec2["decision"], f"~{dec2['alt_sugerida']:.0f} m s. n. m.")
        st.write(dec2["motivo"])
    with c3:
        st.subheader("3️⃣ Riesgo de quema")
        st.metric(f"Nivel: {dec3['nivel']}", f"RH mín {dec3['rh_min']:.0f}% | V máx {dec3['vmax']:.1f} m/s")
        st.write(dec3["mensaje"])

    st.divider()

    # Comparación GFS (27 km) vs Modelo IA (3 km)
    st.subheader(f"Comparación GFS (27 km) vs. Modelo IA (3 km) en {sector}")
    g1, g2 = st.columns(2)
    g1.metric("Diferencia T° mínima (IA - GFS)", f"{comp['delta_tmin']:+.1f} °C")
    g2.metric("Diferencia Viento máx (IA - GFS)", f"{comp['delta_vmax']:+.1f} m/s")

    df_comp = pd.DataFrame(
        {"GFS 27 km (°C)": comp["t_gfs"], "Modelo IA 3 km (°C)": comp["t_ia"]},
        index=comp["horas_lima"],
    )
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