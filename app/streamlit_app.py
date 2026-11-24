import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

APP = Path(__file__).resolve().parent
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from gfs_forecast import make_windows
from motor import (
    analizar_heladas,
    cargar_recursos,
    crear_figura_mapa,
    obtener_gfs_seguro,
    predecir,
)

st.set_page_config(page_title="Ganado Protegido · GAIA Lab", layout="wide")


@st.cache_resource
def obtener_modelo_en_cache():
    return cargar_recursos()


st.title("Ganado Protegido · Pronóstico a 3 km")
st.caption("Esqueleto funcional: GFS (Open-Meteo) -> Modelo IA (70x35) -> Alerta en Puna")

recursos = obtener_modelo_en_cache()

col1, col2, col3 = st.columns(3)
with col1:
    horas = st.selectbox("Horizonte de pronóstico", options=[48, 72, 168], index=2, format_func=lambda h: f"{h} horas ({h // 24} días)")
with col2:
    alt_min = st.number_input("Altura mínima puna (m)", min_value=2000, max_value=5000, value=3800, step=100)
with col3:
    alt_max = st.number_input("Altura máxima puna (m)", min_value=3000, max_value=5800, value=4800, step=100)

if st.button("Actualizar pronóstico y correr modelo", type="primary"):
    with st.spinner("Descargando GFS y ejecutando modelo a 3 km..."):
        X, times, fuente = obtener_gfs_seguro(forecast_hours=horas)
        Xw, times_w = make_windows(X, times)
        Y = predecir(Xw, times_w, recursos)
        res = analizar_heladas(Y, times_w, recursos["static"], alt_min=alt_min, alt_max=alt_max)

    st.info(f"Fuente: {fuente} | Periodo: {res['periodo_lima']}")

    m1, m2, m3 = st.columns(3)
    m1.metric("Temp. mínima en puna", f"{res['tmin_puna']:.1f} °C")
    m2.metric("Temp. mínima en todo el mapa", f"{res['tmin_dominio']:.1f} °C")
    m3.metric("Puna con helada (< 0 °C)", f"{res['pct_helada_puna']:.1f}%")

    pf = res["punto_frio"]
    st.write(
        f"**Punto más frío en puna:** lat {pf['lat']:.2f}, lon {pf['lon']:.2f} "
        f"({pf['alt']:.0f} m) — **{pf['temp']:.1f} °C** el {pf['hora_lima']}"
    )

    fig = crear_figura_mapa(
        res["tmin"],
        res["helada"],
        recursos["static"],
        f"Temperatura mínima a 3 km · {res['periodo_lima']}",
    )
    st.pyplot(fig)
    plt.close(fig)