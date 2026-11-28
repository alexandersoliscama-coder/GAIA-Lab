"""
Interfaz móvil-first en Streamlit para Ganado Protegido.
Solo invoca las funciones lógicas definidas en app/motor.py.
"""
import csv
import sys
from datetime import date, datetime, timedelta
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

# 1) SIEMPRE debe ser la primera llamada de Streamlit
st.set_page_config(
    page_title="Ganado Protegido · GAIA Lab",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# 2) CSS para estilo móvil, paleta crema/verde y tarjetas
CSS = """
<style>
    .block-container {
        max-width: 480px !important;
        padding-top: 1rem !important;
        padding-bottom: 2.5rem !important;
    }
    header[data-testid="stHeader"] {
        background: transparent;
    }
    .gp-banner {
        background: #135B2B;
        color: #FAF7F0;
        padding: 1rem 1.15rem;
        border-radius: 16px;
        margin-bottom: 0.9rem;
    }
    .gp-banner h1 {
        font-size: 1.35rem;
        margin: 0;
        color: #FAF7F0;
        line-height: 1.2;
    }
    .gp-banner p {
        font-size: 0.85rem;
        margin: 0.25rem 0 0 0;
        color: #DCE9DF;
    }
    .gp-warn {
        background: #FCD262;
        color: #2B2100;
        padding: 0.65rem 0.9rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.9rem;
    }
    .gp-card {
        background: #FAF7F0;
        border: 1px solid #D5CEBE;
        border-radius: 14px;
        padding: 1rem 1.05rem;
        margin-bottom: 0.9rem;
    }
    .gp-card-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #5C574B;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .gp-decision {
        font-size: 1.18rem;
        font-weight: 700;
        color: #1C1C1C;
        margin: 0.2rem 0 0.45rem 0;
        line-height: 1.25;
    }
    .gp-pill {
        display: inline-block;
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }
    .gp-pill-rojo { background: #D93829; color: #FFFFFF; }
    .gp-pill-ambar { background: #E89B1E; color: #1C1C1C; }
    .gp-pill-verde { background: #1E7E43; color: #FFFFFF; }
    .gp-meta {
        font-size: 0.88rem;
        color: #3D3A33;
        margin: 0.2rem 0;
    }
    .gp-zones {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.55rem;
        margin-top: 0.65rem;
    }
    .gp-zone-box {
        background: #EFE9D8;
        border-radius: 10px;
        padding: 0.65rem;
        font-size: 0.82rem;
    }
    .gp-zone-box strong {
        display: block;
        margin-bottom: 0.2rem;
        color: #135B2B;
    }
    .gp-day-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #FAF7F0;
        border: 1px solid #D5CEBE;
        border-radius: 12px;
        padding: 0.7rem 0.95rem;
        margin-bottom: 0.5rem;
        font-size: 0.92rem;
    }
    .gp-day-cold {
        background: #FDECEB;
        border-color: #E39B95;
    }
    div.stButton > button[kind="primary"] {
        background-color: #135B2B;
        border-color: #135B2B;
        border-radius: 12px;
        padding: 0.6rem 1rem;
        font-weight: 600;
    }
    div.stButton > button[kind="secondary"] {
        border-radius: 12px;
        border-color: #C7BFA9;
        background-color: #FAF7F0;
    }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# 3) Diccionario bilingüe base (revisar traducción Quechua Áncash con hablantes locales)
TEXTOS = {
    "es": {
        "titulo": "Ganado Protegido 🦙🌲",
        "subtitulo": "Decisiones diarias para cuidar el ganado · 3 km",
        "sector": "Sector",
        "aviso_cache": "Mostrando datos de respaldo local (sin conexión en vivo).",
        "t1_etiqueta": "1. Cobertizo esta noche",
        "t1_rojo": "Usa el cobertizo esta noche",
        "t1_ambar": "Prepárate para usar el cobertizo",
        "t1_verde": "No hace falta usar cobertizo",
        "pill_frio_rojo": "Alerta por frío",
        "pill_frio_ambar": "Precaución por frío",
        "pill_frio_verde": "Noche sin helada",
        "t2_etiqueta": "2. Dónde pastorear",
        "zona_alta": "Pampa alta",
        "zona_baja": "Quebrada baja",
        "sin_zona": "Sin datos en este rango",
        "ver_porque": "Ver por qué (IA 3 km vs. GFS 27 km)",
        "t3_etiqueta": "3. Riesgo de quema (13:00)",
        "t3_evitar": "Evita quemar hoy",
        "t3_precaucion": "Quema solo con precaución",
        "t3_seguro": "Condiciones bajas de riesgo",
        "btn_7d": "Ver próximos 7 días y mapa >",
        "btn_comprobar": "Comprobar pronóstico anterior",
        "btn_volver": "< Volver a decisiones",
        "titulo_7d": "Próximos días y mapa",
        "titulo_comprobar": "Comprobar pronóstico anterior",
        "pregunta_comprobar": "¿Viste escarcha o hielo en la madrugada?",
        "opt_si": "Sí, vi escarcha o hielo",
        "opt_no": "No, estuve allí y no vi",
        "opt_nose": "No pude comprobarlo",
        "btn_guardar": "Guardar observación",
        "guardado_ok": "Observación guardada para calibrar el modelo.",
        "mapa_titulo": "Mapa de temperatura mínima (3 km)",
    },
    "qu": {
        "titulo": "Ganado Protegido 🦙🌲",
        "subtitulo": "Uywakuna qawananpaq yuyaykuna · 3 km",
        "sector": "Llaqta / Sector",
        "aviso_cache": "Ñawpaq waqaychasqa datokunata rikuchkan (mana internetwan).",
        "t1_etiqueta": "1. Kancha kay paqas",
        "t1_rojo": "Kanchaman uywata yaykuchiy kay paqas",
        "t1_ambar": "Kanchata alistay kay paqas",
        "t1_verde": "Manam kanchaqa necesitakunchu",
        "pill_frio_rojo": "Sinchi chiri / Qasa",
        "pill_frio_ambar": "Chiri kanman",
        "pill_frio_verde": "Mana qasa",
        "t2_etiqueta": "2. Maypi michina",
        "zona_alta": "Hanaq pampa",
        "zona_baja": "Ura wayqu",
        "sin_zona": "Mana dato kanchu",
        "ver_porque": "Imarayku (IA 3 km vs. GFS 27 km)",
        "t3_etiqueta": "3. Rupachiy riesgo (13:00)",
        "t3_evitar": "Ama rupachiychu kanan punchaw",
        "t3_precaucion": "Allinta qawaspa rupachiy",
        "t3_seguro": "Pisi riesgo rupachiypaq",
        "btn_7d": "Qatiq 7 punchawkuna y mapa >",
        "btn_comprobar": "Ñawpaq pronósticota qaway",
        "btn_volver": "< Kutiy yuyaykunaman",
        "titulo_7d": "Qatiq punchawkuna y mapa",
        "titulo_comprobar": "Ñawpaq pronósticota qaway",
        "pregunta_comprobar": "¿Paqarin qasata o chullunkuta rikurqankichu?",
        "opt_si": "Awmi, qasata rikurqani",
        "opt_no": "Manam, chaypi karqani, mana karqanchu",
        "opt_nose": "Manam qawayta atirqanichu",
        "btn_guardar": "Waqaychay",
        "guardado_ok": "Waqaychasqa. ¡Yuspagarasunki!",
        "mapa_titulo": "Chiri mapa (3 km)",
    },
}


@st.cache_resource
def obtener_modelo_en_cache():
    return cargar_recursos()


def clasificar_nivel(texto: str) -> str:
    s = (texto or "").upper()
    if "ROJO" in s or "ALTO" in s or "CRÍT" in s or "CRIT" in s:
        return "rojo"
    if "AMBAR" in s or "ÁMBAR" in s or "MODERAD" in s or "MEDIO" in s:
        return "ambar"
    return "verde"


def ejecutar_calculo(recursos, sector, modo, horas, f_ini, f_fin, alt_min, alt_max, mae_nocturno):
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

    return {
        "fuente": fuente,
        "res_heladas": res_heladas,
        "dec1": dec1,
        "dec2": dec2,
        "dec3": dec3,
        "comp": comp,
    }


def guardar_observacion(sector: str, fecha_pron: str, tmin_pron: float, respuesta: str):
    ruta = APP.parent / "data" / "observaciones_campo.csv"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    existe = ruta.exists()
    with ruta.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not existe:
            writer.writerow(["timestamp_lima", "sector", "ventana_noche", "tmin_pron_c", "observacion"])
        writer.writerow([datetime.now().isoformat(timespec="seconds"), sector, fecha_pron, f"{tmin_pron:.1f}", respuesta])


def mostrar_mapa(res_heladas, recursos, titulo_tarjeta: str):
    st.markdown(
        f"""
        <div class="gp-card" style="padding-bottom: 0.4rem;">
            <div class="gp-card-title">🗺️ {titulo_tarjeta}</div>
            <div class="gp-meta">{res_heladas["periodo_lima"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    fig = crear_figura_mapa(
        res_heladas["tmin"],
        res_heladas["helada"],
        recursos["static"],
        f"Mínima a 3 km · {res_heladas['periodo_lima']}",
    )
    st.pyplot(fig)
    plt.close(fig)


# Estado de navegación e idioma
st.session_state.setdefault("vista", "decisiones")
st.session_state.setdefault("idioma", "es")

recursos = obtener_modelo_en_cache()
t = TEXTOS[st.session_state["idioma"]]

# Encabezado superior estilo móvil
st.markdown(
    f"""
    <div class="gp-banner">
        <h1>{t["titulo"]}</h1>
        <p>{t["subtitulo"]}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Barra compacta de sector e idioma
col_sec, col_lang = st.columns([3, 2])
with col_sec:
    sector = st.selectbox(t["sector"], options=list(SECTORES_PUNA.keys()), label_visibility="collapsed")
with col_lang:
    idioma_sel = st.radio(
        "Idioma",
        options=["es", "qu"],
        format_func=lambda x: "Español" if x == "es" else "Quechua",
        horizontal=True,
        label_visibility="collapsed",
        index=0 if st.session_state["idioma"] == "es" else 1,
    )
    if idioma_sel != st.session_state["idioma"]:
        st.session_state["idioma"] = idioma_sel
        st.rerun()

# Parámetros técnicos guardados en desplegable discreto
with st.expander("Ajustes técnicos (horizonte, histórico, altitud, MAE)"):
    modo = st.radio("Fuente temporal", ["Pronóstico actual", "Fechas pasadas (histórico)"], horizontal=True)
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

    c_a1, c_a2 = st.columns(2)
    alt_min = c_a1.number_input("Altura mín. (m)", 2500, 5000, 3800, 100)
    alt_max = c_a2.number_input("Altura máx. (m)", 3000, 5800, 4800, 100)
    mae_nocturno = st.number_input("MAE nocturno (°C)", 0.0, 6.0, 0.0, 0.1)

# Ejecuta solo cuando cambian parámetros (así navegar entre pantallas es instantáneo)
clave_params = (sector, modo, horas, str(f_ini), str(f_fin), alt_min, alt_max, mae_nocturno)
if st.session_state.get("ultima_clave") != clave_params:
    with st.spinner("Calculando pronóstico a 3 km..."):
        st.session_state["resultado"] = ejecutar_calculo(
            recursos, sector, modo, horas, f_ini, f_fin, alt_min, alt_max, mae_nocturno
        )
        st.session_state["ultima_clave"] = clave_params

res = st.session_state["resultado"]
dec1, dec2, dec3 = res["dec1"], res["dec2"], res["dec3"]
comp, res_heladas = res["comp"], res["res_heladas"]

# Aviso amarillo si viene de respaldo/caché
if "cache" in res["fuente"].lower() or "respaldo" in res["fuente"].lower():
    st.markdown(f'<div class="gp-warn">⚠️ {t["aviso_cache"]}</div>', unsafe_allow_html=True)

# ==================== VISTA 1: DECISIONES ====================
if st.session_state["vista"] == "decisiones":
    # Tarjeta 1: Cobertizo
    niv1 = clasificar_nivel(dec1["semaforo"])
    titulo1 = t[f"t1_{niv1}"]
    pill1 = t[f"pill_frio_{niv1}"]
    lluvia_html = (
        f'<div class="gp-meta">🌧️ Lluvia 24 h: <b>{dec1["lluvia_24h_max"]:.1f} mm</b> (agrava el frío)</div>'
        if dec1["lluvia_24h_max"] >= 1.0
        else ""
    )
    st.markdown(
        f"""
        <div class="gp-card">
            <div class="gp-card-title">{t["t1_etiqueta"]}</div>
            <span class="gp-pill gp-pill-{niv1}">{pill1}</span>
            <div class="gp-decision">{titulo1}</div>
            <div class="gp-meta">🌡️ Mínima esperada: <b>{dec1["tmin_noche"]:.1f} °C</b> a las <b>{dec1["hora_critica"]}</b></div>
            <div class="gp-meta">📍 {dec1["altitud_m"]:.0f} m · {dec1["ventana_noche"]}</div>
            {lluvia_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Tarjeta 2: Pastoreo
    def _bloque_zona(titulo_z, z):
        if z is None:
            return f'<div class="gp-zone-box"><strong>{titulo_z}</strong>{t["sin_zona"]}</div>'
        hel = "Sí" if z["helada_madrugada"] else "No"
        return (
            f'<div class="gp-zone-box">'
            f'<strong>{titulo_z} ({z["altura_m"]:.0f} m)</strong>'
            f'Mín: <b>{z["tmin_madrugada"]:.1f} °C</b><br>'
            f'Día: {z["tmedia_dia"]:.1f} °C<br>'
            f'Helada: {hel}'
            f'</div>'
        )

    zonas_html = ""
    if "zona_alta" in dec2 and "zona_baja" in dec2:
        zonas_html = (
            '<div class="gp-zones">'
            + _bloque_zona(t["zona_alta"], dec2["zona_alta"])
            + _bloque_zona(t["zona_baja"], dec2["zona_baja"])
            + "</div>"
        )

    st.markdown(
        f"""
        <div class="gp-card">
            <div class="gp-card-title">{t["t2_etiqueta"]}</div>
            <div class="gp-decision">{dec2["decision"]} (~{dec2["alt_sugerida"]:.0f} m)</div>
            <div class="gp-meta">{dec2["motivo"]}</div>
            {zonas_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander(t["ver_porque"]):
        c_g1, c_g2 = st.columns(2)
        c_g1.metric("Δ T° mín (IA - GFS)", f"{comp['delta_tmin']:+.1f} °C")
        c_g2.metric("Δ Viento (IA - GFS)", f"{comp['delta_vmax']:+.1f} m/s")
        df_comp = pd.DataFrame(
            {"GFS 27 km (°C)": comp["t_gfs"], "Modelo IA 3 km (°C)": comp["t_ia"]},
            index=comp["fechas_lima"],
        )
        st.line_chart(df_comp)

    # Tarjeta 3: Riesgo de quema
    niv3 = clasificar_nivel(dec3["nivel"])
    titulo3 = t["t3_evitar"] if niv3 == "rojo" else (t["t3_precaucion"] if niv3 == "ambar" else t["t3_seguro"])
    viento_kmh = dec3["v_13"] * 3.6
    st.markdown(
        f"""
        <div class="gp-card">
            <div class="gp-card-title">{t["t3_etiqueta"]}</div>
            <span class="gp-pill gp-pill-{niv3}">{dec3["nivel"]}</span>
            <div class="gp-decision">{titulo3}</div>
            <div class="gp-meta">💧 Humedad: <b>{dec3["hr_13"]:.0f}%</b> · 💨 Viento: <b>{viento_kmh:.0f} km/h</b></div>
            <div class="gp-meta">☀️ {dec3["dias_secos"]} días secos · Evaluado {dec3["hora_eval"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Mapa espacial directamente en la primera pantalla
    mostrar_mapa(res_heladas, recursos, t["mapa_titulo"])

    # Botones de navegación inferior
    if st.button(t["btn_7d"], type="primary", use_container_width=True):
        st.session_state["vista"] = "pronostico_7d"
        st.rerun()
    if st.button(t["btn_comprobar"], type="secondary", use_container_width=True):
        st.session_state["vista"] = "comprobar"
        st.rerun()

# ==================== VISTA 2: PRÓXIMOS 7 DÍAS Y MAPA ====================
elif st.session_state["vista"] == "pronostico_7d":
    if st.button(t["btn_volver"], type="secondary"):
        st.session_state["vista"] = "decisiones"
        st.rerun()

    st.subheader(t["titulo_7d"])
    subvista = st.radio("Vista", ["Pronóstico", "Mapa"], horizontal=True, label_visibility="collapsed")

    if subvista == "Pronóstico":
        fechas_dt = pd.to_datetime(comp["fechas_lima"], errors="coerce")
        df_dias = pd.DataFrame({"fecha_dt": fechas_dt, "t_ia": comp["t_ia"]}).dropna()
        if not df_dias.empty:
            df_dias["dia"] = df_dias["fecha_dt"].dt.date
            resumen = df_dias.groupby("dia")["t_ia"].agg(["min", "max"]).reset_index()
            idx_mas_frio = resumen["min"].idxmin()
            for i, fila in resumen.iterrows():
                clase = "gp-day-row gp-day-cold" if i == idx_mas_frio else "gp-day-row"
                marca = " ❄️ Más frío" if i == idx_mas_frio else ""
                st.markdown(
                    f"""
                    <div class="{clase}">
                        <div><b>{fila['dia'].strftime('%d/%m')}</b>{marca}</div>
                        <div>Mín: <b>{fila['min']:.1f} °C</b> · Máx: {fila['max']:.1f} °C</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.line_chart(pd.DataFrame({"IA 3 km (°C)": comp["t_ia"]}, index=comp["fechas_lima"]))
    else:
        mostrar_mapa(res_heladas, recursos, t["mapa_titulo"])

# ==================== VISTA 3: COMPROBAR PRONÓSTICO ====================
elif st.session_state["vista"] == "comprobar":
    if st.button(t["btn_volver"], type="secondary"):
        st.session_state["vista"] = "decisiones"
        st.rerun()

    st.subheader(t["titulo_comprobar"])
    st.markdown(
        f"""
        <div class="gp-card">
            <div class="gp-card-title">{sector}</div>
            <div class="gp-decision">{t["pregunta_comprobar"]}</div>
            <div class="gp-meta">Pronóstico: <b>{dec1["tmin_noche"]:.1f} °C</b> ({dec1["ventana_noche"]})</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    opcion = st.radio(
        "Respuesta",
        options=[t["opt_si"], t["opt_no"], t["opt_nose"]],
        label_visibility="collapsed",
    )
    if st.button(t["btn_guardar"], type="primary", use_container_width=True):
        guardar_observacion(sector, dec1["ventana_noche"], dec1["tmin_noche"], opcion)
        st.success(t["guardado_ok"])