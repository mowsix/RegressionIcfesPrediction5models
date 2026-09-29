"""
Predicción del desempeño en las Pruebas Saber 11 (ICFES)
Aplicación de despliegue — Proyecto Integrador CRISP-DM

Carga los 5 modelos de regresión (uno por área) y el esquema de entrada
serializados, construye un formulario con las variables de contexto y
devuelve la predicción de puntaje por área.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Configuración de página
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Predicción Saber 11 · ICFES",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

RUTA = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------
# Paleta institucional sobria (azul marino + gris + acento dorado)
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
      :root{
        --navy:#0B3C5D; --blue:#1E6091; --accent:#C8A951;
        --bg:#F5F7FA; --card:#FFFFFF; --ink:#1B2733; --muted:#5B6B7B;
        --line:#E3E8EE;
      }
      .stApp{ background:var(--bg); }
      .block-container{ padding-top:1.5rem; max-width:1150px; }

      .hero{
        background:linear-gradient(135deg,var(--navy) 0%,var(--blue) 100%);
        color:#fff; padding:1.6rem 1.9rem; border-radius:14px;
        box-shadow:0 4px 18px rgba(11,60,93,.18); margin-bottom:1.4rem;
      }
      .hero h1{ font-size:1.55rem; margin:0 0 .25rem 0; font-weight:700; }
      .hero p{ margin:0; opacity:.9; font-size:.98rem; }
      .hero .tag{
        display:inline-block; background:rgba(255,255,255,.15);
        border:1px solid rgba(255,255,255,.25); color:#fff;
        padding:.15rem .6rem; border-radius:999px; font-size:.75rem;
        margin-top:.7rem; letter-spacing:.02em;
      }

      .section-title{
        color:var(--navy); font-weight:700; font-size:1.05rem;
        border-left:4px solid var(--accent); padding-left:.6rem;
        margin:.4rem 0 .2rem 0;
      }

      /* Tarjetas de resultado */
      .result-card{
        background:var(--card); border:1px solid var(--line);
        border-radius:12px; padding:1rem 1.1rem; text-align:center;
        box-shadow:0 1px 6px rgba(27,39,51,.05); height:100%;
      }
      .result-card .area{ color:var(--muted); font-size:.82rem;
        text-transform:uppercase; letter-spacing:.04em; margin-bottom:.35rem; }
      .result-card .score{ color:var(--navy); font-size:2.05rem; font-weight:800;
        line-height:1; }
      .result-card .of{ color:var(--muted); font-size:.8rem; }
      .band{ display:inline-block; margin-top:.5rem; padding:.12rem .55rem;
        border-radius:999px; font-size:.72rem; font-weight:600; }

      .global-card{
        background:var(--navy); color:#fff; border-radius:12px;
        padding:1.1rem 1.3rem; text-align:center;
        box-shadow:0 4px 16px rgba(11,60,93,.2);
      }
      .global-card .lbl{ opacity:.85; font-size:.8rem; text-transform:uppercase;
        letter-spacing:.04em; }
      .global-card .val{ font-size:2.4rem; font-weight:800; }
      .global-card .of{ opacity:.75; font-size:.82rem; }

      /* Barras por área */
      .bar-row{ display:flex; align-items:center; gap:.8rem; margin:.35rem 0; }
      .bar-row .name{ width:180px; color:var(--ink); font-size:.9rem; }
      .bar-track{ flex:1; background:#EAEEF3; border-radius:8px; height:16px;
        overflow:hidden; }
      .bar-fill{ height:100%; background:linear-gradient(90deg,var(--blue),var(--navy)); }
      .bar-val{ width:46px; text-align:right; font-weight:700; color:var(--navy);
        font-size:.9rem; }

      footer, #MainMenu{ visibility:hidden; }
      .disclaimer{ color:var(--muted); font-size:.8rem; border-top:1px solid var(--line);
        margin-top:1.6rem; padding-top:.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Carga de modelos y esquema (cacheada)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelos…")
def cargar_artefactos():
    esquema = joblib.load(os.path.join(RUTA, "esquema_entrada.joblib"))
    modelos = {}
    for t in esquema["targets"]:
        modelos[t] = joblib.load(os.path.join(RUTA, f"modelo_{t}.joblib"))
    return esquema, modelos


try:
    ESQUEMA, MODELOS = cargar_artefactos()
except Exception as e:  # noqa: BLE001
    st.error(
        "No se pudieron cargar los modelos. Verifica que los archivos "
        "`modelo_*.joblib` y `esquema_entrada.joblib` estén junto a `app.py`.\n\n"
        f"Detalle: {e}"
    )
    st.stop()

FEATURE_COLS = ESQUEMA["feature_cols"]
CATEGORIAS = ESQUEMA["categorias"]
TARGETS = ESQUEMA["targets"]

# --------------------------------------------------------------------------
# Etiquetas legibles
# --------------------------------------------------------------------------
FEATURE_LABELS = {
    "periodo": "Periodo de presentación",
    "estu_tipodocumento": "Tipo de documento",
    "estu_genero": "Género del estudiante",
    "cole_area_ubicacion": "Área del colegio",
    "cole_bilingue": "¿Colegio bilingüe?",
    "cole_calendario": "Calendario del colegio",
    "cole_caracter": "Carácter del colegio",
    "cole_depto_ubicacion": "Departamento del colegio",
    "cole_genero": "Género del colegio",
    "cole_jornada": "Jornada",
    "cole_naturaleza": "Naturaleza",
    "fami_cuartoshogar": "Cuartos en el hogar",
    "fami_educacionmadre": "Educación de la madre",
    "fami_educacionpadre": "Educación del padre",
    "fami_estratovivienda": "Estrato de la vivienda",
    "fami_personashogar": "Personas en el hogar",
    "fami_tieneautomovil": "¿Tiene automóvil?",
    "fami_tienecomputador": "¿Tiene computador?",
    "fami_tieneinternet": "¿Tiene internet?",
    "fami_tienelavadora": "¿Tiene lavadora?",
}

TARGET_LABELS = {
    "punt_lectura_critica": "Lectura Crítica",
    "punt_matematicas": "Matemáticas",
    "punt_c_naturales": "Ciencias Naturales",
    "punt_sociales_ciudadanas": "Sociales y Ciudadanas",
    "punt_ingles": "Inglés",
}

GRUPOS = {
    "Estudiante": ["periodo", "estu_tipodocumento", "estu_genero"],
    "Colegio": ["cole_area_ubicacion", "cole_naturaleza", "cole_caracter",
                "cole_calendario", "cole_jornada", "cole_genero",
                "cole_bilingue", "cole_depto_ubicacion"],
    "Familia y hogar": ["fami_estratovivienda", "fami_educacionmadre",
                        "fami_educacionpadre", "fami_personashogar",
                        "fami_cuartoshogar", "fami_tieneinternet",
                        "fami_tienecomputador", "fami_tienelavadora",
                        "fami_tieneautomovil"],
}


def pretty_cat(v: str) -> str:
    return {"SIN_DATO": "No reporta", "SIN ESTRATO": "Sin estrato"}.get(v, v)


def banda(score: float):
    """Banda cualitativa orientativa por área (0–100)."""
    if score < 40:
        return "Bajo", "#F3D9D9", "#9B2C2C"
    if score < 55:
        return "Medio-bajo", "#FBE9D0", "#9C5A12"
    if score < 70:
        return "Medio-alto", "#DDEBD8", "#3C6E2F"
    return "Alto", "#D5E4EF", "#0B3C5D"


# --------------------------------------------------------------------------
# Encabezado
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>🎓 Predicción del desempeño · Pruebas Saber 11</h1>
      <p>Estimación del puntaje por área a partir del contexto socioeconómico
      y escolar del estudiante. Modelos de regresión (Gradient Boosting) sobre
      datos abiertos del ICFES.</p>
      <span class="tag">Proyecto Integrador · Metodología CRISP-DM</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Formulario de entrada
# --------------------------------------------------------------------------
# Construir los grupos a renderizar solo con las columnas presentes en el esquema.
# Cualquier feature del esquema no contemplada en GRUPOS se agrupa en "Otros".
grupos_render = {g: [c for c in cols if c in FEATURE_COLS] for g, cols in GRUPOS.items()}
_en_grupos = {c for cols in grupos_render.values() for c in cols}
_faltantes = [c for c in FEATURE_COLS if c not in _en_grupos]
if _faltantes:
    grupos_render["Otros"] = _faltantes
grupos_render = {g: cols for g, cols in grupos_render.items() if cols}

seleccion = {}
with st.form("formulario"):
    for grupo, cols in grupos_render.items():
        st.markdown(f'<div class="section-title">{grupo}</div>', unsafe_allow_html=True)
        columnas = st.columns(3)
        for i, c in enumerate(cols):
            opciones = CATEGORIAS[c]
            with columnas[i % 3]:
                seleccion[c] = st.selectbox(
                    FEATURE_LABELS.get(c, c),
                    options=opciones,
                    format_func=pretty_cat,
                    key=c,
                )
        st.write("")

    enviado = st.form_submit_button("Predecir puntajes", type="primary",
                                    use_container_width=True)

# --------------------------------------------------------------------------
# Predicción y resultados
# --------------------------------------------------------------------------
if enviado:
    X = pd.DataFrame([{c: seleccion[c] for c in FEATURE_COLS}], columns=FEATURE_COLS)

    preds = {}
    for t in TARGETS:
        p = float(MODELOS[t].predict(X)[0])
        preds[t] = float(np.clip(round(p, 1), 0, 100))

    # Puntaje global estimado (fórmula de ponderación ICFES, 0–500)
    glob = ((3 * preds["punt_lectura_critica"]
             + 3 * preds["punt_matematicas"]
             + 3 * preds["punt_sociales_ciudadanas"]
             + 3 * preds["punt_c_naturales"]
             + 1 * preds["punt_ingles"]) / 13) * 5

    st.markdown('<div class="section-title">Resultado de la predicción</div>',
                unsafe_allow_html=True)

    # Tarjetas por área
    cols = st.columns(5)
    for col, t in zip(cols, TARGETS):
        nombre, bg, fg = banda(preds[t])
        col.markdown(
            f"""
            <div class="result-card">
              <div class="area">{TARGET_LABELS[t]}</div>
              <div class="score">{preds[t]:.0f}</div>
              <div class="of">de 100</div>
              <span class="band" style="background:{bg};color:{fg};">{nombre}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    izq, der = st.columns([2, 1])

    # Barras comparativas
    with izq:
        barras = ""
        for t in TARGETS:
            barras += (
                f'<div class="bar-row"><div class="name">{TARGET_LABELS[t]}</div>'
                f'<div class="bar-track"><div class="bar-fill" '
                f'style="width:{preds[t]:.0f}%;"></div></div>'
                f'<div class="bar-val">{preds[t]:.0f}</div></div>'
            )
        st.markdown(barras, unsafe_allow_html=True)

    # Global estimado
    with der:
        st.markdown(
            f"""
            <div class="global-card">
              <div class="lbl">Puntaje global estimado</div>
              <div class="val">{glob:.0f}</div>
              <div class="of">de 500 · aproximado</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="disclaimer">
        <b>Nota metodológica.</b> Las predicciones son estimaciones estadísticas basadas
        en variables de contexto, no una medición del desempeño real del estudiante.
        Los modelos se entrenaron con los periodos 2018-1 y 2019-1, muestra dominada por
        colegios de Calendario B (en su mayoría no oficiales), por lo que generalizan mejor
        a esa población. El puntaje global es una aproximación calculada con la ponderación
        oficial del ICFES a partir de las cinco predicciones.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info("Completa el formulario y presiona **Predecir puntajes** para ver el "
            "resultado estimado en las cinco áreas.")
