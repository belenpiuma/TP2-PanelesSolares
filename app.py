"""
App de Streamlit: inspección de paneles solares con Deep Learning.
TP2 - Análisis Predictivo Avanzado (ITBA)

Para correrla localmente:  streamlit run app.py
"""
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.cm as cm

DIR = Path(__file__).parent
IMG_SIZE = 224

st.set_page_config(page_title="Inspección de paneles solares", page_icon="☀️", layout="wide")


# ------------------------------------------------------------------
# Carga del modelo y la configuración (se cachea para no recargar)
# ------------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    modelo = keras.models.load_model(DIR / "modelo_paneles.keras")
    with open(DIR / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    return modelo, config


@st.cache_resource
def preparar_gradcam(_modelo):
    base = [l for l in _modelo.layers if isinstance(l, keras.Model) and l.name != "data_augmentation"][0]
    i_base = _modelo.layers.index(base)
    previas = [l for l in _modelo.layers[:i_base] if not isinstance(l, keras.layers.InputLayer)]
    posteriores = _modelo.layers[i_base + 1:]
    ultima_conv = [l.name for l in base.layers if len(l.output.shape) == 4][-1]
    extractor = keras.Model(base.inputs, [base.get_layer(ultima_conv).output, base.output])
    return previas, extractor, posteriores


def preprocesar(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(img, dtype=np.float32)


def grad_cam(x, partes, clase):
    previas, extractor, posteriores = partes
    t = tf.convert_to_tensor(x[None])
    for capa in previas:
        t = capa(t, training=False)
    with tf.GradientTape() as tape:
        mapas, salida = extractor(t, training=False)
        for capa in posteriores:
            salida = capa(salida, training=False)
        score = salida[:, clase]
    grads = tape.gradient(score, mapas)
    pesos = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(mapas[0] * pesos, axis=-1))
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], (IMG_SIZE, IMG_SIZE))[..., 0].numpy()
    return cam


def superponer(x, cam, alpha=0.45):
    color = cm.jet(cam)[..., :3] * 255
    return np.clip((1 - alpha) * x + alpha * color, 0, 255).astype("uint8")


# ------------------------------------------------------------------
# Interfaz
# ------------------------------------------------------------------
modelo, config = cargar_modelo()
CLASES = config["clases"]
NOMBRES = config["nombres_es"]
ACCIONES = config["acciones"]
CLASE_A_ACCION = config["clase_a_accion"]
COSTOS = np.array(config["matriz_costos"])

st.title("☀️ Inspección automática de paneles solares")
st.write(
    "Subí una foto de un panel solar. El modelo (EfficientNetB0 con *transfer learning* y *fine-tuning*) "
    "clasifica su estado y recomienda la acción de mantenimiento. Si la confianza es baja, "
    "el caso se deriva a revisión humana."
)

with st.sidebar:
    st.header("⚙️ Configuración")
    umbral = st.slider("Umbral de confianza para decisión automática", 0.30, 0.99,
                       float(config["umbral_confianza"]), 0.01,
                       help="Por debajo de este valor, la imagen se deriva a un inspector.")
    mostrar_cam = st.checkbox("Mostrar mapa de calor (Grad-CAM)", value=True)
    st.markdown("---")
    st.caption(f"Accuracy en test: {config['metricas_test']['accuracy']:.1%} · "
               f"Kappa: {config['metricas_test']['kappa']:.2f}")
    st.caption("TP2 · Análisis Predictivo Avanzado · ITBA")

archivo = st.file_uploader("Foto del panel", type=["jpg", "jpeg", "png"])

if archivo is None:
    st.info("Esperando una imagen…")
    st.stop()

img = Image.open(archivo)
x = preprocesar(img)
proba = modelo.predict(x[None], verbose=0)[0]
clase = int(proba.argmax())

# Probabilidad de cada acción y decisión de mínimo costo esperado
mapa = np.array([CLASE_A_ACCION[c] for c in CLASES])
proba_accion = np.array([proba[mapa == k].sum() for k in range(len(ACCIONES))])
accion = int((proba_accion @ COSTOS).argmin())
confianza = float(proba_accion.max())

col1, col2 = st.columns(2)
with col1:
    st.subheader("Imagen")
    if mostrar_cam:
        cam = grad_cam(x, preparar_gradcam(modelo), clase)
        st.image(superponer(x, cam), caption="Zonas que más influyeron en la decisión (rojo = más influencia)",
                 width="stretch")
    else:
        st.image(img, width="stretch")

with col2:
    st.subheader("Diagnóstico")
    st.metric("Estado detectado", NOMBRES[CLASES[clase]])
    st.caption(f"Probabilidad de la clase: {proba[clase]:.0%}")

    if confianza < umbral:
        st.warning(f"🔎 **Derivar a revisión humana**: confianza {confianza:.0%} < umbral {umbral:.0%}.")
    else:
        iconos = {"Nada": "✅", "Limpieza": "🧽", "Técnico": "🛠️"}
        textos = {"Nada": "Panel en buen estado: no requiere acción.",
                  "Limpieza": "Enviar cuadrilla de limpieza.",
                  "Técnico": "Enviar técnico especializado: posible daño."}
        a = ACCIONES[accion]
        (st.error if a == "Técnico" else st.success if a == "Nada" else st.info)(
            f"{iconos[a]} **{a}**: {textos[a]}")
        if accion != int(proba_accion.argmax()):
            st.caption(
                f"La acción más probable era **{ACCIONES[int(proba_accion.argmax())]}** "
                f"({proba_accion.max():.0%}), pero se eligió **{a}** por **mínimo costo esperado**: "
                f"con {proba_accion[accion]:.0%} de probabilidad, el riesgo de no actuar "
                f"es más caro que el costo de la acción.")

    st.markdown("**Probabilidad por acción**")
    for k, nombre_accion in enumerate(ACCIONES):
        st.progress(float(proba_accion[k]), text=f"{nombre_accion}: {proba_accion[k]:.1%}")

    st.markdown("**Probabilidad por clase**")
    orden = np.argsort(proba)[::-1]
    for i in orden:
        st.progress(float(proba[i]), text=f"{NOMBRES[CLASES[i]]}: {proba[i]:.1%}")
