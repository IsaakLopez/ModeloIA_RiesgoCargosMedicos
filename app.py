import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------- Configuración ----------
st.set_page_config(page_title="Riesgo Actuarial", page_icon="📊", layout="centered")

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"


# ---------- Carga de recursos (se hace una sola vez) ----------
@st.cache_resource
def cargar_modelos():
    kmeans = joblib.load(MODELS_DIR / "kmeans_riesgo_actuarial.pkl")
    svm = joblib.load(MODELS_DIR / "svm_riesgo_actuarial.pkl")
    return kmeans, svm


@st.cache_data
def cargar_metadata():
    ruta = MODELS_DIR / "model_metadata.json"
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8"))
    return {}


@st.cache_data
def cargar_datos():
    ruta = OUTPUTS_DIR / "insurance_con_clusters.csv"
    return pd.read_csv(ruta) if ruta.exists() else None


kmeans, svm = cargar_modelos()
metadata = cargar_metadata()
datos = cargar_datos()

# ---------- Interfaz ----------
st.title("📊 Segmentación de riesgo actuarial")
st.write("Ingresa los datos del cliente para estimar su grupo de riesgo.")

tab_pred, tab_datos, tab_info = st.tabs(["Predicción", "Datos y clusters", "Info del modelo"])

with tab_pred:
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Edad", min_value=18, max_value=100, value=45)
        sex = st.selectbox("Sexo", ["male", "female"])
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=31.2, step=0.1)
    with col2:
        children = st.number_input("Hijos", min_value=0, max_value=10, value=2)
        smoker = st.selectbox("Fumador", ["yes", "no"])
        region = st.selectbox("Región", ["southeast", "southwest", "northeast", "northwest"])
    charges = st.number_input("Cargos (charges)", min_value=0.0, value=28000.0, step=500.0)

    if st.button("Predecir", type="primary"):
        cliente = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region,
            "charges": charges,
        }])

        try:
            cluster = int(kmeans.predict(cliente)[0])
            mapa = metadata.get("mapa_riesgo", {})
            nivel = mapa.get(str(cluster), "Sin etiqueta")
            st.success(f"Cluster asignado (KMeans): **{cluster}**")
            st.metric("Nivel de riesgo", nivel)
        except Exception as e:
            st.error(f"Error con KMeans: {e}")

        with st.expander("Ver datos enviados al modelo"):
            st.dataframe(cliente)

with tab_datos:
    if datos is not None:
        st.write("Muestra del dataset con clusters:")
        st.dataframe(datos.head(50))
        if "cluster" in datos.columns:
            st.write("Clientes por cluster:")
            st.bar_chart(datos["cluster"].value_counts().sort_index())
    else:
        st.info("No se encontró outputs/insurance_con_clusters.csv")

with tab_info:
    st.json(metadata if metadata else {"info": "Sin metadata disponible"})
