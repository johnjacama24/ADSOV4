import streamlit as st
import pandas as pd
import pickle

@st.cache_resource
def cargar_modelo_datos():
    with open("best_model.pkl", "rb") as file:
        data = pickle.load(file)
        modelo = data["model"]
        diccionario_inverso = data["label_encoder_mapping"]
        df = data["dataframe_codificado"]
        return modelo, diccionario_inverso, df

modelo, diccionario_inverso, df = cargar_modelo_datos()

st.title("🔍 Predicción del Estado del Aprendiz")
st.write("Modifique las variables necesarias para realizar una predicción.")

edad = st.slider("Edad", 18, 100, 25)
cantidad_quejas = st.selectbox("Cantidad de quejas", list(range(0, 11)))
estrato = st.selectbox("Estrato socioeconómico", [1, 2, 3, 4, 5, 6])

if st.button("Realizar predicción"):
    try:
        muestra = df.drop(columns=["Estado Aprendiz"], errors="ignore").iloc[[0]].copy()
        muestra.columns = df.drop(columns=["Estado Aprendiz"], errors="ignore").columns

        if "Edad" in muestra.columns:
            muestra["Edad"] = edad
        if "Cantidad de quejas" in muestra.columns:
            muestra["Cantidad de quejas"] = cantidad_quejas
        if "Estrato" in muestra.columns:
            muestra["Estrato"] = estrato

        pred = modelo.predict(muestra)[0]
        resultado = diccionario_inverso.get(pred, f"Desconocido ({pred})")

        st.success(f"📈 Estado del aprendiz predicho: **{resultado}**")
        st.subheader("📌 Datos usados:")
        st.write(muestra)

    except Exception as e:
        st.error("❌ Error al hacer la predicción:")
        st.code(str(e))
