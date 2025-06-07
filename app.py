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

        # Limpieza preventiva de columnas
        df.columns = df.columns.str.strip() \
                                .str.replace("\u200b", "", regex=False) \
                                .str.replace("\xa0", "", regex=False) \
                                .str.replace(",", "", regex=False) \
                                .str.replace("–", "-", regex=False)
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

        # Limpieza a columnas de muestra
        muestra.columns = muestra.columns.str.strip() \
                                          .str.replace("\u200b", "", regex=False) \
                                          .str.replace("\xa0", "", regex=False) \
                                          .str.replace(",", "", regex=False) \
                                          .str.replace("–", "-", regex=False)

        columnas_entrenadas = df.drop(columns=["Estado Aprendiz"], errors="ignore").columns
        muestra.columns = columnas_entrenadas

        # Asignar valores de usuario
        if "Edad" in muestra.columns:
            muestra["Edad"] = edad
        if "Cantidad de quejas" in muestra.columns:
            muestra["Cantidad de quejas"] = cantidad_quejas
        if "Estrato" in muestra.columns:
            muestra["Estrato"] = estrato

        # Realizar predicción
        pred = modelo.predict(muestra)[0]
        resultado = diccionario_inverso.get(pred, f"Desconocido ({pred})")

        st.success(f"📈 Estado del aprendiz predicho: **{resultado}**")
        st.subheader("📌 Datos usados:")
        st.write(muestra)

    except Exception as e:
        st.error("❌ Error al hacer la predicción:")
        st.code(str(e))

        # 🔍 Mostrar diferencias entre columnas si hay error
        columnas_modelo = set(df.drop(columns=["Estado Aprendiz"], errors="ignore").columns)
        columnas_muestra = set(muestra.columns)

        faltantes = columnas_modelo - columnas_muestra
        extras = columnas_muestra - columnas_modelo

        if faltantes or extras:
            st.warning("⚠️ Diferencias encontradas en los nombres de columnas:")
            if faltantes:
                st.write("🟥 Faltan en muestra:")
                st.code("\n".join(sorted(faltantes)))
            if extras:
                st.write("🟦 Excedentes en muestra:")
                st.code("\n".join(sorted(extras)))
