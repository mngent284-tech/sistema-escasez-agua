import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pydeck as pdk

st.set_page_config(page_title="Sistema IA - Escasez de Agua", page_icon="💧", layout="wide")

st.title("💧 Sistema Inteligente de Predicción de Escasez de Agua")
st.markdown("Gemelo Digital 3D y Modelo Predictivo impulsado por **Inteligencia Artificial**")

@st.cache_data
def cargar_y_entrenar():
    nombre_archivo = "DATOS.xlsx"
    df_prueba = pd.read_excel(nombre_archivo)
    
    if list(df_prueba.columns) == ["Lluvia_mm", "Temperatura_C", "Consumo_m3", "Escasez"]:
        df_final = df_prueba.copy()
        cols = ["Lluvia_mm", "Temperatura_C", "Consumo_m3"]
    else:
        archivo_excel = pd.ExcelFile(nombre_archivo)
        datos_limpios = []
        for hoja in archivo_excel.sheet_names:
            df_hoja = pd.read_excel(archivo_excel, sheet_name=hoja, header=None)
            for _, fila in df_hoja.iterrows():
                valores = fila.dropna().tolist()
                if len(valores) < 6:
                    continue
                ultimo = str(valores.pop()).strip().lower()
                if ultimo in ['sí', 'si', 'no']:
                    nums = []
                    for item in valores:
                        txt = str(item).replace("'", "").replace(",", ".").strip()
                        try:
                            n = float(txt)
                            if n < 1990 or n > 2030:
                                nums.append(n)
                        except:
                            continue
                    if len(nums) >= 5:
                        fila_lista = nums[slice(0, 5)]
                        fila_lista.append(1 if ultimo in ['sí', 'si'] else 0)
                        datos_limpios.append(fila_lista)
        df_final = pd.DataFrame(datos_limpios, columns=["Lluvia_mm", "Temperatura_C", "Nivel_Agua_Porcentaje", "Poblacion", "Consumo_m3", "Escasez"])
        cols = ["Lluvia_mm", "Temperatura_C", "Nivel_Agua_Porcentaje", "Poblacion", "Consumo_m3"]
        
    X = df_final[cols]
    y = df_final["Escasez"]
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X, y)
    return modelo, cols, len(df_final)

try:
    modelo_ia, columnas_entrada, total_registros = cargar_y_entrenar()
    st.sidebar.success(f"✅ IA Entrenada con {total_registros} registros históricos")
except Exception as e:
    st.error("⚠️ Asegúrate de haber subido el archivo 'DATOS.xlsx' a tu repositorio de GitHub.")
    st.stop()

st.sidebar.header("🎛️ Simulador de Escenarios")
lluvia = st.sidebar.slider("Precipitaciones / Lluvia (mm)", 0.0, 200.0, 110.0)
temp = st.sidebar.slider("Temperatura Promedio (°C)", 10.0, 40.0, 27.5)
consumo = st.sidebar.slider("Consumo de Agua (m³)", 5000.0, 25000.0, 19000.0)

if len(columnas_entrada) == 3:
    datos_usuario = pd.DataFrame([[lluvia, temp, consumo]], columns=columnas_entrada)
else:
    nivel_est = min(100.0, max(10.0, (lluvia * 0.6) + 15.0))
    datos_usuario = pd.DataFrame([[lluvia, temp, nivel_est, 2100000.0, consumo]], columns=columnas_entrada)

prediccion = modelo_ia.predict(datos_usuario).item()
probabilidad = modelo_ia.predict_proba(datos_usuario).flatten().tolist().pop() * 100

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("🧠 Diagnóstico de la IA")
    st.metric(label="Probabilidad de Escasez", value=f"{probabilidad:.1f}%")
    
    if prediccion == 1:
        st.error("🚨 ALERTA CRÍTICA: Riesgo de escasez en los próximos 15 días.")
        st.markdown("### 📋 Plan de Acción (Paso 7):")
        st.write("- Activar racionamiento en turnos de riego.")
        st.write("- Restringir uso no esencial de agua potable.")
        st.write("- Activar pozos de reserva y cisternas.")
        color_rgb = [255, 50, 50, 180]
        altura_columna = 45000
    else:
        st.success("💧 ESTADO ESTABLE: Suministro de agua suficiente.")
        st.markdown("### 📋 Recomendaciones (Paso 7):")
        st.write("- Mantener monitoreo rutinario de cuencas.")
        st.write("- Continuar campañas de ahorro hídrico.")
        color_rgb = [40, 200, 100, 180]
        altura_columna = 15000

with col2:
    st.subheader("🌍 Gemelo Digital 3D (Maqueta Virtual)")
    
    # Datos geográficos de las cuencas
    df_mapa = pd.DataFrame({
        "nombre": ["Reservorio Poechos (Piura)", "Cuenca Río Tumbes", "Valle Agrícola Chira"],
        "lat": [-4.691, -3.566, -4.885],
        "lon": [-80.525, -80.451, -80.685],
        "altura": [altura_columna, altura_columna * 0.8, altura_columna * 0.9]
    })

    # Capa de columnas 3D interactivas
    capa_3d = pdk.Layer(
        "ColumnLayer",
        data=df_mapa,
        get_position=["lon", "lat"],
        get_elevation="altura",
        elevation_scale=1,
        radius=12000,
        get_fill_color=color_rgb,
        pickable=True,
        auto_highlight=True,
    )

    vista = pdk.ViewState(latitude=-4.3, longitude=-80.5, zoom=7.5, pitch=45)

    # Renderizamos el mapa nativo 3D (funciona sin bloqueos ni API Keys)
    st.pydeck_chart(pdk.Deck(
        layers=[capa_3d],
        initial_view_state=vista,
        tooltip={"text": "{nombre}\nRiesgo calculado: " + f"{probabilidad:.1f}%"}
    ))
