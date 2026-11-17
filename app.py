import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pydeck as pdk

st.set_page_config(page_title="Sistema IA - Escasez de Agua", page_icon="💧", layout="wide")

st.title("💧 Sistema Inteligente de Predicción de Escasez de Agua")
st.markdown("Radar Geoespacial y Modelo Predictivo impulsado por **Inteligencia Artificial**")

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
except Exception:
    st.error("⚠️ Verifica que el archivo 'DATOS.xlsx' esté en tu repositorio.")
    st.stop()

# --- BARRA LATERAL ---
st.sidebar.header("🎛️ Simulador de Escenarios")
lluvia = st.sidebar.slider("Precipitaciones / Lluvia (mm)", 0.0, 200.0, 15.0)
temp = st.sidebar.slider("Temperatura Promedio (°C)", 10.0, 40.0, 29.0)
consumo = st.sidebar.slider("Consumo de Agua (m³)", 5000.0, 25000.0, 18500.0)

if len(columnas_entrada) == 3:
    datos_usuario = pd.DataFrame([[lluvia, temp, consumo]], columns=columnas_entrada)
else:
    nivel_est = min(100.0, max(10.0, (lluvia * 0.6) + 15.0))
    datos_usuario = pd.DataFrame([[lluvia, temp, nivel_est, 2100000.0, consumo]], columns=columnas_entrada)

prediccion = modelo_ia.predict(datos_usuario).item()
probabilidad = modelo_ia.predict_proba(datos_usuario).flatten().tolist().pop() * 100

col1, col2 = st.columns([1, 1.6])

with col1:
    st.subheader("🧠 Diagnóstico de la IA")
    st.metric(label="Probabilidad de Escasez", value=f"{probabilidad:.1f}%")
    
    if prediccion == 1:
        st.error("🚨 ALERTA CRÍTICA: Riesgo de escasez en los próximos 15 días.")
        st.markdown("### 📋 Plan de Acción (Paso 7):")
        st.write("- Activar racionamiento en turnos de riego.")
        st.write("- Restringir uso no esencial de agua potable.")
        st.write("- Activar pozos de reserva y cisternas.")
        color_zona = list((235, 50, 50, 150))
        radio_zona = 28000
    else:
        st.success("💧 ESTADO ESTABLE: Suministro de agua suficiente.")
        st.markdown("### 📋 Recomendaciones (Paso 7):")
        st.write("- Mantener monitoreo rutinario de cuencas.")
        st.write("- Continuar campañas de ahorro hídrico.")
        color_zona = list((30, 200, 100, 150))
        radio_zona = 18000

with col2:
    st.subheader("🛰️ Radar Geoespacial de Zonas en Riesgo")
    
    df_zonas = pd.DataFrame({
        "zona": [
            "Reservorio Poechos (Piura) - Zona Crítica",
            "Cuenca Río Tumbes - Zona Agrícola",
            "Valle del Chira - Zona de Riego",
            "Ciudad de Piura - Consumo Urbano"
        ],
        "lat": [-4.691, -3.566, -4.885, -5.194],
        "lon": [-80.525, -80.451, -80.685, -80.632],
        "radio": [radio_zona, radio_zona * 0.85, radio_zona * 0.9, radio_zona * 0.75]
    })

    capa_zonas = pdk.Layer(
        "ScatterplotLayer",
        data=df_zonas,
        get_position=["lon", "lat"],
        get_radius="radio",
        get_fill_color=color_zona,
        get_line_color=list((255, 255, 255, 220)),
        line_width_min_pixels=2,
        stroked=True,
        filled=True,
        pickable=True,
        auto_highlight=True
    )

    vista = pdk.ViewState(latitude=-4.4, longitude=-80.55, zoom=7.3, pitch=0)

    st.pydeck_chart(pdk.Deck(
        layers=[capa_zonas],
        initial_view_state=vista,
        tooltip={"text": "{zona}\nRiesgo IA: " + f"{probabilidad:.1f}%"}
    ))
