import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pydeck as pdk

st.set_page_config(page_title="Sistema IA - Cuenca Hidrográfica", page_icon="💧", layout="wide")

st.title("💧 Sistema Inteligente de Monitoreo y Predicción de Escasez de Agua")
st.markdown("Integración de **Modelado Hidrológico Espacial** e **Inteligencia Artificial Predictiva**")

@st.cache_data
def cargar_datos():
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
    return modelo, cols, df_final

try:
    modelo_ia, columnas_entrada, df_historico = cargar_datos()
    st.sidebar.success(f"✅ Base de datos conectada ({len(df_historico)} registros)")
except Exception:
    st.error("⚠️ Verifica que el archivo 'DATOS.xlsx' esté en tu repositorio.")
    st.stop()

# --- BARRA LATERAL ---
st.sidebar.header("🎛️ Simulador de Variables IA")
lluvia = st.sidebar.slider("Precipitaciones (mm)", 0.0, 200.0, 15.0)
temp = st.sidebar.slider("Temperatura (°C)", 10.0, 40.0, 29.5)
consumo = st.sidebar.slider("Consumo de Agua (m³)", 5000.0, 25000.0, 18500.0)

nivel_estimado = min(100.0, max(10.0, (lluvia * 0.55) - (temp * 0.3) + 20.0))

if len(columnas_entrada) == 3:
    datos_usuario = pd.DataFrame([[lluvia, temp, consumo]], columns=columnas_entrada)
else:
    datos_usuario = pd.DataFrame([[lluvia, temp, nivel_estimado, 2100000.0, consumo]], columns=columnas_entrada)

prediccion = modelo_ia.predict(datos_usuario).item()
probabilidad = modelo_ia.predict_proba(datos_usuario).flatten().tolist().pop() * 100

# --- CREACIÓN DE PESTAÑAS INTEGRADAS ---
tab1, tab2 = st.tabs(["🧠 Predicción IA y Zonas de Riesgo", "🌍 Maqueta Hidrológica Satelital (WikiWatershed)"])

with tab1:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Probabilidad de Escasez", f"{probabilidad:.1f}%")
    m2.metric("Nivel Estimado Reservorio", f"{nivel_estimado:.1f}%")
    m3.metric("Precipitación Simulada", f"{lluvia:.0f} mm")
    m4.metric("Temperatura Simulada", f"{temp:.1f} °C")

    st.divider()

    col_izq, col_der = st.columns([1.1, 1.4])

    with col_izq:
        st.subheader("Diagnóstico y Plan de Acción")
        if prediccion == 1:
            st.error("🚨 ALERTA ROJA: Alto riesgo de escasez hídrica en 15 días.")
            st.markdown("**Medidas Preventivas Recomendadas:**")
            st.write("1. Activar racionamiento controlado en canales de riego agrícola.")
            st.write("2. Priorizar abastecimiento para consumo poblacional.")
            st.write("3. Habilitar pozos subterráneos de contingencia.")
            color_punto = list((235, 50, 50, 170))
        else:
            st.success("💧 ESTADO ÓPTIMO: Suministro hídrico estable.")
            st.markdown("**Acciones de Mantenimiento:**")
            st.write("1. Mantener monitoreo continuo en estaciones meteorológicas.")
            st.write("2. Conservar reservas en el embalse principal.")
            st.write("3. Continuar programas de eficiencia en riego.")
            color_punto = list((30, 200, 110, 170))

        st.subheader("📊 Comportamiento Histórico (Datos del Grupo)")
        st.line_chart(df_historico[["Lluvia_mm", "Temperatura_C"]].head(30), height=220)

    with col_der:
        st.subheader("🗺️ Mapa de Monitoreo Regional")
        df_mapa = pd.DataFrame({
            "lugar": ["Reservorio Poechos (Piura)", "Cuenca Río Tumbes", "Valle Agrícola Chira", "Piura Centro"],
            "lat": [-4.691, -3.566, -4.885, -5.194],
            "lon": [-80.525, -80.451, -80.685, -80.632],
            "radio": list((22000, 19000, 20000, 16000))
        })

        capa = pdk.Layer(
            "ScatterplotLayer",
            data=df_mapa,
            get_position=["lon", "lat"],
            get_radius="radio",
            get_fill_color=color_punto,
            get_line_color=list((255, 255, 255, 220)),
            line_width_min_pixels=2,
            stroked=True,
            filled=True,
            pickable=True
        )

        vista = pdk.ViewState(latitude=-4.4, longitude=-80.55, zoom=7.3, pitch=0)
        st.pydeck_chart(pdk.Deck(
            layers=[capa],
            initial_view_state=vista,
            tooltip={"text": "{lugar}\nRiesgo de Escasez: " + f"{probabilidad:.1f}%"}
        ))

with tab2:
    st.subheader("Simulador Físico de Cuenca Hidrográfica (Model My Watershed)")
    st.info("💡 Utiliza este simulador espacial para evaluar la escorrentía superficial y luego traslada los niveles de precipitación a la pestaña de **Predicción IA**.")
    
    url_wikiwatershed = "https://modelmywatershed.org/project/"
    components.iframe(url_wikiwatershed, height=650, scrolling=True)
