import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import pydeck as pdk

st.set_page_config(page_title="Gemelo Digital Hidrológico IA", page_icon="💧", layout="wide")
st.title("💧 Sistema Inteligente de Predicción Hídrica (Sequías y Desbordes)")
st.markdown("Integración de **Radar Satelital IA** y **Maqueta Topográfica 3D**")

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
    return modelo, cols, df_final

try:
    modelo_ia, columnas_entrada, df_historico = cargar_y_entrenar()
    st.sidebar.success(f"✅ IA entrenada con {len(df_historico)} registros")
except Exception:
    st.error("⚠️ Verifica que 'DATOS.xlsx' esté en tu repositorio.")
    st.stop()

st.sidebar.header("🎛️ Simulador Climático Extremo")
lluvia = st.sidebar.slider("Precipitaciones en Cabecera (mm)", 0.0, 250.0, 120.0)
temp = st.sidebar.slider("Temperatura Media (°C)", 10.0, 40.0, 24.0)
consumo = st.sidebar.slider("Demanda Agrícola y Urbana (m³)", 5000.0, 25000.0, 15000.0)

nivel_reservorio = min(115.0, max(5.0, (lluvia * 0.55) - (temp * 0.30) - (consumo * 0.0004) + 28.0))

if len(columnas_entrada) == 3:
    datos_usuario = pd.DataFrame([[lluvia, temp, consumo]], columns=columnas_entrada)
else:
    datos_usuario = pd.DataFrame([[lluvia, temp, min(100.0, nivel_reservorio), 2100000.0, consumo]], columns=columnas_entrada)

prediccion_sequia = modelo_ia.predict(datos_usuario).item()
prob_sequia = modelo_ia.predict_proba(datos_usuario).flatten().tolist().pop() * 100

if lluvia >= 155.0 or nivel_reservorio >= 92.0:
    estado_global = "DESBORDE"
    prob_desborde = min(99.0, ((lluvia - 140.0) / 110.0) * 100)
elif prediccion_sequia == 1:
    estado_global = "SEQUIA"
else:
    estado_global = "NORMAL"

tab_ia, tab_maqueta = st.tabs(["🧠 1. Motor Predictivo y Mapa Regional", "🏔️ 2. Maqueta Virtual 3D de la Cuenca"])

# ================= PESTAÑA 1: IA + MAPA REGIONAL DE PUNTOS EN RIESGO =================
with tab_ia:
    m1, m2, m3, m4 = st.columns(4)
    if estado_global == "DESBORDE":
        m1.metric("Riesgo de Desborde", f"{prob_desborde:.1f}%", delta="Saturación Crítica", delta_color="inverse")
    else:
        m1.metric("Riesgo de Sequía", f"{prob_sequia:.1f}%")
    m2.metric("Capacidad Embalse", f"{nivel_reservorio:.0f}%")
    m3.metric("Precipitación Simulada", f"{lluvia:.0f} mm")
    m4.metric("Temperatura Simulada", f"{temp:.1f} °C")
    st.divider()
    
    col_diag, col_mapa = st.columns([1, 1.4])
    with col_diag:
        st.markdown("### 📋 Diagnóstico y Plan de Acción")
        if estado_global == "DESBORDE":
            st.warning("🌊 ALERTA DE DESBORDE E INUNDACIÓN: Lluvias extremas (Tipo Fenómeno El Niño).")
            st.write("1. **Compuertas:** Apertura controlada de aliviaderos para evitar ruptura.")
            st.write("2. **Población:** Evacuación preventiva en riberas bajas del río.")
            st.write("3. **Infraestructura:** Activar defensas ribereñas y drenajes pluviales.")
            color_mapa = list((138, 43, 226, 175))
            radio_mapa = 28000
            txt_estado = f"ALERTA DESBORDE ({prob_desborde:.1f}%)"
        elif estado_global == "SEQUIA":
            st.error("🚨 ALERTA DE SEQUÍA: Déficit hídrico proyectado a 15 días.")
            st.write("1. **Compuertas:** Cierre parcial en el Reservorio Poechos.")
            st.write("2. **Agricultura:** Activar turnos estrictos de riego en el valle.")
            st.write("3. **Población:** Priorizar abastecimiento urbano y habilitar pozos.")
            color_mapa = list((235, 50, 50, 175))
            radio_mapa = 26000
            txt_estado = f"ALERTA SEQUÍA ({prob_sequia:.1f}%)"
        else:
            st.success("💧 ESTADO ESTABLE: Equilibrio hídrico en toda la cuenca.")
            st.write("1. **Compuertas:** Flujo normal hacia canales principales.")
            st.write("2. **Agricultura:** Riego abastecido sin restricciones.")
            st.write("3. **Monitoreo:** Continuar registro rutinario en estaciones.")
            color_mapa = list((30, 200, 110, 175))
            radio_mapa = 18000
            txt_estado = f"SUMINISTRO ESTABLE (Riesgo: {prob_sequia:.1f}%)"
            
        st.markdown("### 📊 Tendencia Histórica (Datos Investigados)")
        st.line_chart(df_historico[["Lluvia_mm", "Temperatura_C"]].head(30), height=190)
            
    with col_mapa:
        st.markdown("### 🛰️ Mapa Regional de Zonas en Riesgo (Piura y Tumbes)")
        df_zonas = pd.DataFrame({
            "lugar": [
                "Reservorio Poechos (Piura)",
                "Cuenca Río Tumbes",
                "Valle Agrícola Chira",
                "Ciudad de Piura (Consumo Urbano)"
            ],
            "lat": [-4.691, -3.566, -4.885, -5.194],
            "lon": [-80.525, -80.451, -80.685, -80.632],
            "radio": [radio_mapa, radio_mapa * 0.85, radio_mapa * 0.9, radio_mapa * 0.75]
        })

        capa_zonas = pdk.Layer(
            "ScatterplotLayer",
            data=df_zonas,
            get_position=["lon", "lat"],
            get_radius="radio",
            get_fill_color=color_mapa,
            get_line_color=list((255, 255, 255, 220)),
            line_width_min_pixels=2,
            stroked=True,
            filled=True,
            pickable=True,
            auto_highlight=True
        )

        vista_mapa = pdk.ViewState(latitude=-4.4, longitude=-80.55, zoom=7.3, pitch=0)
        st.pydeck_chart(pdk.Deck(
            layers=[capa_zonas],
            initial_view_state=vista_mapa,
            tooltip={"text": "{lugar}\nEstado IA: " + txt_estado}
        ))

# ================= PESTAÑA 2: MAQUETA TOPOGRÁFICA 3D =================
with tab_maqueta:
    st.subheader("Maqueta Topográfica 3D con Zonas de Riesgo Dinámicas")
    if estado_global == "DESBORDE":
        st.warning("🌊 **MODO INUNDACIÓN ACTIVADO:** El embalse supera su capacidad y el río se desborda sobre el valle.")
        escala_suelo = [[0.0, "#1b4332"], [0.35, "#2d6a4f"], [0.65, "#582f0e"], [1.0, "#ffffff"]]
        color_agua = "rgba(114, 9, 183, 0.88)"
        color_marcador = "#7209b7"
        color_area = "rgba(138, 43, 226, 0.38)"
        z_area, z_texto = 1.2, 2.2
        etiquetas = ["1. Cuenca Alta (Lluvia Extrema)", "⚠️ REBALSE EN REPRESA", "⚠️ INUNDACIÓN URBANA", "⚠️ CULTIVOS ANEGADOS"]
    elif estado_global == "SEQUIA":
        st.error("🚨 **MODO SEQUÍA ACTIVADO:** Embalse en nivel crítico, corte de caudal y cultivos con estrés hídrico.")
        escala_suelo = [[0.0, "#a47148"], [0.35, "#c38e70"], [0.65, "#8a5a44"], [1.0, "#e6ccb2"]]
        color_agua = "rgba(230, 57, 70, 0.90)"
        color_marcador = "red"
        color_area = "rgba(255, 0, 0, 0.35)"
        z_area, z_texto = 0.9, 1.9
        etiquetas = ["1. Cuenca Alta (Baja lluvia)", f"⚠️ RIESGO: Embalse ({nivel_reservorio:.0f}%)", "⚠️ RACIONAMIENTO URBANO", "⚠️ SEQUÍA EN CULTIVOS"]
    else:
        st.info("🔵 **MODO NORMAL ACTIVADO:** Embalse óptimo, flujo continuo hacia el valle y cultivos saludables.")
        escala_suelo = [[0.0, "#2d6a4f"], [0.35, "#52b788"], [0.65, "#7f5539"], [1.0, "#ffffff"]]
        color_agua = "rgba(0, 150, 255, 0.90)"
        color_marcador = "#0077b6"
        color_area = "rgba(0, 150, 255, 0.0)"
        z_area, z_texto = 0.8, 1.6
        etiquetas = ["1. Cuenca Alta (Captación OK)", f"2. Embalse Óptimo ({nivel_reservorio:.0f}%)", "3. Ciudad Abastecida", "4. Cultivos con Riego Normal"]

    x = np.linspace(0, 12, 90)
    y = np.linspace(0, 8, 70)
    X, Y = np.meshgrid(x, y)

    centro_rio = 4.0 + 0.75 * np.sin(X * 0.9)
    distancia_rio = np.abs(Y - centro_rio)
    crestas = np.abs(np.sin(X * 1.4 + Y * 0.8)) * 0.6 + np.abs(np.cos(X * 2.2 - Y * 1.5)) * 0.3
    montanas = np.maximum(0, (5.2 - X) * 0.75) * (0.45 + 0.55 * crestas)
    laderas = np.maximum(0, (distancia_rio - 0.55) ** 1.5) * 0.4
    cauce = -0.65 * np.exp(-(distancia_rio ** 2) * 4.0)
    muro_represa = np.where((X >= 5.3) & (X <= 5.7) & (distancia_rio <= 2.2), 1.85, 0.0)
    Z_terreno = montanas + laderas + cauce + muro_represa

    if estado_global == "DESBORDE":
        altura_embalse, altura_rio_abajo = 1.95, 0.55
    elif estado_global == "SEQUIA":altura_embalse, altura_rio_abajo = -0.25 + (nivel_reservorio / 100.0) * 1.75, -0.43
else:
altura_embalse, altura_rio_abajo = -0.25 + (nivel_reservorio / 100.0) * 1.75, -0.07
Z_agua = np.where(X < 5.3, altura_embalse, altura_rio_abajo)
Z_agua = np.where((X >= 5.3) & (X <= 5.7), 1.90 if estado_global == "DESBORDE" else -0.8, Z_agua)
fig = go.Figure()
fig.add_trace(go.Surface(z=Z_terreno, x=X, y=Y, colorscale=escala_suelo, showscale=False, lighting=dict(ambient=0.75, diffuse=0.85, roughness=0.4)))
fig.add_trace(go.Surface(z=Z_agua, x=X, y=Y, colorscale=[[0, color_agua], [1, color_agua]], showscale=False, opacity=0.92))
color_cultivos = "#e9c46a" if estado_global == "SEQUIA" else "#2a9d8f"
fig.add_trace(go.Scatter3d(
x=[8.2, 8.6, 9.0, 9.5, 10.2, 10.8], y=[2.2, 2.5, 2.1, 5.8, 6.2, 5.5], z=[0.35, 0.35, 0.35, 0.25, 0.25, 0.25],
mode="markers", marker=dict(size=9, symbol="square", color=["#ffffff", "#ffffff", "#ffffff", color_cultivos, color_cultivos, color_cultivos])
))
if estado_global != "NORMAL":
fig.add_trace(go.Scatter3d(
x=[4.8, 8.6, 10.1], y=[4.0, 2.3, 5.8], z=[z_area, z_area, z_area],
mode="markers", marker=dict(size=38, color=color_area, line=dict(color=color_marcador, width=3))
))
fig.add_trace(go.Scatter3d(
x=[1.5, 4.8, 8.6, 10.1], y=[4.0, 4.0, 2.3, 5.8], z=[4.2, z_texto, z_texto, z_texto],
mode="markers+text", marker=dict(size=8, color=color_marcador, symbol="diamond"),
text=etiquetas, textposition="top center", textfont=dict(size=12, color="#1d3557")
))
fig.update_layout(
scene=dict(
xaxis=dict(title="Cuenca Alta ➔ Valle", backgroundcolor="rgb(235, 242, 247)", gridcolor="white"),
yaxis=dict(title="Ancho Cuenca", backgroundcolor="rgb(235, 242, 247)", gridcolor="white"),
zaxis=dict(title="Elevación", backgroundcolor="rgb(235, 242, 247)", gridcolor="white", range=[-0.8, 5.5]),
camera=dict(eye=dict(x=1.45, y=-1.65, z=1.05)), bgcolor="rgb(240, 245, 250)"
),
margin=dict(l=0, r=0, b=0, t=0), height=550, showlegend=False
)
st.plotly_chart(fig, use_container_width=True)
