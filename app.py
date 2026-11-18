import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import pydeck as pdk

st.set_page_config(page_title="Gemelo Digital Hidrológico IA", page_icon="💧", layout="wide")
st.title("💧 Sistema Inteligente de Predicción Hídrica (Sequías y Desbordes)")

@st.cache_data
def cargar_y_entrenar():
    df_prueba = pd.read_excel("DATOS.xlsx")
    if list(df_prueba.columns) == ["Lluvia_mm", "Temperatura_C", "Consumo_m3", "Escasez"]:
        df_final = df_prueba.copy()
        cols = ["Lluvia_mm", "Temperatura_C", "Consumo_m3"]
    else:
        archivo_excel = pd.ExcelFile("DATOS.xlsx")
        datos_limpios = []
        for hoja in archivo_excel.sheet_names:
            df_hoja = pd.read_excel(archivo_excel, sheet_name=hoja, header=None)
            for _, fila in df_hoja.iterrows():
                valores = fila.dropna().tolist()
                if len(valores) < 6: continue
                ultimo = str(valores.pop()).strip().lower()
                if ultimo in ['sí', 'si', 'no']:
                    nums = [float(str(x).replace("'", "").replace(",", ".").strip()) for x in valores if str(x).replace("'", "").replace(",", ".").strip().replace(".","",1).isdigit() and (float(str(x).replace("'", "").replace(",", ".").strip()) < 1990 or float(str(x).replace("'", "").replace(",", ".").strip()) > 2030)]
                    if len(nums) >= 5:
                        fila_lista = nums[slice(0, 5)]
                        fila_lista.append(1 if ultimo in ['sí', 'si'] else 0)
                        datos_limpios.append(fila_lista)
        df_final = pd.DataFrame(datos_limpios, columns=["Lluvia_mm", "Temperatura_C", "Nivel_Agua_Porcentaje", "Poblacion", "Consumo_m3", "Escasez"])
        cols = ["Lluvia_mm", "Temperatura_C", "Nivel_Agua_Porcentaje", "Poblacion", "Consumo_m3"]
    modelo = RandomForestClassifier(n_estimators=100, random_state=42).fit(df_final[cols], df_final["Escasez"])
    return modelo, cols, df_final

modelo_ia, columnas_entrada, df_historico = cargar_y_entrenar()

# --- BARRA LATERAL CON SELECTOR DE REGIÓN GEOGRÁFICA ---
st.sidebar.header("🌎 Ubicación y Clima")
region = st.sidebar.selectbox("Selecciona la Cuenca Geográfica:", [
    "Costa Norte (Piura y Tumbes)",
    "Costa Central (Lima - Río Rímac)",
    "Sierra Sur / Altiplano (Arequipa - Puno)"
])

lluvia = st.sidebar.slider("Precipitaciones (mm)", 0.0, 250.0, 120.0)
temp = st.sidebar.slider("Temperatura (°C)", 10.0, 40.0, 24.0)
consumo = st.sidebar.slider("Consumo (m³)", 5000.0, 25000.0, 15000.0)
nivel_reservorio = min(115.0, max(5.0, (lluvia * 0.55) - (temp * 0.30) - (consumo * 0.0004) + 28.0))

datos_usuario = pd.DataFrame([[lluvia, temp, consumo]] if len(columnas_entrada) == 3 else [[lluvia, temp, min(100.0, nivel_reservorio), 2100000.0, consumo]], columns=columnas_entrada)
prediccion_sequia = modelo_ia.predict(datos_usuario).item()
prob_sequia = modelo_ia.predict_proba(datos_usuario).flatten().tolist().pop() * 100

if lluvia >= 155.0 or nivel_reservorio >= 92.0:
    estado_global, prob_desborde = "DESBORDE", min(99.0, ((lluvia - 140.0) / 110.0) * 100)
elif prediccion_sequia == 1:
    estado_global = "SEQUIA"
else:
    estado_global = "NORMAL"

tab_ia, tab_maqueta = st.tabs(["🧠 1. Motor Predictivo y Mapa Regional", "🏔️ 2. Maqueta Virtual 3D"])
with tab_ia:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Riesgo Desborde" if estado_global == "DESBORDE" else "Riesgo Sequía", f"{prob_desborde if estado_global == 'DESBORDE' else prob_sequia:.1f}%")
    m2.metric("Capacidad Embalse", f"{nivel_reservorio:.0f}%")
    m3.metric("Precipitación", f"{lluvia:.0f} mm")
    m4.metric("Temperatura", f"{temp:.1f} °C")
    st.divider()
    col_diag, col_mapa = st.columns([1.1, 1.3])
    
    with col_diag:
        st.markdown(f"### 📋 Plan de Acción: {region}")
        if estado_global == "DESBORDE":
            color_mapa, radio_mapa, txt_estado = list((138, 43, 226, 175)), 26000, f"DESBORDE ({prob_desborde:.1f}%)"
            if "Norte" in region:
                st.warning("🌊 ALERTA DE DESBORDE EN COSTA NORTE:")
                st.write("1. **Poechos:** Apertura controlada de aliviaderos para evitar colapso.")
                st.write("2. **Valles:** Evacuación preventiva en riberas de Chira, Piura y Tumbes.")
                st.write("3. **Drenaje:** Activar motobombas en zonas urbanas vulnerables.")
            elif "Central" in region:
                st.warning("🌊 ALERTA DE HUAICOS Y TURBIDEZ EN LIMA:")
                st.write("1. **La Atarjea:** Cierre temporal de bocatomas por alta turbidez (lodo).")
                st.write("2. **Cuenca Media:** Alerta de huaicos en Chosica y Santa Eulalia.")
                st.write("3. **Abastecimiento:** Activar pozos subterráneos de Sedapal.")
            else:
                st.warning("🌊 ALERTA DE CRECIDA EN SIERRA SUR:")
                st.write("1. **Represas:** Regulación de descargas en sistema Chili / Lagunillas.")
                st.write("2. **Agro:** Protección de andenes y canales contra erosión.")
                st.write("3. **Población:** Monitoreo de desbordes en zonas ribereñas.")
        elif estado_global == "SEQUIA":
            color_mapa, radio_mapa, txt_estado = list((235, 50, 50, 175)), 24000, f"SEQUÍA ({prob_sequia:.1f}%)"
            if "Norte" in region:
                st.error("🚨 ALERTA DE SEQUÍA EN COSTA NORTE:")
                st.write("1. **Riego:** Turnos estrictos para cultivos de arroz, limón y mango.")
                st.write("2. **Reservorio:** Cierre parcial de compuertas en Poechos.")
                st.write("3. **Ciudad:** Restricción de agua potable para usos no esenciales.")
            elif "Central" in region:
                st.error("🚨 ALERTA DE DÉFICIT HÍDRICO EN LIMA:")
                st.write("1. **Lagunas:** Descarga controlada desde represas altoandinas (Marcapomacocha).")
                st.write("2. **Consumo:** Campaña de racionamiento nocturno en distritos urbanos.")
                st.write("3. **Industria:** Reutilización obligatoria de aguas tratadas.")
            else:
                st.error("🚨 ALERTA DE SEQUÍA SEVERA EN SIERRA SUR:")
                st.write("1. **Siembra de Agua:** Uso de reservas en qochas y bofedales altoandinos.")
                st.write("2. **Ganadería:** Distribución de forraje y abrevaderos de emergencia.")
                st.write("3. **Agro:** Priorizar riego tecnificado por goteo.")
        else:
            color_mapa, radio_mapa, txt_estado = list((30, 200, 110, 175)), 18000, f"ESTABLE ({prob_sequia:.1f}%)"
            st.success(f"💧 ESTADO ESTABLE EN {region.upper()}:")
            st.write("1. **Gestión:** Flujo normal en bocatomas y canales de distribución.")
            st.write("2. **Monitoreo:** Vigilancia rutinaria de calidad (pH, turbidez) y caudal.")
            st.write("3. **Prevención:** Mantenimiento preventivo de infraestructura hídrica.")
            
        st.line_chart(df_historico[["Lluvia_mm", "Temperatura_C"]].head(25), height=180)
        
    with col_mapa:
        if "Norte" in region:
            df_zonas = pd.DataFrame({"lugar": ["Reservorio Poechos", "Cuenca Río Tumbes", "Valle Chira", "Piura Urbano"], "lat": [-4.691, -3.566, -4.885, -5.194], "lon": [-80.525, -80.451, -80.685, -80.632], "radio": [radio_mapa, radio_mapa*0.85, radio_mapa*0.9, radio_mapa*0.75]})
            lat_c, lon_c, zoom_c = -4.4, -80.55, 7.3
        elif "Central" in region:
            df_zonas = pd.DataFrame({"lugar": ["Planta La Atarjea", "Chosica (Cuenca Media)", "Santa Eulalia", "San Mateo (Cuenca Alta)"], "lat": [-12.036, -11.943, -11.898, -11.761], "lon": [-76.981, -76.709, -76.663, -76.298], "radio": [radio_mapa*0.7, radio_mapa*0.8, radio_mapa*0.75, radio_mapa*0.85]})
            lat_c, lon_c, zoom_c = -11.9, -76.65, 8.5
        else:
            df_zonas = pd.DataFrame({"lugar": ["Represa Aguada Blanca", "Valle del Chili (Arequipa)", "Cuenca Coata (Puno)", "Lagunillas"], "lat": [-16.231, -16.398, -15.581, -15.712], "lon": [-71.345, -71.535, -70.025, -70.651], "radio": [radio_mapa, radio_mapa*0.85, radio_mapa*0.95, radio_mapa*0.9]})
            lat_c, lon_c, zoom_c = -16.0, -70.8, 7.2
            
        capa = pdk.Layer("ScatterplotLayer", data=df_zonas, get_position=["lon", "lat"], get_radius="radio", get_fill_color=color_mapa, get_line_color=list((255, 255, 255, 220)), line_width_min_pixels=2, stroked=True, filled=True, pickable=True)
        st.pydeck_chart(pdk.Deck(layers=[capa], initial_view_state=pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=zoom_c), tooltip={"text": "{lugar}\nEstado: " + txt_estado}))

with tab_maqueta:
    x, y = np.linspace(0, 12, 90), np.linspace(0, 8, 70)
    X, Y = np.meshgrid(x, y)
    distancia_rio = np.abs(Y - (4.0 + 0.75 * np.sin(X * 0.9)))
    crestas = np.abs(np.sin(X * 1.4 + Y * 0.8)) * 0.6 + np.abs(np.cos(X * 2.2 - Y * 1.5)) * 0.3
    muro_represa = np.where((X >= 5.3) & (X <= 5.7) & (distancia_rio <= 2.2), 1.85, 0.0)
    Z_terreno = np.maximum(0, (5.2 - X) * 0.75) * (0.45 + 0.55 * crestas) + np.maximum(0, (distancia_rio - 0.55) ** 1.5) * 0.4 - 0.65 * np.exp(-(distancia_rio ** 2) * 4.0) + muro_represa

    if estado_global == "DESBORDE":
        escala_suelo, color_agua, color_marc, color_area, z_area, z_txt, alt_emb, alt_rio = [[0.0, "#1b4332"], [0.5, "#52b788"], [1.0, "#ffffff"]], "rgba(114, 9, 183, 0.88)", "#7209b7", "rgba(138, 43, 226, 0.38)", 1.2, 2.2, 1.95, 0.55
        etiquetas = ["1. Cuenca Alta (Lluvia Extrema)", "⚠️ REBALSE EN REPRESA", "⚠️ INUNDACIÓN URBANA", "⚠️ CULTIVOS ANEGADOS"]
    elif estado_global == "SEQUIA":
        escala_suelo, color_agua, color_marc, color_area, z_area, z_txt, alt_emb, alt_rio = [[0.0, "#a47148"], [0.5, "#c38e70"], [1.0, "#e6ccb2"]], "rgba(230, 57, 70, 0.90)", "red", "rgba(255, 0, 0, 0.35)", 0.9, 1.9, -0.25 + (nivel_reservorio / 100.0) * 1.75, -0.43
        etiquetas = ["1. Cuenca Alta (Baja lluvia)", f"⚠️ RIESGO: Embalse ({nivel_reservorio:.0f}%)", "⚠️ RACIONAMIENTO URBANO", "⚠️ SEQUÍA EN CULTIVOS"]
    else:
        escala_suelo, color_agua, color_marc, color_area, z_area, z_txt, alt_emb, alt_rio = [[0.0, "#2d6a4f"], [0.5, "#52b788"], [1.0, "#ffffff"]], "rgba(0, 150, 255, 0.90)", "#0077b6", "rgba(0, 150, 255, 0.0)", 0.8, 1.6, -0.25 + (nivel_reservorio / 100.0) * 1.75, -0.07
        etiquetas = ["1. Cuenca Alta (OK)", f"2. Embalse ({nivel_reservorio:.0f}%)", "3. Ciudad Abastecida", "4. Cultivos Normales"]

    Z_agua = np.where((X >= 5.3) & (X <= 5.7), 1.90 if estado_global == "DESBORDE" else -0.8, np.where(X < 5.3, alt_emb, alt_rio))
    fig = go.Figure([
        go.Surface(z=Z_terreno, x=X, y=Y, colorscale=escala_suelo, showscale=False),
        go.Surface(z=Z_agua, x=X, y=Y, colorscale=[[0, color_agua], [1, color_agua]], showscale=False, opacity=0.92),
        go.Scatter3d(x=[8.2, 8.6, 9.0, 9.5, 10.2, 10.8], y=[2.2, 2.5, 2.1, 5.8, 6.2, 5.5], z=[0.35, 0.35, 0.35, 0.25, 0.25, 0.25], mode="markers", marker=dict(size=9, symbol="square", color=["#ffffff"]*3 + ["#e9c46a" if estado_global == "SEQUIA" else "#2a9d8f"]*3)),
        go.Scatter3d(x=[1.5, 4.8, 8.6, 10.1], y=[4.0, 4.0, 2.3, 5.8], z=[4.2, z_txt, z_txt, z_txt], mode="markers+text", marker=dict(size=8, color=color_marc, symbol="diamond"), text=etiquetas, textposition="top center", textfont=dict(size=12, color="#1d3557"))
    ])
    if estado_global != "NORMAL":
        fig.add_trace(go.Scatter3d(x=[4.8, 8.6, 10.1], y=[4.0, 2.3, 5.8], z=[z_area, z_area, z_area], mode="markers", marker=dict(size=38, color=color_area, line=dict(color=color_marc, width=3))))
    fig.update_layout(scene=dict(xaxis=dict(title="Cuenca Alta ➔ Valle"), yaxis=dict(title="Ancho"), zaxis=dict(title="Elevación", range=[-0.8, 5.5]), camera=dict(eye=dict(x=1.45, y=-1.65, z=1.05)), bgcolor="rgb(240, 245, 250)"), margin=dict(l=0, r=0, b=0, t=0), height=550, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
