import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

st.set_page_config(page_title="Gemelo Digital Hidrológico IA", page_icon="💧", layout="wide")

st.title("💧 Sistema Inteligente de Predicción de Escasez de Agua")
st.markdown("Integración de **Machine Learning Predictivo** y **Maqueta Topográfica 3D**")

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
    st.sidebar.success(f"✅ Modelo IA entrenado con {len(df_historico)} registros")
except Exception:
    st.error("⚠️ Asegúrate de que 'DATOS.xlsx' esté en tu repositorio.")
    st.stop()

# --- BARRA LATERAL GLOBAL (CONTROLA AMBAS PESTAÑAS) ---
st.sidebar.header("🎛️ Simulador de Variables")
lluvia = st.sidebar.slider("Precipitaciones en Cabecera (mm)", 0.0, 200.0, 120.0)
temp = st.sidebar.slider("Temperatura Media (°C)", 10.0, 40.0, 22.0)
consumo = st.sidebar.slider("Demanda Agrícola y Urbana (m³)", 5000.0, 25000.0, 14000.0)

nivel_reservorio = min(100.0, max(5.0, (lluvia * 0.55) - (temp * 0.35) - (consumo * 0.0004) + 30.0))

if len(columnas_entrada) == 3:
    datos_usuario = pd.DataFrame([[lluvia, temp, consumo]], columns=columnas_entrada)
else:
    datos_usuario = pd.DataFrame([[lluvia, temp, nivel_reservorio, 2100000.0, consumo]], columns=columnas_entrada)

prediccion = modelo_ia.predict(datos_usuario).item()
probabilidad = modelo_ia.predict_proba(datos_usuario).flatten().tolist().pop() * 100

# --- SEPARACIÓN EN DOS PESTAÑAS INDEPENDIENTES ---
tab_ia, tab_maqueta = st.tabs(["🧠 1. Motor Predictivo (Inteligencia Artificial)", "🏔️ 2. Maqueta Virtual 3D (Gemelo Digital)"])

# ================= PESTAÑA 1: SOLO LA IA Y LOS DATOS =================
with tab_ia:
    st.subheader("Análisis de Riesgo y Toma de Decisiones")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Probabilidad de Escasez", f"{probabilidad:.1f}%")
    m2.metric("Capacidad Estimada Embalse", f"{nivel_reservorio:.0f}%")
    m3.metric("Precipitación Ingresada", f"{lluvia:.0f} mm")
    m4.metric("Temperatura Ingresada", f"{temp:.1f} °C")
    
    st.divider()
    
    col_diag, col_graf = st.columns([1, 1.3])
    
    with col_diag:
        st.markdown("### 📋 Diagnóstico y Plan de Acción")
        if prediccion == 1:
            st.error("🚨 ALERTA CRÍTICA: Riesgo de escasez en los próximos 15 días.")
            st.write("1. **Compuertas:** Cierre parcial en el Reservorio Poechos.")
            st.write("2. **Agricultura:** Activar turnos estrictos de riego en el valle.")
            st.write("3. **Población:** Priorizar abastecimiento urbano y activar pozos de reserva.")
        else:
            st.success("💧 ESTADO ESTABLE: Suministro hídrico garantizado.")
            st.write("1. **Compuertas:** Flujo normal hacia canales principales.")
            st.write("2. **Agricultura:** Riego abastecido sin restricciones.")
            st.write("3. **Monitoreo:** Continuar registro rutinario en estaciones.")
            
    with col_graf:
        st.markdown("### 📊 Tendencia Histórica (Datos Investigados)")
        st.line_chart(df_historico[["Lluvia_mm", "Temperatura_C"]].head(35), height=260)

# ================= PESTAÑA 2: SOLO LA MAQUETA 3D VINCULADA =================
with tab_maqueta:
    st.subheader("Simulación Topográfica 3D de la Cuenca Hidrográfica")
    
    if prediccion == 1:
        st.warning("🔴 **Sincronizado con IA:** La maqueta muestra el descenso crítico del embalse, corte del río aguas abajo y sequía en los cultivos.")
        escala_suelo = [[0.0, "#8c6d46"], [0.4, "#c2a36b"], [0.7, "#8a795d"], [1.0, "#dcdcdc"]]
        color_agua = "rgba(230, 57, 70, 0.85)"
    else:
        st.info("🔵 **Sincronizado con IA:** La maqueta muestra el embalse lleno, flujo continuo por el cauce y cultivos verdes abastecidos.")
        escala_suelo = [[0.0, "#2d6a4f"], [0.4, "#52b788"], [0.7, "#7f5539"], [1.0, "#ffffff"]]
        color_agua = "rgba(0, 150, 255, 0.85)"

    x = np.linspace(0, 12, 80)
    y = np.linspace(0, 8, 60)
    X, Y = np.meshgrid(x, y)

    centro_rio = 4.0 + 0.8 * np.sin(X * 0.8)
    distancia_rio = np.abs(Y - centro_rio)

    montanas = np.maximum(0, (5.5 - X) * 0.7) * (0.6 + 0.4 * np.cos(Y * 1.2))
    laderas = np.maximum(0, (distancia_rio - 0.6) ** 1.6) * 0.35
    cauce = -0.6 * np.exp(-(distancia_rio ** 2) * 3.5)
    muro_represa = np.where((X > 5.3) & (X < 5.7) & (distancia_rio < 2.2), 1.8, 0.0)

    Z_terreno = montanas + laderas + cauce + muro_represa

    altura_embalse = -0.3 + (nivel_reservorio / 100.0) * 1.7
    altura_rio_abajo = -0.45 + (0.35 if prediccion == 0 else 0.02)

    Z_agua = np.where(X <= 5.3, altura_embalse, altura_rio_abajo)
    Z_agua = np.where((Z_agua > Z_terreno) & (muro_represa == 0), Z_agua, np.nan)

    fig = go.Figure()

    fig.add_trace(go.Surface(
        z=Z_terreno, x=X, y=Y,
        colorscale=escala_suelo,
        showscale=False,
        lighting=dict(ambient=0.6, diffuse=0.8, roughness=0.5),
        name="Terreno"
    ))

    fig.add_trace(go.Surface(
        z=Z_agua, x=X, y=Y,
        colorscale=[[0, color_agua], [1, color_agua]],
        showscale=False,
        opacity=0.9,
        name="Agua"
    ))

    fig.add_trace(go.Scatter3d(
        x=[1.5, 4.5, 5.5, 9.5],
        y=[4.0, 4.2, 4.0, 4.0],
        z=[3.8, 2.2, 2.6, 1.2],
        mode="markers+text",
        marker=dict(size=6, color="red" if prediccion == 1 else "cyan"),
        text=[
            "1. Cuenca Alta (Captación)",
            f"2. Embalse ({nivel_reservorio:.0f}%)",
            "3. Represa Poechos",
            "4. Valle Agrícola (ALERTA)" if prediccion == 1 else "4. Valle Agrícola (Óptimo)"
        ],
        textposition="top center",
        textfont=dict(size=12, color="white")
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Cuenca Alta ➔ Valle", showgrid=False, zeroline=False),
            yaxis=dict(title="Ancho Cuenca", showgrid=False, zeroline=False),
            zaxis=dict(title="Elevación", showgrid=False, zeroline=False, range=[-1, 5.5]),
            camera=dict(eye=dict(x=1.4, y=-1.7, z=1.1)),
            bgcolor="rgb(15, 20, 30)"
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=550
    )

    st.plotly_chart(fig, use_container_width=True)
