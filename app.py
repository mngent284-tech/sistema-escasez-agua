import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

st.set_page_config(page_title="Gemelo Digital Hidrológico IA", page_icon="💧", layout="wide")

st.title("💧 Sistema Inteligente de Predicción de Escasez de Agua")
st.markdown("Integración de **Machine Learning Predictivo** y **Maqueta Topográfica 3D de Alta Definición**")

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

# --- BARRA LATERAL GLOBAL ---
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

# ================= PESTAÑA 2: MAQUETA 3D AFILADA =================
with tab_maqueta:
    st.subheader("Maqueta Topográfica 3D de Alta Definición")
    
    if prediccion == 1:
        st.warning("🔴 **Sincronizado con IA:** Embalse en nivel crítico, corte de caudal en río y cultivos con estrés hídrico.")
        escala_suelo = [[0.0, "#7f5539"], [0.25, "#b08968"], [0.55, "#9c6644"], [0.85, "#6c584c"], [1.0, "#d6ccc2"]]
        color_agua = "rgba(220, 47, 2, 0.88)"
    else:
        st.info("🔵 **Sincronizado con IA:** Embalse óptimo, flujo continuo hacia el valle y cultivos verdes saludables.")
        escala_suelo = [[0.0, "#1b4332"], [0.25, "#40916c"], [0.55, "#6b705c"], [0.85, "#582f0e"], [1.0, "#f8f9fa"]]
        color_agua = "rgba(0, 119, 182, 0.88)"

    # Malla de alta resolución (160 x 120) para bordes afilados
    x = np.linspace(0, 12, 160)
    y = np.linspace(0, 8, 120)
    X, Y = np.meshgrid(x, y)

    # Cauce afilado del río con curvas naturales
    centro_rio = 4.0 + 0.75 * np.sin(X * 0.9) + 0.25 * np.cos(X * 2.1)
    distancia_rio = np.abs(Y - centro_rio)

    # Cordilleras escarpadas en la cuenca alta (X < 5.2)
    crestas = np.abs(np.sin(X * 1.4 + Y * 0.8)) * 0.65 + np.abs(np.cos(X * 2.5 - Y * 1.9)) * 0.35
    montanas = np.maximum(0, (5.2 - X) * 0.75) * (0.45 + 0.55 * crestas)
    
    # Laderas del cañón con corte definido
    laderas = np.maximum(0, (distancia_rio - 0.45) ** 1.5) * 0.42
    # Canal profundo y afilado del río
    cauce = -0.75 * np.exp(-(distancia_rio ** 2) * 5.5)

    # Terreno base sin la represa
    Z_terreno = montanas + laderas + cauce

    # Represa de concreto como estructura independiente y afilada en X = 5.4 a 5.7
    mascara_presa = (X >= 5.35) & (X <= 5.75) & (distancia_rio <= 2.3)
    Z_presa = np.where(mascara_presa, 1.85, np.nan)

    # Nivel dinámico del agua
    altura_embalse = -0.35 + (nivel_reservorio / 100.0) * 1.85
    altura_rio_abajo = -0.55 + (0.38 if prediccion == 0 else 0.03)

    Z_agua = np.where(X < 5.35, altura_embalse, altura_rio_abajo)
    Z_agua = np.where((Z_agua > Z_terreno) & (~mascara_presa), Z_agua, np.nan)

    fig = go.Figure()

    # 1. Relieve topográfico afilado
    fig.add_trace(go.Surface(
        z=Z_terreno, x=X, y=Y,
        colorscale=escala_suelo,
        showscale=False,
        lighting=dict(ambient=0.5, diffuse=0.9, specular=0.2, roughness=0.4),
        name="Topografía"
    ))

    # 2. Muro de concreto de la Represa (color gris sólido)
    fig.add_trace(go.Surface(
        z=Z_presa, x=X, y=Y,
        colorscale=[[0, "#adb5bd"], [1, "#6c757d"]],
        showscale=False,
        name="Represa Poechos"
    ))

    # 3. Agua dinámica
    fig.add_trace(go.Surface(
        z=Z_agua, x=X, y=Y,
        colorscale=[[0, color_agua], [1, color_agua]],
        showscale=False,
        opacity=0.92,
        name="Agua"
    ))

    # 4. Bloques urbanos y parcelas agrícolas en la cuenca baja (X > 7)
    fig.add_trace(go.Scatter3d(
        x=[8.2, 8.6, 9.0, 8.4, 9.5, 10.2, 10.8],
        y=[2.2, 2.5, 2.1, 2.8, 5.8, 6.2, 5.5],
        z=[0.4, 0.4, 0.4, 0.4, 0.3, 0.3, 0.3],
        mode="markers",
        marker=dict(
            size=[7, 8, 6, 7, 9, 9, 9],
            symbol="square",
            color=["#f8f9fa", "#f8f9fa", "#f8f9fa", "#f8f9fa", 
                   "#e9c46a" if prediccion == 1 else "#2a9d8f",
                   "#e9c46a" if prediccion == 1 else "#2a9d8f",
                   "#e9c46a" if prediccion == 1 else "#2a9d8f"]
        ),
        name="Ciudad y Cultivos"
    ))

    # 5. Etiquetas flotantes en 3D
    fig.add_trace(go.Scatter3d(
        x=[1.5, 4.3, 5.55, 9.8],
        y=[4.0, 4.1, 4.0, 4.0],
        z=[4.2, 2.2, 2.5, 1.3],
        mode="markers+text",
        marker=dict(size=6, color="red" if prediccion == 1 else "cyan"),
        text=[
            "1. Cordillera / Cuenca Alta",
            f"2. Embalse ({nivel_reservorio:.0f}%)",
            "3. Muro Represa",
            "4. Valle y Ciudad (ALERTA)" if prediccion == 1 else "4. Valle y Ciudad (Estable)"
        ],
        textposition="top center",
        textfont=dict(size=12, color="white")
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Cuenca Alta ➔ Valle", showgrid=False, zeroline=False),
            yaxis=dict(title="Ancho Cuenca", showgrid=False, zeroline=False),
            zaxis=dict(title="Elevación", showgrid=False, zeroline=False, range=[-1, 5.5]),
            camera=dict(eye=dict(x=1.45, y=-1.65, z=1.05)),
            bgcolor="rgb(12, 16, 24)"
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=560,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)
