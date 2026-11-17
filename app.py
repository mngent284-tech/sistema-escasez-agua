import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go

st.set_page_config(page_title="Sistema IA - Cuenca Hidrográfica", page_icon="💧", layout="wide")

st.title("💧 Gemelo Digital de Cuenca Hidrográfica e IA Predictiva")
st.markdown("Simulación topográfica 3D (**Cuenca Alta, Media y Baja**) conectada a Inteligencia Artificial")

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
    st.sidebar.success(f"✅ IA Entrenada con {total_registros} registros")
except Exception:
    st.error("⚠️ Verifica que el archivo 'DATOS.xlsx' esté en tu repositorio.")
    st.stop()

# --- BARRA LATERAL ---
st.sidebar.header("🎛️ Parámetros Climáticos")
lluvia = st.sidebar.slider("Precipitaciones en Cuenca Alta (mm)", 0.0, 200.0, 80.0)
temp = st.sidebar.slider("Temperatura Media (°C)", 10.0, 40.0, 25.0)
consumo = st.sidebar.slider("Demanda Agrícola y Urbana (m³)", 5000.0, 25000.0, 15000.0)

nivel_agua_pct = min(100.0, max(5.0, (lluvia * 0.55) - (temp * 0.4) - (consumo * 0.0005) + 25.0))

if len(columnas_entrada) == 3:
    datos_usuario = pd.DataFrame([[lluvia, temp, consumo]], columns=columnas_entrada)
else:
    datos_usuario = pd.DataFrame([[lluvia, temp, nivel_agua_pct, 2100000.0, consumo]], columns=columnas_entrada)

prediccion = modelo_ia.predict(datos_usuario).item()
probabilidad = modelo_ia.predict_proba(datos_usuario).flatten().tolist().pop() * 100

col1, col2 = st.columns([1, 1.7])

with col1:
    st.subheader("🧠 Diagnóstico de la IA")
    st.metric(label="Riesgo de Escasez", value=f"{probabilidad:.1f}%", delta=f"Nivel Reservorio: {nivel_agua_pct:.1f}%")
    
    if prediccion == 1:
        st.error("🚨 ALERTA CRÍTICA: Déficit hídrico en cuenca media y baja.")
        st.markdown("**Acciones en la Cuenca:**")
        st.write("- Cierre parcial de compuertas en Reservorio.")
        st.write("- Turnos de riego restringidos en valle agrícola.")
        st.write("- Priorización de agua para consumo humano.")
        color_agua = "rgba(235, 64, 52, 0.75)"
    else:
        st.success("💧 CUENCA ESTABLE: Flujo normal hacia el valle.")
        st.markdown("**Acciones en la Cuenca:**")
        st.write("- Almacenamiento óptimo en cuenca media.")
        st.write("- Distribución normal a canales de riego.")
        color_agua = "rgba(0, 180, 255, 0.78)"

with col2:
    st.subheader("🏔️ Maqueta Digital 3D de la Cuenca Hidrográfica")
    
    # Generamos el relieve topográfico 3D (Cuenca Alta -> Valle -> Reservorio -> Costa)
    x = np.linspace(0, 10, 50)
    y = np.linspace(0, 10, 50)
    X_grid, Y_grid = np.meshgrid(x, y)
    
    # Montañas a los lados (valle en forma de V) y pendiente desde cuenca alta (X=0) a baja (X=10)
    pendiente = (10 - X_grid) * 0.45
    paredes_valle = ((Y_grid - 5) ** 2) * 0.35
    represa = np.exp(-((X_grid - 5.5) ** 2) * 4) * 1.2  # Muro de contención del reservorio
    Z_terreno = pendiente + paredes_valle + represa
    
    # Altura dinámica del agua según la simulación
    nivel_cota = 1.2 + (nivel_agua_pct / 100.0) * 3.2
    Z_agua = np.full_like(Z_terreno, nivel_cota)
    Z_agua = np.where(Z_terreno < nivel_cota, nivel_cota, np.nan)

    fig = go.Figure()

    # Capa 1: Relieve montañoso y suelo de la cuenca
    fig.add_trace(go.Surface(
        z=Z_terreno, x=X_grid, y=Y_grid,
        colorscale="Earth",
        showscale=False,
        name="Relieve Cuenca"
    ))

    # Capa 2: Volumen dinámico de agua en el río y reservorio
    fig.add_trace(go.Surface(
        z=Z_agua, x=X_grid, y=Y_grid,
        colorscale=[[0, color_agua], [1, color_agua]],
        showscale=False,
        opacity=0.85,
        name="Nivel de Agua"
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title="Cuenca Alta ➔ Cuenca Baja",
            yaxis_title="Ancho del Valle",
            zaxis_title="Altitud (msnm)",
            camera=dict(eye=dict(x=1.5, y=-1.6, z=1.1))
        ),
        margin=dict(l=0, r=0, b=0, t=10),
        height=480
    )

    st.plotly_chart(fig, use_container_width=True)
