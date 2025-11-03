import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import io
from datetime import datetime
import requests

st.set_page_config(page_title="🚗 Análisis de Aprovisionamiento de Vehículos Nissan", layout="wide")
st.title("🚗 Análisis de Aprovisionamiento de Vehículos Nissan")

# --- Configuración de URLs públicas CON DIAGNÓSTICO MEJORADO ---
@st.cache_data(ttl=3600)  # Cache por 1 hora
def load_data_from_url(url, descripcion="archivo"):

    try:        
        # Hacer request con timeout
        response = requests.get(url, timeout=30)
        response.raise_for_status()
                
        # Intentar parsear CSV
        df = pd.read_csv(io.StringIO(response.text))
        
        return df
        
    except requests.exceptions.Timeout:
        st.error(f"⏱️ **Timeout** al cargar {descripcion} desde {url}")
        st.error("El servidor tardó más de 30 segundos en responder.")
        return None
        
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ **Error HTTP {e.response.status_code}** al cargar {descripcion}")
        if e.response.status_code == 403:
            st.error("🔒 **Acceso Denegado (403)**")
            st.info("""
            **Soluciones:**
            1. Verifica que el archivo sea público en Google Cloud Storage
            2. Comando para hacer público:
            ```bash
            gsutil iam ch allUsers:objectViewer gs://TU_BUCKET/archivo.csv
            ```
            3. O desde la consola web: Bucket → archivo → Permisos → Agregar → allUsers → Storage Object Viewer
            """)
        elif e.response.status_code == 404:
            st.error("📂 **Archivo No Encontrado (404)**")
            st.info("Verifica que la URL sea correcta y que el archivo exista en el bucket.")
        return None
        
    except pd.errors.ParserError as e:
        st.error(f"📄 **Error al parsear CSV**: {str(e)}")
        st.info("El archivo descargado no tiene formato CSV válido.")
        return None
        
    except Exception as e:
        st.error(f"❌ **Error inesperado**: {type(e).__name__}")
        st.error(f"Detalles: {str(e)}")
        return None

# --- DIAGNÓSTICO DE CONEXIÓN (OPCIONAL) ---
if st.sidebar.checkbox("🔧 Modo Diagnóstico Avanzado"):
    st.subheader("🔍 Diagnóstico de Conexión al Bucket")
    
    try:
        url_orders = st.secrets["URL_ORDERS"]
        url_colors = st.secrets["URL_COLORS"]
                
        # Test de conexión detallado
        st.write("### 🔌 Test de Conexión")
        for name, url in [("Orders", url_orders), ("Colors", url_colors)]:
            with st.expander(f"Probando {name}..."):
                try:
                    # HEAD request primero (más rápido)
                    st.write("**1. Verificando accesibilidad...**")
                    response_head = requests.head(url, timeout=10)
                    st.success(f"✅ Status Code: {response_head.status_code}")
                    st.write(f"- Content-Type: `{response_head.headers.get('content-type', 'N/A')}`")
                    st.write(f"- Content-Length: `{response_head.headers.get('content-length', 'N/A')}` bytes")
                    
                    # GET request para ver contenido
                    st.write("**2. Descargando primeras líneas...**")
                    response_get = requests.get(url, timeout=10)
                    primeras_lineas = response_get.text[:500]
                    st.code(primeras_lineas, language="text")
                    
                    st.success(f"✅ {name} es accesible y descargable")
                    
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Error al conectar: {str(e)}")
                    
        st.write("---")
                    
    except KeyError as e:
        st.error(f"❌ **Falta configurar secret**: {str(e)}")
        st.info("""
        **Configura los secrets en Streamlit Cloud:**
        
        1. Ve a tu app en Streamlit Cloud
        2. Settings → Secrets
        3. Agrega:
        ```toml
        URL_ORDERS = "[https://storage.googleapis.com/](https://storage.googleapis.com/)..."
        URL_COLORS = "[https://storage.googleapis.com/](https://storage.googleapis.com/)..."
        ```
        
        **Para desarrollo local**, crea `.streamlit/secrets.toml`:
        ```toml
        URL_ORDERS = "[https://storage.googleapis.com/](https://storage.googleapis.com/)..."
        URL_COLORS = "[https://storage.googleapis.com/](https://storage.googleapis.com/)..."
        ```
        """)
        st.stop()

# --- Usuario ---
usuario = st.sidebar.text_input("Nombre de usuario", value="Usuario")

# --- Verificar que existen los secrets ---
try:
    URL_ORDERS = st.secrets["URL_ORDERS"]
    URL_COLORS = st.secrets["URL_COLORS"]
except KeyError as e:
    st.error(f"❌ **Error de configuración**: Falta el secret {str(e)}")
    st.info("""
    **Configura los secrets:**
    
    En Streamlit Cloud: Settings → Secrets
    
    ```toml
    URL_ORDERS = "[https://storage.googleapis.com/tu-bucket/orders.csv](https://storage.googleapis.com/tu-bucket/orders.csv)"
    URL_COLORS = "[https://storage.googleapis.com/tu-bucket/colors.csv](https://storage.googleapis.com/tu-bucket/colors.csv)"
    ```
    """)
    st.stop()

# Botón de recarga en sidebar
if st.sidebar.button("🔄 Recargar datos"):
    st.cache_data.clear()
    st.rerun()

# --- Carga de datos desde URLs públicas ---
with st.spinner("📥 Cargando datos desde Google Cloud Storage..."):
    df = load_data_from_url(URL_ORDERS, "órdenes")
    df_colores = load_data_from_url(URL_COLORS, "colores")

if df is None:
    st.error("❌ No se pudo cargar el archivo de órdenes.")
    st.info(f"**URL configurada:** {URL_ORDERS}")
    st.info("""
    **Pasos para solucionar:**
    
    1. **Verifica que la URL sea correcta**
       - Formato: `https://storage.googleapis.com/BUCKET_NAME/path/file.csv`
    
    2. **Haz el archivo público en GCP:**
       ```bash
       gsutil iam ch allUsers:objectViewer gs://TU_BUCKET/archivo.csv
       ```
    
    3. **Prueba la URL en tu navegador**
       - Debe descargar el CSV directamente sin pedir login
    
    4. **Verifica el formato del archivo**
       - Debe ser CSV válido con encabezados
    """)
    st.stop()

# --- Mapeo flexible de columnas ---
def map_column_names(df):
    """Mapea nombres de columnas alternativos a los esperados"""
    column_mapping = {}
    
    # Columnas OBLIGATORIAS
    required_map = {
        'CODIGO': ['CODIGO', 'CÓDIGO', 'COD', 'MODELO', 'SKU', 'Producto', 'Código', 'codigo'],
        'ORIGEN': ['ORIGEN', 'FUENTE', 'SOURCE', 'PROCEDENCIA', 'Origen', 'origen'],
        'Stock': ['Stock', 'STOCK', 'INVENTARIO', 'INVENTORY', 'EXISTENCIAS', 'stock']
    }
    
    # Columnas OPCIONALES (pueden o no estar)
    optional_map = {
        'RES_IVN': ['RES_IVN', 'RESERVAS_IVN', 'IVN_RES', 'RESERVA_IVN', 'res_ivn'],
        'RES_TRANS': ['RES_TRANS', 'RESERVAS_TRANS', 'TRANS_RES', 'RESERVA_TRANS', 'res_trans'],
        'RES_PED': ['RES_PED', 'RESERVAS_PED', 'PED_RES', 'RESERVA_PED', 'res_ped']
    }
    
    missing_cols = []
    
    # Procesar columnas obligatorias
    for expected_col, possible_names in required_map.items():
        found = False
        for possible in possible_names:
            if possible in df.columns:
                column_mapping[expected_col] = possible
                found = True
                break
        
        if not found:
            missing_cols.append(expected_col)
            st.sidebar.error(f"❌ No se encontró: {expected_col}")
    
    # Procesar columnas opcionales (no generan error si faltan)
    for expected_col, possible_names in optional_map.items():
        found = False
        for possible in possible_names:
            if possible in df.columns:
                column_mapping[expected_col] = possible
                found = True
                break
            
    return column_mapping, missing_cols

# Aplicar mapeo
column_mapping, missing_cols = map_column_names(df)

if missing_cols:
    st.error(f"⚠️ Faltan columnas requeridas: {', '.join(missing_cols)}")
    
    st.info("""
    **📋 Solución:**
    
    1. **Verifica el archivo CSV fuente:** Asegúrate de que contenga columnas con estos nombres (o similares):
       - `CODIGO` o `MODELO` o `CÓDIGO`
       - `ORIGEN` o `FUENTE` 
       - `Stock` o `STOCK` o `INVENTARIO`
       - `RES_IVN` o `RESERVAS_IVN`
       - `RES_TRANS` o `RESERVAS_TRANS`
    
    2. **Columnas actuales en tu archivo:**
    """)
    st.dataframe(pd.DataFrame({"Columnas Actuales": df.columns.tolist()}))
    
    st.info("""
    3. **Si los nombres son diferentes:**        - Modifica el mapeo en el código 
       - O renombra las columnas en tu CSV
       - O actualiza los datos en la fuente
    """)
    st.stop()

# Renombrar columnas
df = df.rename(columns=column_mapping)

# Crear columnas opcionales con valor 0 si no existen
columnas_opcionales = ['RES_IVN', 'RES_TRANS', 'RES_PED']
for col in columnas_opcionales:
    if col not in df.columns:
        df[col] = 0

st.sidebar.success("✅ Todas las columnas requeridas están disponibles")

if df_colores is not None:
    st.sidebar.success("✅ Datos de colores cargados")
else:
    st.sidebar.warning("⚠️ No se pudieron cargar los datos de colores")

# --- Columnas de fechas ---
date_cols = [c for c in df.columns if re.match(r'^\d{4}-\d{2}-\d{2}$', str(c))]
date_cols = sorted(date_cols, key=lambda x: pd.to_datetime(x))
if not date_cols:
    st.error("No se detectaron columnas de fechas (YYYY-MM-DD).")
    st.stop()
num_months = st.sidebar.slider("Cantidad de meses a mostrar", 6, len(date_cols), min(12, len(date_cols)))
date_cols = date_cols[-num_months:]

# --- Lead time por ORIGEN ---
def get_lead_time(origen):
    return {'NMEX':2,'NTE':3,'NTJ':4}.get(origen,3)
df['Lead_Time'] = df['ORIGEN'].apply(get_lead_time)

# --- Nivel de servicio ---
nivel_servicio = st.sidebar.selectbox("Nivel de servicio (%)", options=[80,85,90,95,97.5,99], index=3)
z_dict = {80:0.84,85:1.04,90:1.28,95:1.65,97.5:1.96,99:2.33}
z = z_dict[nivel_servicio]

# --- Métricas base con protección matemática ---
df['Media'] = df[date_cols].mean(axis=1)
df['Media_Safe'] = df['Media'].clip(lower=0.01)

df['Desviacion'] = df[date_cols].std(axis=1)
df['Coef_Variacion'] = np.where(
    df['Media'] > 0.01,
    df['Desviacion'] / df['Media'],
    0
)

df['Stock_Seguridad'] = np.where(
    (df['Media'] > 0.01) & (df['Desviacion'] > 0),
    z * df['Desviacion'] * np.sqrt(df['Lead_Time']),
    0
)

df['Total_Pedidos'] = df.filter(like='Ped').fillna(0).sum(axis=1)
df['Total_Transito'] = df.filter(like='Trans').fillna(0).sum(axis=1)

# Total de reservas considerando las columnas que existan
df['Total_Reservas'] = 0
if 'RES_IVN' in df.columns:
    df['Total_Reservas'] += df['RES_IVN'].fillna(0)
if 'RES_TRANS' in df.columns:
    df['Total_Reservas'] += df['RES_TRANS'].fillna(0)
if 'RES_PED' in df.columns:
    df['Total_Reservas'] += df['RES_PED'].fillna(0)

df['Stock_Disponible'] = (
    df['Stock'].fillna(0) + 
    df['Total_Transito'].fillna(0) + 
    df['Total_Pedidos'].fillna(0) - 
    df['Total_Reservas'].fillna(0)
)

df['Meses_Inventario'] = np.where(
    df['Media'] > 0.01, 
    df['Stock_Disponible'] / df['Media'], 
    999
)

# --- Selección de familia y producto ---
df['FAMILIA'] = df['CODIGO'].str[:3]
familias = sorted(df['FAMILIA'].unique().tolist())
selected_fam = st.sidebar.selectbox("Selecciona familia", familias)
productos = sorted(df[df['FAMILIA']==selected_fam]['CODIGO'].unique().tolist())
sel = st.selectbox("Selecciona un producto", productos)
prod = df[df['CODIGO']==sel].iloc[0]
lead_time = int(prod['Lead_Time'])

# --- Inicialización de inputs ---
if 'UserInputs' not in st.session_state:
    st.session_state['UserInputs'] = {}
# **SUGERENCIA 1: Inicialización con 0. Usamos el 0 internamente para que number_input funcione.**
# La diferenciación con "NO REVISADO" se hace en la función de exportación.
if sel not in st.session_state['UserInputs']:
    hist_mean = int(prod[date_cols].mean()) if not np.isnan(prod[date_cols].mean()) else 0
    st.session_state['UserInputs'][sel] = {
        'Proyecciones': [hist_mean]*12, 
        'Pedidos': [0]*4, 
        'MOS': [2.0]*4
    }

# --- Gráfico histórico + proyección ---
hist = prod[date_cols].T.reset_index()
hist.columns = ['Fecha','Ventas']
hist['Fecha'] = pd.to_datetime(hist['Fecha'])
proy_fechas = pd.date_range(start=hist['Fecha'].max() + pd.offsets.MonthBegin(), periods=12, freq='MS')
proy = pd.DataFrame({
    'Fecha': proy_fechas, 
    'Proyección': st.session_state['UserInputs'][sel]['Proyecciones']
})

fig = px.line()
fig.add_scatter(x=hist['Fecha'], y=hist['Ventas'], mode='lines+markers', name='Histórico', line=dict(color='blue'))
fig.add_scatter(x=proy['Fecha'], y=proy['Proyección'], mode='lines+markers', name='Proyección', line=dict(color='orange', dash='dash'))
fig.update_layout(
    title=f"Serie de tiempo: {sel}", 
    xaxis_title="Mes", 
    yaxis_title="Unidades",
    hovermode='x unified'
)
st.plotly_chart(fig, use_container_width=True)

# --- Ventas proyectadas ---
st.subheader("✍️ Ventas proyectadas (12 meses)")
cols_proj = st.columns(4)
for i in range(12):
    with cols_proj[i%4]:
        val = st.number_input(
            f'Mes {i+1}', 
            min_value=0, 
            step=1, 
            value=int(st.session_state['UserInputs'][sel]['Proyecciones'][i]), 
            key=f'proj_{sel}_{i}'
        )
        st.session_state['UserInputs'][sel]['Proyecciones'][i] = val

# --- Métricas del producto ---
st.subheader(f"📊 Métricas del producto {sel}")
col1, col2 = st.columns(2)
with col1:
    st.metric("Media de ventas", f"{prod['Media']:.2f}")
    st.metric("Coef. Variación", f"{prod['Coef_Variacion']*100:.2f}%")
    
    if 'Movimientos_4_Meses' in prod:
        st.write(f"Movimientos 4 meses: {prod['Movimientos_4_Meses']}")
        st.progress(min(prod['Movimientos_4_Meses']/12, 1.0))
    if 'Movimientos_6_Meses' in prod:
        st.write(f"Movimientos 6 meses: {prod['Movimientos_6_Meses']}")
        st.progress(min(prod['Movimientos_6_Meses']/12, 1.0))
        
with col2:
    if 'Movimientos_12_Meses' in prod:
        st.write(f"Movimientos 12 meses: {prod['Movimientos_12_Meses']}")
        st.progress(min(prod['Movimientos_12_Meses']/12, 1.0))
    
    if 'Tendencia' in prod:
        tend_color = {'++':'green', '+':'lightgreen', '0':'gray', '-':'red', '--':'darkred'}
        st.markdown(
            f"**Tendencia:** <span style='color:{tend_color.get(prod['Tendencia'],'black')}; font-size:30px'>{prod['Tendencia']}</span>", 
            unsafe_allow_html=True
        )
    if 'ESTRAT' in prod:
        estrat_color = {'A':'green', 'B':'gray', 'C':'yellow', 'D':'deepskyblue', 'E':'red'}
        st.markdown(
            f"**Estratificación:** <span style='color:{estrat_color.get(prod['ESTRAT'],'black')}; font-size:30px'>{prod['ESTRAT']}</span>", 
            unsafe_allow_html=True
        )
    
    st.metric("Stock de seguridad", f"{prod['Stock_Seguridad']:.0f}")
    st.metric("Stock Disponible", f"{prod['Stock_Disponible']:.0f}")

# --- Inventario Detallado ---
st.subheader("📦 Inventario Detallado")
cols_inv = st.columns(4)
cols_names = ['Stock', 'Pedidos', 'Tránsito', 'Reservas']

# Definir columnas para cada categoría
cols_data = [
    ['Stock'], 
    [c for c in df.columns if re.match(r'^Ped', c)],
    [c for c in df.columns if re.match(r'^Trans', c)], 
    []  # Reservas se construyen dinámicamente
]

# Agregar columnas de reservas que existan
reservas_cols = []
for res_col in ['RES_IVN', 'RES_TRANS', 'RES_PED']:
    if res_col in df.columns:
        reservas_cols.append(res_col)
cols_data[3] = reservas_cols

for col_name, col, data_cols in zip(cols_names, cols_inv, cols_data):
    with col:
        st.write(f"**{col_name}**")
        available_cols = [c for c in data_cols if c in df.columns]
        if available_cols:
            st.dataframe(prod[available_cols])
        else:
            st.info(f"No hay datos de {col_name}")

# --- Totales ---
tot_cols = st.columns(5)
tot_cols[0].metric("Total Stock", f"{prod['Stock']:.0f}")
tot_cols[1].metric("Total Pedido", f"{prod['Total_Pedidos']:.0f}")
tot_cols[2].metric("Total Tránsito", f"{prod['Total_Transito']:.0f}")
tot_cols[3].metric("Total Reservas", f"{prod['Total_Reservas']:.0f}")
tot_cols[4].metric("Stock Disponible", f"{prod['Stock_Disponible']:.0f}")

# --- ALERTAS ---
st.subheader("⚠️ Alertas de Inventario")
alert_col1, alert_col2, alert_col3 = st.columns(3)

with alert_col1:
    if prod['Stock_Disponible'] < prod['Stock_Seguridad']:
        st.error(f"🚨 Stock bajo seguridad: {prod['Stock_Disponible']:.0f} < {prod['Stock_Seguridad']:.0f}")
    elif prod['Stock_Disponible'] < prod['Stock_Seguridad'] * 1.5:
        st.warning(f"⚠️ Stock cerca del mínimo: {prod['Stock_Disponible']:.0f}")
    else:
        st.success(f"✅ Stock saludable: {prod['Stock_Disponible']:.0f}")

with alert_col2:
    meses_inv = prod['Meses_Inventario']
    if meses_inv < 1:
        st.error(f"🚨 Menos de 1 mes: {meses_inv:.1f} meses")
    elif meses_inv > 6:
        st.warning(f"⚠️ Sobreinventario: {meses_inv:.1f} meses")
    else:
        st.success(f"✅ Cobertura: {meses_inv:.1f} meses")

with alert_col3:
    if prod['Coef_Variacion'] > 1.0:
        st.warning(f"⚠️ Alta variabilidad: CV={prod['Coef_Variacion']*100:.0f}%")
    elif prod['Media'] < 0.1:
        st.info("ℹ️ Baja rotación")
    else:
        st.success("✅ Variabilidad normal")

# --- Órdenes planificadas ---
st.subheader("✍️ Órdenes planificadas y sugeridas (4 meses)")
st.info(f"ℹ️ Lead Time: {lead_time} meses")

orden_cols = st.columns(4)
stock_proj = prod['Stock_Disponible']

for j in range(4):
    with orden_cols[j]:
        st.markdown(f"### 📅 Mes {j+1}")
        
        MOS_val = st.number_input(
            f'MOS objetivo', 
            min_value=1.0, 
            max_value=12.0, 
            step=0.1, 
            value=st.session_state['UserInputs'][sel]['MOS'][j], 
            key=f'MOS_{sel}_{j}'
        )
        st.session_state['UserInputs'][sel]['MOS'][j] = MOS_val

        demanda_lead_time = sum(st.session_state['UserInputs'][sel]['Proyecciones'][j:min(j+lead_time, 12)])
        
        mos_sug = max(MOS_val * prod['Media_Safe'] - stock_proj, 0)
        dem_sug = max(demanda_lead_time - stock_proj, 0)
        ss_sug = max(prod['Stock_Seguridad'] - stock_proj, 0)

        sugerencias = [mos_sug, dem_sug, ss_sug]
        max_sug = max(sugerencias)
        min_sug = min(sugerencias)
        if max_sug > 0 and (max_sug - min_sug) / max_sug > 0.5:
            st.warning("⚠️ Sugerencias divergentes")

        st.metric("💡 Sugerida MOS", f"{mos_sug:.0f}")
        st.metric("📊 Sugerida Demanda", f"{dem_sug:.0f}")
        st.metric("🛡️ Sugerida SS", f"{ss_sug:.0f}")

        plan_val = st.number_input(
            f'✏️ Orden a colocar', 
            min_value=0, 
            step=1, 
            # Usamos el valor guardado, que es 0 por defecto si no ha sido modificado
            value=int(st.session_state['UserInputs'][sel]['Pedidos'][j]), 
            key=f'order_{sel}_{j}'
        )
        st.session_state['UserInputs'][sel]['Pedidos'][j] = plan_val

        demanda_mes_actual = st.session_state['UserInputs'][sel]['Proyecciones'][j] if j < 12 else 0
        stock_proj = stock_proj + plan_val - demanda_mes_actual
        
        if stock_proj < 0:
            st.error(f"🚨 Stock proyectado: **{stock_proj:.0f}**")
        elif stock_proj < prod['Stock_Seguridad']:
            st.warning(f"⚠️ Stock proyectado: **{stock_proj:.0f}**")
        else:
            st.success(f"✅ Stock proyectado: **{stock_proj:.0f}**")

# --- ANÁLISIS DE COLORES ---
if df_colores is not None:
    st.markdown("---")
    st.title("🎨 Análisis de Velocidad de Venta por Color")
    
    if 'MODELO' not in df_colores.columns:
        for col in ['MODELO', 'CODIGO', 'CÓDIGO', 'COD', 'Producto']:
            if col in df_colores.columns:
                df_colores = df_colores.rename(columns={col: 'MODELO'})
                break
    
    df_colores_prod = df_colores[df_colores['MODELO'] == sel].copy()
    
    if len(df_colores_prod) > 0:
        rangos_dias = ['0-29', '30-59', '60-89', '90-119', '120-149', '150-179', 
                       '180-209', '210-239', '240-269', '270-299', '300-329', 
                       '330-359', '360-389', '390-419', '420-449', 'Mayor a 450']
        
        rangos_existentes = [r for r in rangos_dias if r in df_colores_prod.columns]
        
        if not rangos_existentes:
            st.warning("⚠️ No se encontraron columnas de rangos")
        else:
            st.subheader(f"📊 Resumen - {sel}")
            
            col_summary1, col_summary2, col_summary3 = st.columns(3)
            
            total_ventas = df_colores_prod['Total'].sum() if 'Total' in df_colores_prod.columns else 0
            num_colores = len(df_colores_prod)
            ventas_rapidas = df_colores_prod[rangos_existentes[:3]].sum().sum()
            ventas_lentas = df_colores_prod[rangos_existentes[-3:]].sum().sum() if len(rangos_existentes) >= 3 else 0
            
            with col_summary1:
                st.metric("Total ventas", f"{total_ventas:.0f}")
            with col_summary2:
                st.metric("Colores", num_colores)
            with col_summary3:
                pct_rapidas = (ventas_rapidas / total_ventas * 100) if total_ventas > 0 else 0
                st.metric("Ventas rápidas (0-89)", f"{pct_rapidas:.1f}%")
            
            st.subheader("📈 Distribución por color")
            
            df_plot = df_colores_prod[['Sig. Color'] + rangos_existentes].set_index('Sig. Color')
            
            fig_colores = px.bar(
                df_plot.T,
                orientation='h',
                title=f"Días hasta venta - {sel}",
                labels={'value': 'Vehículos', 'index': 'Rango días'},
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            fig_colores.update_layout(
                xaxis_title="Cantidad",
                yaxis_title="Rango días",
                legend_title="Color",
                height=500
            )
            st.plotly_chart(fig_colores, use_container_width=True)
            
            st.subheader("📋 Detalle por color")
            
            df_colores_prod['Ventas_Rapidas_0-89'] = df_colores_prod[rangos_existentes[:3]].sum(axis=1)
            df_colores_prod['Ventas_Medias_90-269'] = df_colores_prod[[r for r in rangos_existentes if any(x in r for x in ['90','120','150','180','210','240'])]].sum(axis=1)
            df_colores_prod['Ventas_Lentas_270+'] = df_colores_prod[[r for r in rangos_existentes if any(x in r for x in ['270','300','330','360','390','420','Mayor'])]].sum(axis=1)
            
            df_colores_prod['% Rápidas'] = (df_colores_prod['Ventas_Rapidas_0-89'] / df_colores_prod['Total'] * 100).fillna(0)
            df_colores_prod['% Lentas'] = (df_colores_prod['Ventas_Lentas_270+'] / df_colores_prod['Total'] * 100).fillna(0)
            
            dias_promedio = []
            for idx, row in df_colores_prod.iterrows():
                total = 0
                suma_ponderada = 0
                for i, rango in enumerate(rangos_existentes):
                    if rango == 'Mayor a 450':
                        dias_medio = 480
                    else:
                        numeros = [int(n) for n in rango.split('-')]
                        dias_medio = sum(numeros) / len(numeros)
                    
                    cantidad = row[rango]
                    suma_ponderada += dias_medio * cantidad
                    total += cantidad
                
                dias_prom = suma_ponderada / total if total > 0 else 0
                dias_promedio.append(dias_prom)
            
            df_colores_prod['Días_Promedio_Venta'] = dias_promedio
            
            df_colores_prod_sorted = df_colores_prod.sort_values('Días_Promedio_Venta')
            
            tabla_resumen = df_colores_prod_sorted[[
                'Sig. Color', 
                'Total', 
                'Ventas_Rapidas_0-89', 
                'Ventas_Medias_90-269',
                'Ventas_Lentas_270+',
                '% Rápidas',
                '% Lentas',
                'Días_Promedio_Venta'
            ]].copy()
            
            tabla_resumen['% Rápidas'] = tabla_resumen['% Rápidas'].apply(lambda x: f"{x:.1f}%")
            tabla_resumen['% Lentas'] = tabla_resumen['% Lentas'].apply(lambda x: f"{x:.1f}%")
            tabla_resumen['Días_Promedio_Venta'] = tabla_resumen['Días_Promedio_Venta'].apply(lambda x: f"{x:.0f}")
            
            tabla_resumen.columns = [
                'Color', 
                'Total Ventas', 
                'Ventas Rápidas (0-89)', 
                'Ventas Medias (90-269)',
                'Ventas Lentas (270+)',
                '% Rápidas',
                '% Lentas',
                'Días Prom.'
            ]
            
            st.dataframe(tabla_resumen, use_container_width=True, hide_index=True)
            
            st.subheader("💡 Recomendaciones")
            
            mejor_color = df_colores_prod_sorted.iloc[0]
            peor_color = df_colores_prod_sorted.iloc[-1]
            
            col_rec1, col_rec2 = st.columns(2)
            
            with col_rec1:
                st.success(f"**✅ Color más rápido:** {mejor_color['Sig. Color']}")
                st.write(f"- Días promedio: {mejor_color['Días_Promedio_Venta']:.0f}")
                st.write(f"- {mejor_color['% Rápidas']:.1f}% en < 90 días")
                st.write("🎯 **Acción:** Priorizar en órdenes")
            
            with col_rec2:
                st.warning(f"**⚠️ Color más lento:** {peor_color['Sig. Color']}")
                st.write(f"- Días promedio: {peor_color['Días_Promedio_Venta']:.0f}")
                st.write(f"- {peor_color['% Lentas']:.1f}% después de 270 días")
                st.write("🎯 **Acción:** Reducir inventario")
            
            col_pie1, col_pie2 = st.columns(2)
            
            with col_pie1:
                dist_velocidad = pd.DataFrame({
                    'Categoría': ['Rápidas (0-89)', 'Medias (90-269)', 'Lentas (270+)'],
                    'Cantidad': [
                        df_colores_prod['Ventas_Rapidas_0-89'].sum(),
                        df_colores_prod['Ventas_Medias_90-269'].sum(),
                        df_colores_prod['Ventas_Lentas_270+'].sum()
                    ]
                })
                fig_pie_vel = px.pie(
                    dist_velocidad, 
                    values='Cantidad', 
                    names='Categoría',
                    title='Distribución por velocidad',
                    color='Categoría',
                    color_discrete_map={
                        'Rápidas (0-89)': 'green',
                        'Medias (90-269)': 'orange',
                        'Lentas (270+)': 'red'
                    }
                )
                st.plotly_chart(fig_pie_vel, use_container_width=True)
            
            with col_pie2:
                top_colores = df_colores_prod_sorted.nlargest(5, 'Total')[['Sig. Color', 'Total']]
                fig_pie_top = px.pie(
                    top_colores,
                    values='Total',
                    names='Sig. Color',
                    title='Top 5 colores por volumen'
                )
                st.plotly_chart(fig_pie_top, use_container_width=True)
    
    else:
        st.info(f"ℹ️ No hay datos de colores para {sel}")
else:
    st.info("💡 No se cargaron datos de colores")

# --- Exportar datos ---
st.subheader("📥 Descargar datos ingresados")

# **SUGERENCIA 2: LÓGICA DE EXPORTACIÓN MODIFICADA**
def generar_excel():
    """Genera archivo Excel con los datos ingresados y marca como 'NO REVISADO' los productos no manipulados."""
    try:
        all_data = []
        fecha_export = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Obtener todos los productos de la familia actualmente seleccionada
        df_familia = df[df['FAMILIA']==selected_fam].copy()
        productos_familia = df_familia['CODIGO'].unique().tolist()

        # Iterar sobre TODOS los productos de la familia, NO solo los del session_state
        for prod_name in productos_familia:
            
            es_manipulado = prod_name in st.session_state['UserInputs']
            
            # --- Proyecciones (Solo se exportan si fueron manipuladas) ---
            if es_manipulado:
                vals = st.session_state['UserInputs'][prod_name]
                proy_dates = pd.date_range(start=datetime.today(), periods=12, freq='MS')
                for i, v in enumerate(vals['Proyecciones']):
                    all_data.append({
                        'Producto': prod_name,
                        'Tipo': 'Proyección',
                        'Mes': proy_dates[i].strftime('%Y-%m'),
                        'Valor': int(v),
                        'MOS': None,
                        'Usuario': usuario,
                        'Fecha_Exportacion': fecha_export
                    })

            # --- Pedidos (Aprovisionamiento) ---
            order_dates = pd.date_range(start=datetime.today(), periods=4, freq='MS')
            
            for j in range(4):
                
                if es_manipulado:
                    pedido_val = int(st.session_state['UserInputs'][prod_name]['Pedidos'][j])
                    mos_val = st.session_state['UserInputs'][prod_name]['MOS'][j]
                else:
                    # Asignamos el valor indicador si no fue revisado
                    pedido_val = 'NO REVISADO'
                    mos_val = 'NO REVISADO'
                
                all_data.append({
                    'Producto': prod_name,
                    'Tipo': 'Pedido',
                    'Mes': order_dates[j].strftime('%Y-%m'),
                    'Valor': pedido_val,
                    'MOS': mos_val,
                    'Usuario': usuario,
                    'Fecha_Exportacion': fecha_export
                })
        
        export_df = pd.DataFrame(all_data)
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            export_df.to_excel(writer, index=False, sheet_name='Datos_Ingresados')
            
            worksheet = writer.sheets['Datos_Ingresados']
            
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        # Ajuste para manejar el valor no numérico 'NO REVISADO'
                        value_to_check = str(cell.value) if cell.value is not None else ''
                        if len(value_to_check) > max_length:
                            max_length = len(value_to_check)
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        return output.getvalue()
    
    except Exception as e:
        st.error(f"Error al generar archivo: {str(e)}")
        return None

excel_data = generar_excel()
if excel_data:
    st.download_button(
        label="📥 Descargar Orden (.xlsx)",
        data=excel_data,
        file_name=f"Nissan_Order_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.error("No se pudo generar archivo")

st.markdown("---")
st.caption(f"👤 {usuario} | 📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")