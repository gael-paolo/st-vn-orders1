import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import io
from datetime import datetime
from google.cloud import storage
from google.oauth2 import service_account
import json

st.set_page_config(page_title="🚗 Análisis de Aprovisionamiento de Vehículos Nissan", layout="wide")
st.title("🚗 Análisis de Aprovisionamiento de Vehículos Nissan")

# --- Configuración de GCP ---
@st.cache_resource
def init_gcp_client():
    """Inicializa el cliente de GCP Storage"""
    try:
        # Intentar obtener credenciales desde secrets de Streamlit
        credentials_dict = st.secrets["gcp_service_account"]
        credentials = service_account.Credentials.from_service_account_info(credentials_dict)
        client = storage.Client(credentials=credentials, project=credentials_dict["project_id"])
        return client
    except Exception as e:
        st.error(f"Error al conectar con GCP: {str(e)}")
        return None

@st.cache_data(ttl=3600)  # Cache por 1 hora
def load_data_from_gcs(bucket_name, file_name):
    """Carga datos desde Google Cloud Storage"""
    try:
        client = init_gcp_client()
        if client is None:
            return None
        
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        
        # Descargar como string y leer con pandas
        data = blob.download_as_string()
        df = pd.read_csv(io.BytesIO(data))
        
        return df
    except Exception as e:
        st.error(f"Error al cargar {file_name}: {str(e)}")
        return None

# --- Usuario ---
usuario = st.sidebar.text_input("Nombre de usuario", value="Usuario")

# --- Configuración fija del bucket (no visible para el usuario) ---
BUCKET_NAME = "bk_vn"
FILE_ORDERS = "nissan/orders/vn_nissan_order.csv"
FILE_COLORS = "nissan/orders/vn_nissan_colors.csv"

# Botón de recarga en sidebar
if st.sidebar.button("🔄 Recargar datos"):
    st.cache_data.clear()
    st.rerun()

# --- Carga de datos desde GCS ---
with st.spinner("📥 Cargando datos desde Google Cloud Storage..."):
    df = load_data_from_gcs(BUCKET_NAME, FILE_ORDERS)
    df_colores = load_data_from_gcs(BUCKET_NAME, FILE_COLORS)

if df is None:
    st.error("❌ No se pudo cargar el archivo de órdenes. Verifica la configuración de GCP y el nombre del bucket.")
    st.info("""
    **Configuración necesaria:**
    1. Crea un archivo `.streamlit/secrets.toml` con las credenciales de GCP
    2. Formato del archivo:
    ```toml
    [gcp_service_account]
    type = "service_account"
    project_id = "tu-proyecto"
    private_key_id = "key-id"
    private_key = "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n"
    client_email = "tu-email@proyecto.iam.gserviceaccount.com"
    client_id = "123456789"
    auth_uri = "https://accounts.google.com/o/oauth2/auth"
    token_uri = "https://oauth2.googleapis.com/token"
    auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
    client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/..."
    ```
    """)
    st.stop()

st.sidebar.success("✅ Datos cargados desde GCS")

if df_colores is not None:
    st.sidebar.success("✅ Datos de colores cargados")

# --- Validación de columnas requeridas ---
required_cols = ['CODIGO', 'ORIGEN', 'Stock', 'RES_IVN', 'RES_TRANS']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    st.error(f"⚠️ Faltan columnas requeridas: {', '.join(missing_cols)}")
    st.stop()

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
# Calcular media con mínimo para evitar división por cero
df['Media'] = df[date_cols].mean(axis=1)
df['Media_Safe'] = df['Media'].clip(lower=0.01)  # Mínimo de 0.01 para cálculos

# Coeficiente de variación protegido
df['Desviacion'] = df[date_cols].std(axis=1)
df['Coef_Variacion'] = np.where(
    df['Media'] > 0.01,
    df['Desviacion'] / df['Media'],
    0  # Si no hay ventas, CV = 0
)

# Stock de seguridad mejorado
df['Stock_Seguridad'] = np.where(
    (df['Media'] > 0.01) & (df['Desviacion'] > 0),
    z * df['Desviacion'] * np.sqrt(df['Lead_Time']),
    0  # Si no hay variabilidad o ventas, SS = 0
)

# Totales con manejo de NaN - CORREGIDO
df['Total_Pedidos'] = df.filter(like='Ped').fillna(0).sum(axis=1)
df['Total_Transito'] = df.filter(like='Trans').fillna(0).sum(axis=1)
df['Total_Reservas'] = df['RES_IVN'].fillna(0) + df['RES_TRANS'].fillna(0)

# Stock Disponible CORREGIDO: Stock + Tránsito + Pedidos - Reservas
df['Stock_Disponible'] = (
    df['Stock'].fillna(0) + 
    df['Total_Transito'].fillna(0) + 
    df['Total_Pedidos'].fillna(0) - 
    df['Total_Reservas'].fillna(0)
)

# Meses de inventario protegido
df['Meses_Inventario'] = np.where(
    df['Media'] > 0.01, 
    df['Stock_Disponible'] / df['Media'], 
    999  # Valor alto si no hay ventas
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

# --- Métricas del producto con colores y barras ---
st.subheader(f"📊 Métricas del producto {sel}")
col1, col2 = st.columns(2)
with col1:
    st.metric("Media de ventas", f"{prod['Media']:.2f}")
    st.metric("Coef. Variación", f"{prod['Coef_Variacion']*100:.2f}%")
    st.write(f"Movimientos 4 meses: {prod['Movimientos_4_Meses']}")
    st.progress(min(prod['Movimientos_4_Meses']/12, 1.0))
    st.write(f"Movimientos 6 meses: {prod['Movimientos_6_Meses']}")
    st.progress(min(prod['Movimientos_6_Meses']/12, 1.0))
with col2:
    st.write(f"Movimientos 12 meses: {prod['Movimientos_12_Meses']}")
    st.progress(min(prod['Movimientos_12_Meses']/12, 1.0))
    tend_color = {'++':'green', '+':'lightgreen', '0':'gray', '-':'red', '--':'darkred'}
    st.markdown(
        f"**Tendencia:** <span style='color:{tend_color.get(prod['Tendencia'],'black')}; font-size:30px'>{prod['Tendencia']}</span>", 
        unsafe_allow_html=True
    )
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
cols_data = [
    ['Stock'], 
    [c for c in df.columns if re.match(r'^Ped', c)],
    [c for c in df.columns if re.match(r'^Trans', c)], 
    ['RES_IVN', 'RES_TRANS']
]
for col_name, col, data_cols in zip(cols_names, cols_inv, cols_data):
    with col:
        st.write(f"**{col_name}**")
        available_cols = [c for c in data_cols if c in df.columns]
        if available_cols:
            st.dataframe(prod[available_cols])

# --- Totales grandes repartidos ---
tot_cols = st.columns(5)
tot_cols[0].metric("Total Stock", f"{prod['Stock']:.0f}")
tot_cols[1].metric("Total Pedido", f"{prod['Total_Pedidos']:.0f}")
tot_cols[2].metric("Total Tránsito", f"{prod['Total_Transito']:.0f}")
tot_cols[3].metric("Total Reservas", f"{prod['Total_Reservas']:.0f}")
tot_cols[4].metric("Stock Disponible", f"{prod['Stock_Disponible']:.0f}")

# --- ALERTAS VISUALES ---
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
        st.error(f"🚨 Menos de 1 mes de inventario: {meses_inv:.1f} meses")
    elif meses_inv > 6:
        st.warning(f"⚠️ Sobreinventario: {meses_inv:.1f} meses")
    else:
        st.success(f"✅ Cobertura adecuada: {meses_inv:.1f} meses")

with alert_col3:
    if prod['Coef_Variacion'] > 1.0:
        st.warning(f"⚠️ Alta variabilidad: CV={prod['Coef_Variacion']*100:.0f}%")
    elif prod['Media'] < 0.1:
        st.info("ℹ️ Producto de baja rotación")
    else:
        st.success("✅ Variabilidad normal")

# --- Órdenes planificadas y sugeridas ---
st.subheader("✍️ Órdenes planificadas y sugeridas (4 meses)")
st.info(f"ℹ️ Lead Time del producto: {lead_time} meses")

orden_cols = st.columns(4)

# Inicializar stock proyectado correctamente con pedidos existentes
stock_proj = prod['Stock_Disponible']

for j in range(4):
    with orden_cols[j]:
        st.markdown(f"### 📅 Mes {j+1}")
        
        # MOS esperado editable por mes
        MOS_val = st.number_input(
            f'MOS objetivo', 
            min_value=1.0, 
            max_value=12.0, 
            step=0.1, 
            value=st.session_state['UserInputs'][sel]['MOS'][j], 
            key=f'MOS_{sel}_{j}'
        )
        st.session_state['UserInputs'][sel]['MOS'][j] = MOS_val

        # Sugerencias dinámicas
        demanda_lead_time = sum(st.session_state['UserInputs'][sel]['Proyecciones'][j:min(j+lead_time, 12)])
        
        mos_sug = max(MOS_val * prod['Media_Safe'] - stock_proj, 0)
        dem_sug = max(demanda_lead_time - stock_proj, 0)
        ss_sug = max(prod['Stock_Seguridad'] - stock_proj, 0)

        # Alertas si las sugerencias difieren mucho
        sugerencias = [mos_sug, dem_sug, ss_sug]
        max_sug = max(sugerencias)
        min_sug = min(sugerencias)
        if max_sug > 0 and (max_sug - min_sug) / max_sug > 0.5:
            st.warning("⚠️ Sugerencias divergentes")

        # Mostrar métricas grandes
        st.metric("💡 Sugerida MOS", f"{mos_sug:.0f}", help="Orden sugerida según MOS")
        st.metric("📊 Sugerida Demanda", f"{dem_sug:.0f}", help=f"Orden según demanda proyectada (LT: {lead_time} meses)")
        st.metric("🛡️ Sugerida SS", f"{ss_sug:.0f}", help="Orden sugerida según stock de seguridad")

        # Selector de orden
        plan_val = st.number_input(
            f'✏️ Orden a colocar', 
            min_value=0, 
            step=1, 
            value=int(mos_sug), 
            key=f'order_{sel}_{j}'
        )
        st.session_state['UserInputs'][sel]['Pedidos'][j] = plan_val

        # Actualizar stock proyectado CORRECTAMENTE
        demanda_mes_actual = st.session_state['UserInputs'][sel]['Proyecciones'][j] if j < 12 else 0
        stock_proj = stock_proj + plan_val - demanda_mes_actual
        
        # Mostrar stock proyectado con alerta
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
    
    # Filtrar datos del producto seleccionado
    df_colores_prod = df_colores[df_colores['MODELO'] == sel].copy()
    
    if len(df_colores_prod) > 0:
        # Definir rangos de días
        rangos_dias = ['0-29', '30-59', '60-89', '90-119', '120-149', '150-179', 
                       '180-209', '210-239', '240-269', '270-299', '300-329', 
                       '330-359', '360-389', '390-419', '420-449', 'Mayor a 450']
        
        # Verificar qué columnas de rangos existen
        rangos_existentes = [r for r in rangos_dias if r in df_colores_prod.columns]
        
        if not rangos_existentes:
            st.warning("⚠️ No se encontraron columnas de rangos de días en el archivo")
        else:
            # Resumen general
            st.subheader(f"📊 Resumen de ventas por color - {sel}")
            
            col_summary1, col_summary2, col_summary3 = st.columns(3)
            
            total_ventas = df_colores_prod['Total'].sum() if 'Total' in df_colores_prod.columns else 0
            num_colores = len(df_colores_prod)
            ventas_rapidas = df_colores_prod[rangos_existentes[:3]].sum().sum()  # 0-89 días
            ventas_lentas = df_colores_prod[rangos_existentes[-3:]].sum().sum() if len(rangos_existentes) >= 3 else 0  # Últimos 3 rangos
            
            with col_summary1:
                st.metric("Total de ventas", f"{total_ventas:.0f}")
            with col_summary2:
                st.metric("Colores disponibles", num_colores)
            with col_summary3:
                pct_rapidas = (ventas_rapidas / total_ventas * 100) if total_ventas > 0 else 0
                st.metric("Ventas rápidas (0-89 días)", f"{pct_rapidas:.1f}%")
            
            # Gráfico de barras apiladas por color
            st.subheader("📈 Distribución de velocidad de venta por color")
            
            # Preparar datos para el gráfico
            df_plot = df_colores_prod[['Sig. Color'] + rangos_existentes].set_index('Sig. Color')
            
            # Crear gráfico de barras apiladas horizontal
            fig_colores = px.bar(
                df_plot.T,
                orientation='h',
                title=f"Días hasta la venta por color - {sel}",
                labels={'value': 'Cantidad de vehículos', 'index': 'Rango de días'},
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            fig_colores.update_layout(
                xaxis_title="Cantidad de vehículos",
                yaxis_title="Rango de días hasta la venta",
                legend_title="Color",
                height=500,
                showlegend=True
            )
            st.plotly_chart(fig_colores, use_container_width=True)
            
            # Tabla detallada con formato condicional
            st.subheader("📋 Detalle por color")
            
            # Calcular métricas adicionales por color
            df_colores_prod['Ventas_Rapidas_0-89'] = df_colores_prod[rangos_existentes[:3]].sum(axis=1)
            df_colores_prod['Ventas_Medias_90-269'] = df_colores_prod[[r for r in rangos_existentes if '90' in r or '120' in r or '150' in r or '180' in r or '210' in r or '240' in r]].sum(axis=1) if len(rangos_existentes) > 3 else 0
            df_colores_prod['Ventas_Lentas_270+'] = df_colores_prod[[r for r in rangos_existentes if '270' in r or '300' in r or '330' in r or '360' in r or '390' in r or '420' in r or 'Mayor' in r]].sum(axis=1)
            
            # Calcular porcentajes
            df_colores_prod['% Rápidas'] = (df_colores_prod['Ventas_Rapidas_0-89'] / df_colores_prod['Total'] * 100).fillna(0)
            df_colores_prod['% Lentas'] = (df_colores_prod['Ventas_Lentas_270+'] / df_colores_prod['Total'] * 100).fillna(0)
            
            # Calcular días promedio ponderado (aproximado)
            dias_promedio = []
            for idx, row in df_colores_prod.iterrows():
                total = 0
                suma_ponderada = 0
                for i, rango in enumerate(rangos_existentes):
                    if rango == 'Mayor a 450':
                        dias_medio = 480  # Asumimos 480 días
                    else:
                        # Extraer el punto medio del rango
                        numeros = [int(n) for n in rango.split('-')]
                        dias_medio = sum(numeros) / len(numeros)
                    
                    cantidad = row[rango]
                    suma_ponderada += dias_medio * cantidad
                    total += cantidad
                
                dias_prom = suma_ponderada / total if total > 0 else 0
                dias_promedio.append(dias_prom)
            
            df_colores_prod['Días_Promedio_Venta'] = dias_promedio
            
            # Ordenar por días promedio (más rápidos primero)
            df_colores_prod_sorted = df_colores_prod.sort_values('Días_Promedio_Venta')
            
            # Mostrar tabla resumen
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
            
            # Formatear columnas
            tabla_resumen['% Rápidas'] = tabla_resumen['% Rápidas'].apply(lambda x: f"{x:.1f}%")
            tabla_resumen['% Lentas'] = tabla_resumen['% Lentas'].apply(lambda x: f"{x:.1f}%")
            tabla_resumen['Días_Promedio_Venta'] = tabla_resumen['Días_Promedio_Venta'].apply(lambda x: f"{x:.0f}")
            
            # Renombrar columnas
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
            
            # Recomendaciones basadas en el análisis
            st.subheader("💡 Recomendaciones")
            
            mejor_color = df_colores_prod_sorted.iloc[0]
            peor_color = df_colores_prod_sorted.iloc[-1]
            
            col_rec1, col_rec2 = st.columns(2)
            
            with col_rec1:
                st.success(f"**✅ Color más rápido:** {mejor_color['Sig. Color']}")
                st.write(f"- Días promedio de venta: {mejor_color['Días_Promedio_Venta']:.0f}")
                st.write(f"- {mejor_color['% Rápidas']:.1f}% vendidos en menos de 90 días")
                st.write("🎯 **Acción:** Priorizar este color en órdenes futuras")
            
            with col_rec2:
                st.warning(f"**⚠️ Color más lento:** {peor_color['Sig. Color']}")
                st.write(f"- Días promedio de venta: {peor_color['Días_Promedio_Venta']:.0f}")
                st.write(f"- {peor_color['% Lentas']:.1f}% vendidos después de 270 días")
                st.write("🎯 **Acción:** Reducir inventario de este color")
            
            # Gráfico de pastel de distribución
            col_pie1, col_pie2 = st.columns(2)
            
            with col_pie1:
                # Distribución por velocidad
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
                    title='Distribución por velocidad de venta',
                    color='Categoría',
                    color_discrete_map={
                        'Rápidas (0-89)': 'green',
                        'Medias (90-269)': 'orange',
                        'Lentas (270+)': 'red'
                    }
                )
                st.plotly_chart(fig_pie_vel, use_container_width=True)
            
            with col_pie2:
                # Top 5 colores por volumen
                top_colores = df_colores_prod_sorted.nlargest(5, 'Total')[['Sig. Color', 'Total']]
                fig_pie_top = px.pie(
                    top_colores,
                    values='Total',
                    names='Sig. Color',
                    title='Top 5 colores por volumen de ventas'
                )
                st.plotly_chart(fig_pie_top, use_container_width=True)
            

            
            # Tabla detallada expandible
            with st.expander("📊 Ver distribución completa por rango de días"):
                tabla_completa = df_colores_prod_sorted[['Sig. Color'] + rangos_existentes + ['Total']].copy()
                st.dataframe(tabla_completa, use_container_width=True, hide_index=True)
    
    else:
        st.info(f"ℹ️ No hay datos de colores para el producto {sel}")
else:
    st.info("💡 Sube un archivo de análisis de colores en el sidebar para ver estadísticas detalladas")

# --- Exportar datos ingresados ---
st.subheader("📥 Descargar datos ingresados")

def generar_excel():
    """Genera archivo Excel con los datos ingresados"""
    try:
        all_data = []
        fecha_export = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for prod_name, vals in st.session_state['UserInputs'].items():
            proy_dates = pd.date_range(start=datetime.today(), periods=12, freq='MS')
            order_dates = pd.date_range(start=datetime.today(), periods=4, freq='MS')
            
            # Proyecciones
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
            
            # Pedidos
            for j, v in enumerate(vals['Pedidos']):
                all_data.append({
                    'Producto': prod_name,
                    'Tipo': 'Pedido',
                    'Mes': order_dates[j].strftime('%Y-%m'),
                    'Valor': int(v),
                    'MOS': vals['MOS'][j],
                    'Usuario': usuario,
                    'Fecha_Exportacion': fecha_export
                })
        
        export_df = pd.DataFrame(all_data)
        
        # Crear archivo Excel en memoria
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            export_df.to_excel(writer, index=False, sheet_name='Datos_Ingresados')
            
            # Acceder a la hoja para dar formato
            worksheet = writer.sheets['Datos_Ingresados']
            
            # Ajustar ancho de columnas
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
        
        return output.getvalue()
    
    except Exception as e:
        st.error(f"Error al generar el archivo: {str(e)}")
        return None

# Botón de descarga
excel_data = generar_excel()
if excel_data:
    st.download_button(
        label="📥 Descargar Orden Generada (.xlsx)",
        data=excel_data,
        file_name=f"Nissan_Order_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.error("No se pudo generar el archivo de exportación")

# --- Footer con información ---
st.markdown("---")
st.caption(f"Usuario: {usuario} | Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")