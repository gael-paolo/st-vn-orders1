import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import io
from datetime import datetime, timedelta
import requests
import json
import warnings
from google.cloud import storage
from google.oauth2 import service_account
warnings.filterwarnings('ignore')

st.set_page_config(page_title="🚗 Análisis de Aprovisionamiento de Vehículos Nissan", layout="wide")
st.title("🚗 Análisis de Aprovisionamiento de Vehículos Nissan")

# --- Configuración de Google Cloud Storage con autenticación por JSON ---
def get_gcp_client():
    """Inicializa el cliente de Google Cloud Storage con credenciales desde secrets"""
    try:
        # Obtener credenciales desde Streamlit secrets
        if 'gcp_service_account' in st.secrets:
            service_account_info = dict(st.secrets['gcp_service_account'])
            credentials = service_account.Credentials.from_service_account_info(service_account_info)
            client = storage.Client(credentials=credentials)
            return client
        else:
            st.error("❌ No se encontraron las credenciales de GCP en los secrets")
            return None
    except Exception as e:
        st.error(f"❌ Error al inicializar cliente GCP: {str(e)}")
        return None

# --- Configuración mejorada de GCP ---
BUCKET_NAME = "bk_vn"  # Nombre de tu bucket

def get_user_filename(usuario):
    """Genera el nombre de archivo para el usuario"""
    current_month = datetime.now().strftime("%m_%Y")
    safe_username = re.sub(r'[^a-zA-Z0-9_]', '_', usuario)
    return f"nissan/orders/users/{current_month}_{safe_username}.csv"

def save_user_data(usuario, user_data):
    """Guarda los datos del usuario en GCP usando la API de Google Cloud Storage"""
    try:
        if not usuario or usuario == "Usuario":
            st.warning("⚠️ Nombre de usuario no válido para guardar")
            return False
            
        client = get_gcp_client()
        if client is None:
            st.error("❌ No se pudo inicializar el cliente de GCP")
            return False
            
        filename = get_user_filename(usuario)
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(filename)
        
        # Convertir datos a CSV optimizado
        records = []
        timestamp = datetime.now().isoformat()
        
        for producto, datos in user_data.items():
            if 'Proyecciones' in datos:
                for i, proyeccion in enumerate(datos['Proyecciones']):
                    records.append({
                        'usuario': usuario,
                        'producto': producto,
                        'tipo': 'proyeccion',
                        'mes': i,
                        'valor': proyeccion,
                        'mos_objetivo': None,
                        'fecha_actualizacion': timestamp
                    })
            
            if 'Pedidos' in datos:
                for i, (pedido, mos) in enumerate(zip(datos['Pedidos'], datos.get('MOS', [4.0]*4))):
                    records.append({
                        'usuario': usuario,
                        'producto': producto,
                        'tipo': 'pedido',
                        'mes_orden': i,
                        'valor': pedido,
                        'mos_objetivo': mos,
                        'fecha_actualizacion': timestamp
                    })
        
        if not records:
            st.warning("No hay datos para guardar")
            return False
            
        df_save = pd.DataFrame(records)
        
        # Convertir a CSV en memoria
        csv_buffer = io.StringIO()
        df_save.to_csv(csv_buffer, index=False)
        csv_content = csv_buffer.getvalue()
        
        # Subir a GCP
        blob.upload_from_string(csv_content, content_type='text/csv')
        
        st.success(f"💾 Datos guardados exitosamente para {usuario}")
        st.info(f"📍 Archivo guardado en: gs://{BUCKET_NAME}/{filename}")
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error al guardar datos en GCP: {str(e)}")
        return False

def load_user_data(usuario):
    """Carga los datos del usuario desde GCP usando la API de Google Cloud Storage"""
    try:
        if not usuario or usuario == "Usuario":
            return None
            
        client = get_gcp_client()
        if client is None:
            return None
            
        filename = get_user_filename(usuario)
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(filename)
        
        # Verificar si el archivo existe
        if not blob.exists():
            return None
        
        # Descargar contenido
        content = blob.download_as_text()
        df_loaded = pd.read_csv(io.StringIO(content))
        
        st.sidebar.success("✅ Datos de usuario cargados desde GCP")
        return df_loaded
        
    except Exception as e:
        st.sidebar.info(f"ℹ️ No se encontraron datos previos del usuario: {str(e)}")
        return None

# --- Configuración optimizada para cargar datos base desde GCP usando la API ---
@st.cache_data(ttl=3600, show_spinner=False)
def load_data_from_gcp(file_path, descripcion="archivo"):
    """Carga datos desde Google Cloud Storage usando la API"""
    try:
        client = get_gcp_client()
        if client is None:
            st.error("❌ No se pudo inicializar el cliente de GCP")
            return None
            
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(file_path)
        
        # Verificar si el archivo existe
        if not blob.exists():
            st.error(f"❌ El archivo {file_path} no existe en el bucket")
            return None
        
        # Descargar contenido
        content = blob.download_as_text()
        df = pd.read_csv(io.StringIO(content))
        
        st.sidebar.success(f"✅ {descripcion} cargados desde GCP")
        return df
        
    except Exception as e:
        st.error(f"❌ Error al cargar {descripcion} desde GCP: {str(e)}")
        return None

# Función de respaldo por si falla la carga desde GCP (usa URLs públicas)
@st.cache_data(ttl=3600, show_spinner=False)
def load_data_from_url(url, descripcion="archivo"):
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        st.sidebar.success(f"✅ {descripcion} cargados desde URL")
        return df
    except Exception as e:
        st.error(f"❌ Error al cargar {descripcion}: {str(e)}")
        return None

# --- Gestión de estado mejorada ---
def initialize_session_state():
    """Inicializa y gestiona el estado de la sesión"""
    if 'UserInputs' not in st.session_state:
        st.session_state.UserInputs = {}
    if 'last_save' not in st.session_state:
        st.session_state.last_save = datetime.now()
    if 'current_product_index' not in st.session_state:
        st.session_state.current_product_index = 0
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'calculations_cache' not in st.session_state:
        st.session_state.calculations_cache = {}
    if 'recalculate_orders' not in st.session_state:
        st.session_state.recalculate_orders = False
    if 'selected_family' not in st.session_state:
        st.session_state.selected_family = 'Todas'
    if 'selected_strat' not in st.session_state:
        st.session_state.selected_strat = 'Todas'
    if 'product_key' not in st.session_state:
        st.session_state.product_key = 0

initialize_session_state()

# --- Usuario y carga de datos optimizada ---
usuario = st.sidebar.text_input("Nombre de usuario", value="Usuario")

# Cargar datos existentes del usuario solo si es necesario
if not st.session_state.data_loaded:
    user_existing_data = load_user_data(usuario)
    st.session_state.data_loaded = True
else:
    user_existing_data = None

# Paths de datos en GCP
GCP_ORDERS_PATH = "nissan/orders/vn_nissan_order.csv"
GCP_COLORS_PATH = "nissan/orders/vn_nissan_colors.csv"

# URLs de respaldo (por si falla la carga desde GCP)
URL_ORDERS = "https://storage.googleapis.com/bk_vn/nissan/orders/vn_nissan_order.csv"
URL_COLORS = "https://storage.googleapis.com/bk_vn/nissan/orders/vn_nissan_colors.csv"

# Botón de recarga
if st.sidebar.button("🔄 Recargar datos"):
    st.cache_data.clear()
    st.session_state.data_loaded = False
    st.session_state.calculations_cache = {}
    st.session_state.current_product_index = 0
    st.session_state.product_key += 1
    st.rerun()

# Carga de datos optimizada - Primero intenta desde GCP, luego desde URL
@st.cache_data(ttl=3600, show_spinner=False)
def load_all_data():
    with st.spinner("📥 Cargando datos desde Google Cloud Storage..."):
        # Intentar cargar desde GCP primero
        df = load_data_from_gcp(GCP_ORDERS_PATH, "órdenes")
        df_colores = load_data_from_gcp(GCP_COLORS_PATH, "colores")
        
        # Si falla la carga desde GCP, usar URLs de respaldo
        if df is None:
            st.warning("⚠️ Intentando cargar datos desde URLs de respaldo...")
            df = load_data_from_url(URL_ORDERS, "órdenes")
        
        if df_colores is None:
            df_colores = load_data_from_url(URL_COLORS, "colores")
            
    return df, df_colores

df, df_colores = load_all_data()

if df is None:
    st.error("❌ No se pudo cargar el archivo de órdenes.")
    st.stop()

# --- Mapeo flexible de columnas mejorado ---
def map_column_names(df):
    column_mapping = {}
    required_map = {
        'CODIGO': ['CODIGO', 'CÓDIGO', 'COD', 'MODELO', 'SKU', 'Producto', 'Código', 'codigo'],
        'ORIGEN': ['ORIGEN', 'FUENTE', 'SOURCE', 'PROCEDENCIA', 'Origen', 'origen'],
        'Stock': ['Stock', 'STOCK', 'INVENTARIO', 'INVENTORY', 'EXISTENCIAS', 'stock']
    }
    
    optional_map = {
        'RES_IVN': ['RES_IVN', 'RESERVAS_IVN', 'IVN_RES', 'RESERVA_IVN', 'res_ivn'],
        'RES_TRANS': ['RES_TRANS', 'RESERVAS_TRANS', 'TRANS_RES', 'RESERVA_TRANS', 'res_trans'],
        'RES_PED': ['RES_PED', 'RESERVAS_PED', 'PED_RES', 'RESERVA_PED', 'res_ped'],
        'ESTRAT': ['ESTRAT', 'ESTRATEGIA', 'STRAT', 'CATEGORIA', 'estrat']
    }
    
    missing_cols = []
    
    for expected_col, possible_names in required_map.items():
        found = False
        for possible in possible_names:
            if possible in df.columns:
                column_mapping[expected_col] = possible
                found = True
                break
        if not found:
            missing_cols.append(expected_col)
    
    for expected_col, possible_names in optional_map.items():
        for possible in possible_names:
            if possible in df.columns:
                column_mapping[expected_col] = possible
                break
            
    return column_mapping, missing_cols

column_mapping, missing_cols = map_column_names(df)
if missing_cols:
    st.error(f"⚠️ Faltan columnas requeridas: {', '.join(missing_cols)}")
    st.stop()

df = df.rename(columns=column_mapping)
columnas_opcionales = ['RES_IVN', 'RES_TRANS', 'RES_PED', 'ESTRAT']
for col in columnas_opcionales:
    if col not in df.columns:
        df[col] = 0 if col.startswith('RES') else 'A'

# --- Validación de datos mejorada ---
def validar_datos_dataframe(df, date_cols):
    """Valida la integridad de los datos"""
    issues = []
    
    # Validar fechas
    try:
        fechas = pd.to_datetime(date_cols)
        diferencias = (fechas[1:] - fechas[:-1]).days
        if any(diff > 35 for diff in diferencias):
            issues.append("⚠️ Hay saltos en las fechas históricas (más de 35 días entre meses)")
    except Exception as e:
        issues.append(f"❌ Error en formato de fechas: {e}")
    
    # Validar valores negativos
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if (df[col] < 0).any():
            issues.append(f"⚠️ Columna {col} tiene valores negativos")
    
    # Validar datos faltantes
    for col in ['CODIGO', 'ORIGEN', 'Stock']:
        if df[col].isna().any():
            issues.append(f"⚠️ Columna {col} tiene valores faltantes")
    
    return issues

# --- Configuración inicial optimizada ---
@st.cache_data(ttl=3600)
def get_date_columns(df):
    date_cols = [c for c in df.columns if re.match(r'^\d{4}-\d{2}-\d{2}$', str(c))]
    date_cols = sorted(date_cols, key=lambda x: pd.to_datetime(x))
    return date_cols

date_cols = get_date_columns(df)
if not date_cols:
    st.error("No se detectaron columnas de fechas.")
    st.stop()

# Validar datos
validation_issues = validar_datos_dataframe(df, date_cols)
if validation_issues:
    with st.expander("🔍 Issues de Validación de Datos"):
        for issue in validation_issues:
            st.write(issue)

num_months = st.sidebar.slider("Cantidad de meses a mostrar", 6, len(date_cols), min(12, len(date_cols)))
date_cols = date_cols[-num_months:]

# --- Lead time y nivel de servicio ---
def get_lead_time(origen):
    """Define lead time según origen"""
    lead_times = {
        'NMEX': 2, 'NTE': 2, 'UK': 2, 'USA': 2, 'NBA': 2,  # n+4 → n+6 (2 meses lead time)
        'NTJ': 3   # n+2 → n+5 (3 meses lead time)
    }
    return lead_times.get(origen, 2)

df['Lead_Time'] = df['ORIGEN'].apply(get_lead_time)

nivel_servicio = st.sidebar.selectbox("Nivel de servicio (%)", options=[80,85,90,95,97.5,99], index=3)
z_dict = {80:0.84,85:1.04,90:1.28,95:1.65,97.5:1.96,99:2.33}
z = z_dict[nivel_servicio]

# --- Cálculos base optimizados CON STOCK DE SEGURIDAD DINÁMICO ---
@st.cache_data(ttl=3600)
def calcular_metricas_base(df, date_cols, z):
    """Calcula métricas base de manera optimizada con stock de seguridad base"""
    df_calc = df.copy()
    
    # Métricas estadísticas vectorizadas
    df_calc['Media'] = df_calc[date_cols].mean(axis=1)
    df_calc['Media_Safe'] = df_calc['Media'].clip(lower=0.01)
    df_calc['Desviacion'] = df_calc[date_cols].std(axis=1, ddof=0)
    df_calc['Coef_Variacion'] = np.where(
        df_calc['Media'] > 0.01, 
        df_calc['Desviacion'] / df_calc['Media'], 
        0
    )
    
    # Stock de seguridad BASE (para referencia inicial)
    df_calc['Stock_Seguridad_Base'] = np.where(
        (df_calc['Media'] > 0.01) & (df_calc['Desviacion'] > 0),
        z * df_calc['Desviacion'] * np.sqrt(df_calc['Lead_Time']),
        0
    )
    
    # Totales optimizados
    df_calc['Total_Pedidos'] = df_calc.filter(like='Ped').fillna(0).sum(axis=1)
    df_calc['Total_Transito'] = df_calc.filter(like='Trans').fillna(0).sum(axis=1)
    
    # Reservas
    reserva_cols = [c for c in ['RES_IVN', 'RES_TRANS', 'RES_PED'] if c in df_calc.columns]
    df_calc['Total_Reservas'] = df_calc[reserva_cols].fillna(0).sum(axis=1)
    
    # Stock disponible y meses de inventario
    df_calc['Stock_Disponible'] = (
        df_calc['Stock'].fillna(0) + 
        df_calc['Total_Transito'].fillna(0) + 
        df_calc['Total_Pedidos'].fillna(0) - 
        df_calc['Total_Reservas'].fillna(0)
    )
    
    df_calc['Meses_Inventario'] = np.where(
        df_calc['Media'] > 0.01,
        df_calc['Stock_Disponible'] / df_calc['Media'],
        999
    )
    
    return df_calc

df = calcular_metricas_base(df, date_cols, z)

# --- Pestañas para familias optimizadas ---
df['FAMILIA'] = df['CODIGO'].str[:3]
familias = ['Todas'] + sorted(df['FAMILIA'].unique().tolist())
estrats_all = ['Todas'] + sorted(df['ESTRAT'].dropna().unique().tolist())

# Selección en sidebar con manejo de estado
selected_fam = st.sidebar.selectbox(
    "Selecciona familia", 
    familias, 
    index=familias.index(st.session_state.selected_family),
    key="family_selector"
)

selected_estrat = st.sidebar.selectbox(
    "Selecciona estratificación", 
    estrats_all, 
    index=estrats_all.index(st.session_state.selected_strat),
    key="strat_selector"
)

# Actualizar estado cuando cambian los selectores
if selected_fam != st.session_state.selected_family or selected_estrat != st.session_state.selected_strat:
    st.session_state.selected_family = selected_fam
    st.session_state.selected_strat = selected_estrat
    st.session_state.current_product_index = 0
    st.session_state.product_key += 1
    st.rerun()

# Filtrar productos optimizado
@st.cache_data(ttl=3600)
def filtrar_productos(df, familia, estrat):
    df_filtrado = df.copy()
    if familia != 'Todas':
        df_filtrado = df_filtrado[df_filtrado['FAMILIA'] == familia]
    if estrat != 'Todas':
        df_filtrado = df_filtrado[df_filtrado['ESTRAT'] == estrat]
    return sorted(df_filtrado['CODIGO'].unique().tolist())

productos = filtrar_productos(df, selected_fam, selected_estrat)

if not productos:
    st.warning("No se encontraron productos para la combinación seleccionada.")
    st.stop()

# --- Navegación mejorada ---
col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])

with col_nav1:
    if st.button("⬅️ Anterior"):
        st.session_state.current_product_index = (st.session_state.current_product_index - 1) % len(productos)
        st.session_state.product_key += 1
        st.rerun()

with col_nav3:
    if st.button("Siguiente ➡️"):
        st.session_state.current_product_index = (st.session_state.current_product_index + 1) % len(productos)
        st.session_state.product_key += 1
        st.rerun()

with col_nav2:
    current_index = st.session_state.current_product_index
    
    # Selector de producto con key dinámica para evitar conflictos
    sel = st.selectbox(
        "Selecciona un producto", 
        productos, 
        index=current_index, 
        key=f"product_selector_{st.session_state.product_key}"
    )
    
    # Sincronizar el índice si el usuario selecciona un producto diferente
    if productos.index(sel) != current_index:
        st.session_state.current_product_index = productos.index(sel)
        st.session_state.product_key += 1
        st.rerun()

st.write(f"**Producto {current_index + 1} de {len(productos)}**")

prod = df[df['CODIGO'] == sel].iloc[0]
lead_time = int(prod['Lead_Time'])
origen_actual = prod['ORIGEN']

# --- Inicialización mejorada de UserInputs ---
def inicializar_datos_usuario(sel, prod, date_cols, user_existing_data=None):
    """Inicializa o carga datos del usuario de manera optimizada"""
    
    if sel in st.session_state.UserInputs:
        return st.session_state.UserInputs[sel]
    
    hist_mean = int(prod[date_cols].mean()) if not np.isnan(prod[date_cols].mean()) else 0
    
    # Intentar cargar datos guardados
    datos_iniciales = {
        'Proyecciones': [hist_mean] * 12,
        'Pedidos': [0] * 4,
        'MOS': [4.0] * 4,
        'GUARDADO': False,
        'last_update': datetime.now()
    }
    
    if user_existing_data is not None:
        user_prod_data = user_existing_data[user_existing_data['producto'] == sel]
        if not user_prod_data.empty:
            proyecciones_user = [0] * 12
            pedidos_user = [0] * 4
            mos_user = [4.0] * 4
            
            proyecciones_data = user_prod_data[user_prod_data['tipo'] == 'proyeccion']
            for _, row in proyecciones_data.iterrows():
                if 0 <= row['mes'] < 12:
                    proyecciones_user[int(row['mes'])] = row['valor']
            
            pedidos_data = user_prod_data[user_prod_data['tipo'] == 'pedido']
            for _, row in pedidos_data.iterrows():
                if 0 <= row['mes_orden'] < 4:
                    idx = int(row['mes_orden'])
                    pedidos_user[idx] = row['valor']
                    mos_user[idx] = row['mos_objetivo']
            
            datos_iniciales.update({
                'Proyecciones': proyecciones_user,
                'Pedidos': pedidos_user,
                'MOS': mos_user,
                'GUARDADO': True
            })
    
    st.session_state.UserInputs[sel] = datos_iniciales
    return datos_iniciales

user_data = inicializar_datos_usuario(sel, prod, date_cols, user_existing_data)

# --- Función de autoguardado mejorada ---
def auto_save():
    """Guarda automáticamente si hay cambios pendientes"""
    current_time = datetime.now()
    time_diff = (current_time - st.session_state.last_save).total_seconds()
    
    # Verificar si hay datos no guardados
    has_unsaved = any(not data.get('GUARDADO', False) for data in st.session_state.UserInputs.values())
    
    if has_unsaved and time_diff > 120:  # 2 minutos
        if save_user_data(usuario, st.session_state.UserInputs):
            st.session_state.last_save = current_time
            for product in st.session_state.UserInputs:
                st.session_state.UserInputs[product]['GUARDADO'] = True
            st.sidebar.success("💾 Autoguardado completado")

# --- CÁLCULO DE STOCK PROYECTADO CON TIMING CORREGIDO Y FECHAS AL FINAL DE MES ---
def calcular_stock_proyectado_corregido(proyecciones, pedidos_planificados, lead_time, stock_inicial, origen):
    """Calcula el stock proyectado con timing CORREGIDO según las especificaciones"""
    
    # CORRECCIÓN: Timing específico según origen
    if origen == 'NTJ':
        # NTJ: Pedido n+2, Llegada n+5
        meses_pedido = [2, 3, 4, 5]  # Meses desde planificación para colocar órdenes
    else:
        # Otros orígenes: Pedido n+4, Llegada n+6  
        meses_pedido = [4, 5, 6, 7]  # Meses desde planificación para colocar órdenes
    
    # Inicializar stock proyectado (mes actual = mes 0)
    stock_proyectado = [stock_inicial]
    
    # Para cada mes proyectado (mes 1 a mes 12 desde planificación)
    for mes_proyectado in range(1, 13):
        stock_actual = stock_proyectado[-1]
        
        # Calcular pedidos que llegan en este mes proyectado
        pedidos_que_llegan = 0
        for i, pedido in enumerate(pedidos_planificados):
            # Mes en que se coloca la orden
            mes_colocacion_orden = meses_pedido[i]
            
            # Mes en que llega la orden
            mes_llegada_orden = mes_colocacion_orden + lead_time
            
            # Si la orden llega en este mes proyectado
            if mes_llegada_orden == mes_proyectado:
                pedidos_que_llegan += pedido
        
        # Calcular demanda de este mes
        demanda_mes = proyecciones[mes_proyectado - 1] if (mes_proyectado - 1) < len(proyecciones) else 0
        
        # Calcular nuevo stock
        nuevo_stock = stock_actual + pedidos_que_llegan - demanda_mes
        stock_proyectado.append(max(0, nuevo_stock))
    
    return stock_proyectado

# --- Visualización principal con SS dinámico CORREGIDO Y FECHAS AL FINAL DE MES ---
def crear_visualizacion_principal_corregida(prod_codigo, proyecciones, pedidos, lead_time, origen):
    """Crea la visualización principal con stock proyectado y SS dinámico CORREGIDO"""
    prod_row = df[df['CODIGO'] == prod_codigo].iloc[0]
    
    # Datos históricos - CORRECCIÓN: Usar último día del mes
    hist_data = prod_row[date_cols].T.reset_index()
    hist_data.columns = ['Fecha', 'Ventas']
    hist_data['Fecha'] = pd.to_datetime(hist_data['Fecha'])
    # Ajustar al último día del mes para datos históricos
    hist_data['Fecha'] = hist_data['Fecha'] + pd.offsets.MonthEnd(0)
    
    # Fechas de proyección CORREGIDAS - ÚLTIMO DÍA DEL MES
    ultima_fecha_ventas = pd.to_datetime(date_cols[-1])  # Última fecha histórica
    ultima_fecha_ventas = ultima_fecha_ventas + pd.offsets.MonthEnd(0)  # Ajustar al último día del mes
    
    # Fecha de planificación es el último día del mes actual
    fecha_planificacion = ultima_fecha_ventas + pd.DateOffset(months=1)  # Último día del mes siguiente
    fecha_planificacion = fecha_planificacion + pd.offsets.MonthEnd(0)  # Asegurar que es último día
    
    proy_dates = []
    for i in range(len(proyecciones)):
        # Generar fechas al último día de cada mes
        fecha = fecha_planificacion + pd.DateOffset(months=i+1)
        fecha = fecha + pd.offsets.MonthEnd(0)
        proy_dates.append(fecha)
    
    proy_dates = pd.DatetimeIndex(proy_dates)
    
    # Calcular stock proyectado CORREGIDO
    stock_inicial = prod_row['Stock_Disponible']
    stock_proyectado = calcular_stock_proyectado_corregido(
        proyecciones, pedidos, lead_time, stock_inicial, origen
    )
    
    # CALCULAR STOCK DE SEGURIDAD DINÁMICO POR MES CORREGIDO
    ss_dinamico_por_mes = []
    for mes in range(len(proyecciones)):
        # Usar las proyecciones futuras para calcular variabilidad
        inicio_ss = max(0, mes - 5)  # Últimos 6 meses incluyendo el actual
        fin_ss = min(mes + 1, len(proyecciones))
        periodo_ss = proyecciones[inicio_ss:fin_ss]
        
        if len(periodo_ss) > 1:
            std_dinamico = np.std(periodo_ss)
            media_dinamica = np.mean(periodo_ss)
        else:
            std_dinamico = prod_row['Desviacion']
            media_dinamica = prod_row['Media']
        
        # Stock de seguridad dinámico para este mes basado en proyecciones futuras
        ss_mes = z_dict[nivel_servicio] * std_dinamico * np.sqrt(lead_time)
        ss_dinamico_por_mes.append(max(ss_mes, media_dinamica * 0.5))  # Mínimo 50% de la media
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Serie histórica - CORRECCIÓN: Usar último día del mes
    fig.add_trace(
        go.Scatter(
            x=hist_data['Fecha'], y=hist_data['Ventas'],
            mode='lines+markers', name='Ventas Históricas',
            line=dict(color='blue', width=3),
            marker=dict(size=6),
            hovertemplate='Fecha: %{x|%b %Y}<br>Ventas: %{y:.0f}<extra></extra>'
        ),
        secondary_y=False,
    )
    
    # Proyecciones - CORRECCIÓN: Usar último día del mes
    fig.add_trace(
        go.Scatter(
            x=proy_dates, y=proyecciones,
            mode='lines+markers', name='Ventas Proyectadas',
            line=dict(color='orange', width=3, dash='dash'),
            marker=dict(size=6, symbol='diamond'),
            hovertemplate='Fecha: %{x|%b %Y}<br>Proyección: %{y:.0f}<extra></extra>'
        ),
        secondary_y=False,
    )
    
    # Stock proyectado - CORRECCIÓN: Usar último día del mes
    fig.add_trace(
        go.Bar(
            x=proy_dates, y=stock_proyectado[1:13],  # Desde mes n+1
            name='Stock Proyectado',
            marker_color='lightgreen', opacity=0.7,
            hovertemplate='Fecha: %{x|%b %Y}<br>Stock: %{y:.0f} unidades<extra></extra>'
        ),
        secondary_y=True,
    )
    
    # LÍNEA DE STOCK DE SEGURIDAD DINÁMICO POR MES - CORRECCIÓN
    fig.add_trace(
        go.Scatter(
            x=proy_dates, y=ss_dinamico_por_mes,
            mode='lines',
            name='Stock Seguridad Dinámico',
            line=dict(color='red', width=2, dash='dot'),
            hovertemplate='Fecha: %{x|%b %Y}<br>SS Dinámico: %{y:.0f} unidades<extra></extra>'
        ),
        secondary_y=True,
    )
    
    fig.update_layout(
        title=f"Serie de Tiempo y Stock Proyectado: {prod_codigo}",
        xaxis_title="Mes",
        hovermode='x unified',
        height=500,
        showlegend=True,
        plot_bgcolor='rgba(240,240,240,0.1)',
        xaxis=dict(
            tickformat='%b %Y',
            tickmode='auto',
            nticks=12
        )
    )
    
    fig.update_yaxes(title_text="Ventas (unidades)", secondary_y=False)
    fig.update_yaxes(title_text="Stock Proyectado (unidades)", secondary_y=True)
    
    return fig, stock_proyectado, ss_dinamico_por_mes

# Generar y mostrar gráfico principal CORREGIDO
current_proy = user_data['Proyecciones']
current_pedidos = user_data['Pedidos']

fig_principal, stock_proyectado, ss_dinamico_por_mes = crear_visualizacion_principal_corregida(
    sel, current_proy, current_pedidos, lead_time, origen_actual
)
st.plotly_chart(fig_principal, use_container_width=True)

# --- Ventas proyectadas (12 meses) - CORRECCIÓN: Usar último día del mes ---
st.subheader("✍️ Ventas proyectadas (12 meses)")

# Calcular fechas de proyección (último día de cada mes)
ultima_fecha_ventas = pd.to_datetime(date_cols[-1]) + pd.offsets.MonthEnd(0)
fecha_planificacion = ultima_fecha_ventas + pd.DateOffset(months=1)  # Último día del mes siguiente
fecha_planificacion = fecha_planificacion + pd.offsets.MonthEnd(0)

proy_dates = []
for i in range(12):
    fecha = fecha_planificacion + pd.DateOffset(months=i+1)
    fecha = fecha + pd.offsets.MonthEnd(0)
    proy_dates.append(fecha)

cols_proj = st.columns(4)

for col_idx in range(4):
    with cols_proj[col_idx]:
        for row_idx in range(3):
            i = (col_idx * 3) + row_idx
            if i < 12:
                key_name = f'proj_{sel}_{i}'
                value_current = int(user_data['Proyecciones'][i])
                mes_label = proy_dates[i].strftime('%b %Y')
                
                val = st.number_input(
                    f'{mes_label}', 
                    min_value=0, 
                    step=1, 
                    value=value_current, 
                    key=key_name
                )
                
                if val != user_data['Proyecciones'][i]:
                    user_data['Proyecciones'][i] = val
                    user_data['last_update'] = datetime.now()
                    user_data['GUARDADO'] = False

# --- Botón para actualizar gráfico después de las proyecciones ---
col_btn_proj = st.columns([1, 2, 1])
with col_btn_proj[1]:
    if st.button("🔄 Actualizar Cálculos de Órdenes y Gráfico", type="primary", use_container_width=True):
        st.session_state.recalculate_orders = True
        # Recalcular el gráfico con las nuevas proyecciones
        fig_principal, stock_proyectado, ss_dinamico_por_mes = crear_visualizacion_principal_corregida(
            sel, user_data['Proyecciones'], current_pedidos, lead_time, origen_actual
        )
        st.rerun()

# --- Métricas del producto mejoradas ---
st.subheader(f"📊 Métricas del producto {sel}")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Media de ventas", f"{prod['Media']:.2f}")
    st.metric("Desviación estándar", f"{prod['Desviacion']:.2f}")
    
    # Coeficiente de variación con umbral corregido (150%)
    cv_color = "red" if prod['Coef_Variacion'] > 1.5 else "green"
    st.markdown(f"<span style='color: {cv_color}'>Coef. Variación: {prod['Coef_Variacion']*100:.1f}%</span>", 
                unsafe_allow_html=True)
    
    st.metric("Stock Seguridad Base", f"{prod['Stock_Seguridad_Base']:.0f}")

with col2:
    if 'ESTRAT' in prod:
        estrat_color = {'A': 'green', 'B': 'gray', 'C': 'yellow', 'D': 'deepskyblue', 'E': 'red'}
        color = estrat_color.get(prod['ESTRAT'], 'black')
        st.markdown(
            f"**Estratificación:** <span style='color:{color}; font-size:24px'>{prod['ESTRAT']}</span>", 
            unsafe_allow_html=True
        )
    
    st.metric("Lead Time", f"{lead_time} meses")
    st.metric("Stock Disponible", f"{prod['Stock_Disponible']:.0f}")
    st.metric("Meses Inventario", f"{prod['Meses_Inventario']:.1f}")

with col3:
    # Alertas visuales mejoradas
    st.write("**🔍 Estado de Inventario**")
    
    # Alerta stock vs seguridad
    if prod['Stock_Disponible'] < prod['Stock_Seguridad_Base']:
        st.error(f"🚨 Stock bajo seguridad: {prod['Stock_Disponible']:.0f} < {prod['Stock_Seguridad_Base']:.0f}")
    elif prod['Stock_Disponible'] < prod['Stock_Seguridad_Base'] * 1.5:
        st.warning(f"⚠️ Stock cerca del mínimo: {prod['Stock_Disponible']:.0f}")
    else:
        st.success(f"✅ Stock adecuado: {prod['Stock_Disponible']:.0f}")
    
    # Alerta coeficiente variación
    if prod['Coef_Variacion'] > 1.5:
        st.error(f"📊 Alta variabilidad: CV={prod['Coef_Variacion']*100:.0f}%")
    elif prod['Coef_Variacion'] > 1.0:
        st.warning(f"📊 Variabilidad media: CV={prod['Coef_Variacion']*100:.0f}%")
    else:
        st.success(f"📊 Variabilidad normal: CV={prod['Coef_Variacion']*100:.0f}%")

# --- INVENTARIO DETALLADO ---
st.subheader("📦 Inventario Detallado")
cols_inv = st.columns(4)
cols_names = ['Stock', 'Tránsito', 'Pedidos', 'Reservas']

cols_data = [
    ['Stock'], 
    [c for c in df.columns if re.match(r'^Trans', c)], 
    [c for c in df.columns if re.match(r'^Ped', c)],
    []
]

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
            # Crear DataFrame para mostrar
            display_data = []
            for c in available_cols:
                display_data.append({
                    'Concepto': c,
                    'Valor': f"{prod[c]:.0f}" if pd.notna(prod[c]) else "0"
                })
            display_df = pd.DataFrame(display_data)
            st.dataframe(display_df, hide_index=True, use_container_width=True)
        else:
            st.info(f"No hay datos de {col_name}")

# --- Totales ---
st.subheader("🧮 Totales de Inventario")
tot_cols = st.columns(6)
tot_cols[0].metric("Total Stock", f"{prod['Stock']:.0f}")
tot_cols[1].metric("Total Tránsito", f"{prod['Total_Transito']:.0f}")
tot_cols[2].metric("Total Pedido", f"{prod['Total_Pedidos']:.0f}")
tot_cols[3].metric("Total Reservas", f"{prod['Total_Reservas']:.0f}")
tot_cols[4].metric("Stock Disponible", f"{prod['Stock_Disponible']:.0f}")
tot_cols[5].metric("Meses Inventario", f"{prod['Meses_Inventario']:.1f}")

# --- ALERTAS ---
st.subheader("⚠️ Alertas de Inventario")
alert_col1, alert_col2, alert_col3 = st.columns(3)

with alert_col1:
    if prod['Stock_Disponible'] < prod['Stock_Seguridad_Base']:
        st.error(f"🚨 Stock bajo seguridad: {prod['Stock_Disponible']:.0f} < {prod['Stock_Seguridad_Base']:.0f}")
    elif prod['Stock_Disponible'] < prod['Stock_Seguridad_Base'] * 1.5:
        st.warning(f"⚠️ Stock cerca del mínimo: {prod['Stock_Disponible']:.0f}")
    else:
        st.success(f"✅ Stock adecuado: {prod['Stock_Disponible']:.0f}")

with alert_col2:
    ratio_cobertura = prod['Stock_Disponible'] / prod['Media_Safe'] if prod['Media_Safe'] > 0 else 999
    if ratio_cobertura < 2:
        st.success(f"✅ Stock saludable: {ratio_cobertura:.1f} meses")
    elif ratio_cobertura > 4:
        st.warning(f"⚠️ Sobreinventario: {ratio_cobertura:.1f} meses")
    else:
        st.info(f"ℹ️ Stock normal: {ratio_cobertura:.1f} meses")

with alert_col3:
    if prod['Coef_Variacion'] > 1.5:  # Umbral corregido a 150%
        st.warning(f"⚠️ Alta variabilidad: CV={prod['Coef_Variacion']*100:.0f}%")
    elif prod['Media'] < 0.1:
        st.info("ℹ️ Baja rotación")
    else:
        st.success("✅ Variabilidad normal")

# --- ÓRDENES PLANIFICADAS CON TIMING CORREGIDO Y FECHAS AL FINAL DE MES ---
st.subheader("✍️ Órdenes planificadas y sugeridas")
st.info(f"ℹ️ Lead Time: {lead_time} meses | Nivel de servicio: {nivel_servicio}%")

# TIMING CORREGIDO SEGÚN ESPECIFICACIONES
if origen_actual == 'NTJ':
    # NTJ: Pedido n+2, Llegada n+5
    meses_pedido = 4
    offset_pedido = 2
    st.success(f"🔶 **Estructura NTJ** - Pedidos: n+{offset_pedido} | Llegada: n+{offset_pedido + lead_time}")
    meses_desde_planificacion = [2, 3, 4, 5]  # Enero, Febrero, Marzo, Abril
else:
    # Otros orígenes: Pedido n+4, Llegada n+6
    meses_pedido = 4
    offset_pedido = 4
    st.success(f"🔷 **Estructura No-NTJ** - Pedidos: n+{offset_pedido} | Llegada: n+{offset_pedido + lead_time}")
    meses_desde_planificacion = [4, 5, 6, 7]  # Marzo, Abril, Mayo, Junio

# Fechas para órdenes CORREGIDAS - ÚLTIMO DÍA DEL MES
ultima_fecha_ventas = pd.to_datetime(date_cols[-1]) + pd.offsets.MonthEnd(0)  # Último día del último mes histórico
fecha_planificacion = ultima_fecha_ventas + pd.DateOffset(months=1)  # Último día del mes actual de planificación
fecha_planificacion = fecha_planificacion + pd.offsets.MonthEnd(0)

# Calcular fechas de órdenes basadas en el timing corregido (último día del mes)
fechas_ordenes = []
for mes_offset in meses_desde_planificacion:
    fecha_orden = fecha_planificacion + pd.DateOffset(months=mes_offset)
    fecha_orden = fecha_orden + pd.offsets.MonthEnd(0)  # Asegurar último día del mes
    fechas_ordenes.append(fecha_orden)

# Calcular fechas de arribo (último día del mes)
fechas_arribo = []
for fecha_orden in fechas_ordenes:
    fecha_arribo = fecha_orden + pd.DateOffset(months=lead_time)
    fecha_arribo = fecha_arribo + pd.offsets.MonthEnd(0)
    fechas_arribo.append(fecha_arribo)

# Mostrar timeline CORREGIDO
st.info(f"**Planificación actual:** {fecha_planificacion.strftime('%d %b %Y')} (n)")
st.info(f"**Primera orden:** {fechas_ordenes[0].strftime('%d %b %Y')} (n+{meses_desde_planificacion[0]})")
st.info(f"**Primer arribo:** {fechas_arribo[0].strftime('%d %b %Y')} (n+{meses_desde_planificacion[0] + lead_time})")

orden_cols = st.columns(meses_pedido)

for j in range(meses_pedido):
    with orden_cols[j]:
        mes_label = fechas_ordenes[j].strftime('%d %b %Y')
        st.markdown(f"### 📅 {mes_label}")
        
        # CÁLCULOS CON TIMING CORREGIDO
        mes_colocacion_orden = meses_desde_planificacion[j]  # Mes en que se coloca la orden desde planificación
        mes_llegada_orden = mes_colocacion_orden + lead_time  # Mes en que llega la orden
        
        # Mostrar información de timing CORREGIDA
        st.info(f"**Timing:** Orden n+{mes_colocacion_orden} → Llega n+{mes_llegada_orden}")
        st.info(f"**Arribo:** {fechas_arribo[j].strftime('%d %b %Y')}")
        
        # CÁLCULOS DE STOCK PROYECTADO CORREGIDOS
        # CORRECCIÓN: Usar los índices correctos basados en el timing
        # stock_proyectado[0] = stock inicial en mes n (planificación actual)
        # stock_proyectado[1] = mes n+1, etc.
        
        # Stock proyectado al momento de colocar la orden (ej: n+4 para no-NTJ)
        stock_proyectado_colocacion = 0
        if mes_colocacion_orden < len(stock_proyectado):
            stock_proyectado_colocacion = stock_proyectado[mes_colocacion_orden]
        elif len(stock_proyectado) > 0:
            # Si excede el horizonte, usar el último valor disponible
            stock_proyectado_colocacion = stock_proyectado[-1]
        
        # Stock proyectado al momento de la llegada SIN considerar este pedido (ej: n+6 para no-NTJ)
        stock_proyectado_llegada_sin_pedido = 0
        if mes_llegada_orden < len(stock_proyectado):
            stock_proyectado_llegada_sin_pedido = stock_proyectado[mes_llegada_orden]
        elif len(stock_proyectado) > 0:
            # Si excede el horizonte, extrapolar basándose en la tendencia
            stock_proyectado_llegada_sin_pedido = stock_proyectado[-1]
        
        # Stock de seguridad dinámico para el mes de llegada
        # NOTA: ss_dinamico_por_mes tiene 12 elementos (meses n+1 a n+12)
        ss_para_este_mes = prod['Stock_Seguridad_Base']
        if mes_llegada_orden > 0 and (mes_llegada_orden - 1) < len(ss_dinamico_por_mes):
            ss_para_este_mes = ss_dinamico_por_mes[mes_llegada_orden - 1]
        elif len(ss_dinamico_por_mes) > 0:
            ss_para_este_mes = ss_dinamico_por_mes[-1]
        
        # Calcular promedio de ventas de los últimos 6 meses proyectados antes de la llegada
        # Usamos mes_llegada_orden como referencia para calcular los 6 meses previos
        inicio_promedio = max(0, mes_llegada_orden - 6)
        fin_promedio = min(mes_llegada_orden, len(current_proy))
        periodo_promedio = current_proy[inicio_promedio:fin_promedio]
        
        demanda_promedio_6m = 0
        if len(periodo_promedio) > 0:
            demanda_promedio_6m = np.mean(periodo_promedio)
        elif mes_llegada_orden - 1 < len(current_proy):
            # Si no hay 6 meses, usar el mes anterior a la llegada
            demanda_promedio_6m = current_proy[mes_llegada_orden - 1]
        elif len(current_proy) > 0:
            # Si no hay datos específicos, usar el promedio general
            demanda_promedio_6m = np.mean(current_proy)
        
        # Input de MOS objetivo
        mos_val = st.number_input(
            f'MOS objetivo al arribo', 
            min_value=1.0, 
            max_value=12.0, 
            step=0.5, 
            value=user_data['MOS'][j],
            key=f'MOS_{sel}_{j}'
        )
        
        # Calcular pedido sugerido para alcanzar MOS objetivo
        # CORRECCIÓN: Usar stock_proyectado_llegada_sin_pedido como base
        stock_deseado = mos_val * demanda_promedio_6m
        sugerido_mos = max(stock_deseado - stock_proyectado_llegada_sin_pedido, 0)
        
        # MOS actual proyectado (sin este pedido)
        mos_actual_proyectado = 0
        if demanda_promedio_6m > 0:
            mos_actual_proyectado = stock_proyectado_llegada_sin_pedido / demanda_promedio_6m
        
        # SUGERIDO POR STOCK DE SEGURIDAD DINÁMICO
        sugerido_ss = max(ss_para_este_mes - stock_proyectado_llegada_sin_pedido, 0)
        
        # Mostrar las dos perspectivas
        st.metric("💡 Sugerido por MOS", f"{sugerido_mos:.0f}")
        st.metric("🛡️ Sugerido por SS", f"{sugerido_ss:.0f}")
        
        # Mostrar información adicional
        if demanda_promedio_6m > 0:
            st.info(f"**MOS actual proyectado:** {mos_actual_proyectado:.1f} meses")
            st.info(f"**Demanda prom. 6m:** {demanda_promedio_6m:.1f}")
        st.info(f"**SS dinámico:** {ss_para_este_mes:.0f}")
        
        # Input de pedido del usuario
        plan_val = st.number_input(
            f'✏️ Orden a colocar', 
            min_value=0, 
            step=1, 
            value=int(user_data['Pedidos'][j]), 
            key=f'order_{sel}_{j}'
        )
        
        # Actualizar datos
        if plan_val != user_data['Pedidos'][j]:
            user_data['Pedidos'][j] = plan_val
            user_data['last_update'] = datetime.now()
            user_data['GUARDADO'] = False
        
        if user_data['MOS'][j] != mos_val:
            user_data['MOS'][j] = mos_val
            user_data['last_update'] = datetime.now()
            user_data['GUARDADO'] = False
        
        # Mostrar stocks proyectados CORREGIDOS
        # CORRECCIÓN: Estos ahora muestran los valores correctos según el timing
        st.metric("📦 Stock Proy. al Orden", f"{stock_proyectado_colocacion:.0f}")
        st.metric("🚚 Stock Proy. al Arribo", f"{stock_proyectado_llegada_sin_pedido:.0f}")
        
        # Información adicional sobre fechas
        with st.expander("📅 Detalles de fechas"):
            st.write(f"**Fecha colocación orden:** {fechas_ordenes[j].strftime('%d/%m/%Y')}")
            st.write(f"**Fecha arribo pedido:** {fechas_arribo[j].strftime('%d/%m/%Y')}")
            st.write(f"**Días entre orden y arribo:** {(fechas_arribo[j] - fechas_ordenes[j]).days} días")

# --- Autoguardado periódico ---
auto_save()

# --- Estado de guardado mejorado ---
st.markdown("---")
st.subheader("💾 Estado de Datos")
col_status1, col_status2, col_status3 = st.columns(3)

with col_status1:
    estado = user_data.get('GUARDADO', False)
    if estado:
        st.success("✅ **ESTADO: GUARDADO**")
    else:
        st.warning("⚠️ **ESTADO: PENDIENTE DE GUARDAR**")

with col_status2:
    last_update = user_data.get('last_update')
    if last_update:
        if isinstance(last_update, str):
            last_update = datetime.fromisoformat(last_update)
        st.info(f"🕒 **Última actualización:** {last_update.strftime('%H:%M:%S')}")

with col_status3:
    if st.button("💾 Guardar Manualmente", type="primary", use_container_width=True):
        if save_user_data(usuario, st.session_state.UserInputs):
            for product in st.session_state.UserInputs:
                st.session_state.UserInputs[product]['GUARDADO'] = True
            st.session_state.last_save = datetime.now()
            st.rerun()

# --- ANÁLISIS DE COLORES ---
if df_colores is not None:
    st.title("🎨 Análisis de Velocidad de Venta por Color")
    
    # Mapeo de columnas para colores
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
            ventas_rapidas = df_colores_prod[[r for r in rangos_existentes if any(x in r for x in ['0-29','30-59','60-89','90-119'])]].sum().sum()
            
            with col_summary1:
                st.metric("Total ventas", f"{total_ventas:.0f}")
            with col_summary2:
                st.metric("Colores", num_colores)
            with col_summary3:
                pct_rapidas = (ventas_rapidas / total_ventas * 100) if total_ventas > 0 else 0
                st.metric("Ventas rápidas (0-119)", f"{pct_rapidas:.1f}%")
            
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
            
            df_colores_prod['Ventas_Rapidas_0-119'] = df_colores_prod[[r for r in rangos_existentes if any(x in r for x in ['0-29','30-59','60-89','90-119'])]].sum(axis=1)
            df_colores_prod['Ventas_Medias_120-240'] = df_colores_prod[[r for r in rangos_existentes if any(x in r for x in ['120','150','180','210','240'])]].sum(axis=1)
            df_colores_prod['Ventas_Lentas_240+'] = df_colores_prod[[r for r in rangos_existentes if any(x in r for x in ['270','300','330','360','390','420','Mayor'])]].sum(axis=1)
            
            df_colores_prod['% Rápidas'] = (df_colores_prod['Ventas_Rapidas_0-119'] / df_colores_prod['Total'] * 100).fillna(0)
            df_colores_prod['% Lentas'] = (df_colores_prod['Ventas_Lentas_240+'] / df_colores_prod['Total'] * 100).fillna(0)
            df_colores_prod['% por Color'] = (df_colores_prod['Total'] / df_colores_prod['Total'].sum() * 100).fillna(0)
            
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
                '% por Color',
                'Ventas_Rapidas_0-119', 
                'Ventas_Medias_120-240',
                'Ventas_Lentas_240+',
                '% Rápidas',
                '% Lentas',
                'Días_Promedio_Venta'
            ]].copy()
            
            tabla_resumen['% por Color'] = tabla_resumen['% por Color'].apply(lambda x: f"{x:.1f}%")
            tabla_resumen['% Rápidas'] = tabla_resumen['% Rápidas'].apply(lambda x: f"{x:.1f}%")
            tabla_resumen['% Lentas'] = tabla_resumen['% Lentas'].apply(lambda x: f"{x:.1f}%")
            tabla_resumen['Días_Promedio_Venta'] = tabla_resumen['Días_Promedio_Venta'].apply(lambda x: f"{x:.0f}")
            
            tabla_resumen.columns = [
                'Color', 
                'Total Ventas',
                '% del Total',
                'Ventas Rápidas (0-119)', 
                'Ventas Medias (120-240)',
                'Ventas Lentas (240+)',
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
                st.write(f"- {mejor_color['% Rápidas']:.1f}% en < 120 días")
                st.write("🎯 **Acción:** Priorizar en órdenes")
            
            with col_rec2:
                st.warning(f"**⚠️ Color más lento:** {peor_color['Sig. Color']}")
                st.write(f"- Días promedio: {peor_color['Días_Promedio_Venta']:.0f}")
                st.write(f"- {peor_color['% Lentas']:.1f}% después de 240 días")
                st.write("🎯 **Acción:** Reducir inventario")
    
    else:
        st.info(f"ℹ️ No hay datos de colores para {sel}")
else:
    st.info("💡 No se cargaron datos de colores")

# --- EXPORTAR DATOS ---
st.subheader("📥 Descargar datos ingresados")

def generar_excel_mejorado():
    """Genera archivo Excel optimizado con todos los datos"""
    try:
        all_data = []
        fecha_export = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        for prod_name, prod_data in st.session_state.UserInputs.items():
            # Proyecciones
            proy_dates = pd.date_range(start=datetime.today(), periods=12, freq='MS')
            for i, v in enumerate(prod_data.get('Proyecciones', [])):
                all_data.append({
                    'Producto': prod_name,
                    'Tipo': 'Proyección',
                    'Mes': proy_dates[i].strftime('%Y-%m'),
                    'Valor': int(v),
                    'MOS_Objetivo': None,
                    'Usuario': usuario,
                    'Fecha_Exportacion': fecha_export,
                    'Estado_Guardado': prod_data.get('GUARDADO', False)
                })

            # Pedidos
            order_dates = pd.date_range(start=datetime.today(), periods=4, freq='MS')
            for j in range(4):
                all_data.append({
                    'Producto': prod_name,
                    'Tipo': 'Pedido',
                    'Mes_Orden': order_dates[j].strftime('%Y-%m'),
                    'Valor': int(prod_data.get('Pedidos', [0]*4)[j]),
                    'MOS_Objetivo': prod_data.get('MOS', [4.0]*4)[j],
                    'Usuario': usuario,
                    'Fecha_Exportacion': fecha_export,
                    'Estado_Guardado': prod_data.get('GUARDADO', False)
                })
        
        if not all_data:
            st.warning("No hay datos para exportar")
            return None
            
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

excel_data = generar_excel_mejorado()
if excel_data:
    st.download_button(
        label="📥 Descargar Datos Completos (.xlsx)",
        data=excel_data,
        file_name=f"Nissan_Aprovisionamiento_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )
else:
    st.error("No se pudo generar archivo de exportación")

# --- Footer informativo ---
st.markdown("---")
col_footer1, col_footer2, col_footer3 = st.columns(3)
with col_footer1:
    st.caption(f"👤 Usuario: {usuario}")
with col_footer2:
    st.caption(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
with col_footer3:
    st.caption("🔄 Autoguardado activo")

# --- Limpieza periódica de caché ---
if st.sidebar.button("🧹 Limpiar caché", type="secondary"):
    st.cache_data.clear()
    st.session_state.calculations_cache = {}
    st.sidebar.success("Caché limpiado")