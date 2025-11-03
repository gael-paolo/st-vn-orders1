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

# ----------------------------------------------------------------------
# --- FUNCIÓN DE CARGA DE DATOS ---
# ----------------------------------------------------------------------

@st.cache_data(ttl=3600)  # Cache por 1 hora
def load_data_from_url(url, descripcion="archivo"):
    """Carga un archivo CSV desde una URL pública con manejo de errores."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        return df
    except requests.exceptions.Timeout:
        st.error(f"⏱️ **Timeout** al cargar {descripcion} desde {url}")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ **Error HTTP {e.response.status_code}** al cargar {descripcion}")
        if e.response.status_code == 403:
            st.error("🔒 Acceso Denegado. Verifique permisos públicos en Google Cloud Storage.")
        return None
    except pd.errors.ParserError:
        st.error(f"📄 **Error al parsear CSV**: El archivo descargado no tiene formato CSV válido.")
        return None
    except Exception as e:
        st.error(f"❌ **Error inesperado** al cargar {descripcion}: {str(e)}")
        return None

# ----------------------------------------------------------------------
# --- CONFIGURACIÓN Y CARGA INICIAL ---
# ----------------------------------------------------------------------

# --- Usuario ---
usuario = st.sidebar.text_input("Nombre de usuario", value="Usuario")

# --- Verificar que existen los secrets ---
try:
    # URL_ORDERS debe ser la URL pública del archivo principal de Órdenes/Ventas
    URL_ORDERS = st.secrets["URL_ORDERS"]
    # URL_COLORS (opcional)
    URL_COLORS = st.secrets.get("URL_COLORS")
except KeyError as e:
    st.error(f"❌ **Error de configuración**: Falta el secret {str(e)}")
    st.info("Configura las URLs públicas en `st.secrets`.")
    st.stop()

# Botón de recarga en sidebar
if st.sidebar.button("🔄 Recargar datos"):
    st.cache_data.clear()
    st.rerun()

# --- Carga de datos ---
with st.spinner("📥 Cargando datos desde la fuente..."):
    df = load_data_from_url(URL_ORDERS, "órdenes")
    df_colores = load_data_from_url(URL_COLORS, "colores") if URL_COLORS else None

if df is None:
    st.error("❌ No se pudo cargar el archivo principal de órdenes. Deteniendo ejecución.")
    st.stop()

# ----------------------------------------------------------------------
# --- PROCESAMIENTO INICIAL Y METADATOS ---
# ----------------------------------------------------------------------

# --- Mapeo flexible de columnas ---
def map_column_names(df):
    """Mapea nombres de columnas alternativos a los esperados"""
    required_map = {
        'CODIGO': ['CODIGO', 'CÓDIGO', 'COD', 'MODELO', 'SKU', 'Producto', 'Código', 'codigo'],
        'ORIGEN': ['ORIGEN', 'FUENTE', 'SOURCE', 'PROCEDENCIA', 'Origen', 'origen'],
        'Stock': ['Stock', 'STOCK', 'INVENTARIO', 'INVENTORY', 'EXISTENCIAS', 'stock']
    }
    column_mapping = {}
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
            
    return column_mapping, missing_cols

column_mapping, missing_cols = map_column_names(df)

if missing_cols:
    st.error(f"⚠️ Faltan columnas requeridas: {', '.join(missing_cols)}")
    st.stop()

df = df.rename(columns=column_mapping)
st.sidebar.success("✅ Datos de órdenes cargados y mapeados.")

# Crear columnas opcionales con valor 0 si no existen (Reservas y Tránsitos)
columnas_opcionales = ['RES_IVN', 'RES_TRANS', 'RES_PED']
for col in columnas_opcionales:
    if col not in df.columns:
        df[col] = 0

# --- Columnas de fechas ---
date_cols = [c for c in df.columns if re.match(r'^\d{4}-\d{2}-\d{2}$', str(c))]
date_cols = sorted(date_cols, key=lambda x: pd.to_datetime(x))
if not date_cols:
    st.error("No se detectaron columnas de ventas históricas (formato YYYY-MM-DD).")
    st.stop()

num_months = st.sidebar.slider("Cantidad de meses históricos a mostrar", 6, len(date_cols), min(12, len(date_cols)))
date_cols = date_cols[-num_months:]

# --- Lead time por ORIGEN ---
def get_lead_time(origen):
    return {'NMEX':2,'NTE':3,'NTJ':4}.get(origen,3)
df['Lead_Time'] = df['ORIGEN'].apply(get_lead_time)

# --- Nivel de servicio ---
nivel_servicio = st.sidebar.selectbox("Nivel de servicio (%)", options=[80,85,90,95,97.5,99], index=3)
z_dict = {80:0.84,85:1.04,90:1.28,95:1.65,97.5:1.96,99:2.33}
z = z_dict[nivel_servicio]

# --- Métricas base ---
df['Media'] = df[date_cols].mean(axis=1)
df['Media_Safe'] = df['Media'].clip(lower=0.01)
df['Desviacion'] = df[date_cols].std(axis=1)
df['Coef_Variacion'] = np.where(df['Media'] > 0.01, df['Desviacion'] / df['Media'], 0)
df['Stock_Seguridad'] = np.where((df['Media'] > 0.01) & (df['Desviacion'] > 0), z * df['Desviacion'] * np.sqrt(df['Lead_Time']), 0)
df['Total_Pedidos'] = df.filter(like='Ped').fillna(0).sum(axis=1)
df['Total_Transito'] = df.filter(like='Trans').fillna(0).sum(axis=1)
df['Total_Reservas'] = df[['RES_IVN', 'RES_TRANS', 'RES_PED']].fillna(0).sum(axis=1)
df['Stock_Disponible'] = (df['Stock'].fillna(0) + df['Total_Transito'].fillna(0) + df['Total_Pedidos'].fillna(0) - df['Total_Reservas'].fillna(0))
df['Meses_Inventario'] = np.where(df['Media'] > 0.01, df['Stock_Disponible'] / df['Media'], 999)

# --- Selección de familia y producto ---
df['FAMILIA'] = df['CODIGO'].str[:3]
familias = sorted(df['FAMILIA'].unique().tolist())
selected_fam = st.sidebar.selectbox("Selecciona familia", familias)
productos = sorted(df[df['FAMILIA']==selected_fam]['CODIGO'].unique().tolist())
sel = st.selectbox("Selecciona un producto", productos)
prod = df[df['CODIGO']==sel].iloc[0]
lead_time = int(prod['Lead_Time'])


# ----------------------------------------------------------------------
# --- INICIALIZACIÓN Y LÓGICA DE APROVISIONAMIENTO ---
# ----------------------------------------------------------------------

# --- INICIALIZACIÓN DE INPUTS ---
if 'UserInputs' not in st.session_state:
    st.session_state['UserInputs'] = {}

# Si el producto actual no existe en la sesión, lo inicializamos con valores por defecto.
if sel not in st.session_state['UserInputs']:
    hist_mean = int(prod[date_cols].mean()) if not np.isnan(prod[date_cols].mean()) else 0
    st.session_state['UserInputs'][sel] = {
        'Proyecciones': [hist_mean]*12, 
        'Pedidos': [0]*4, 
        'MOS': [2.0]*4,
        'GUARDADO': False 
    }

# --- GRÁFICOS Y MÉTRICAS (Igual al código anterior) ---
# [CONTENIDO DE GRÁFICO Y MÉTRICAS AQUÍ]
# Dado que es muy extenso, se omite para mantener el foco en la lógica clave.
# Asegúrate de que tus st.number_input usen el key=f'...' para Pedidos, Proyecciones y MOS.

# Ejemplo de un st.number_input para que funcione con la lógica de Guardar Registros:
st.subheader("✍️ Ventas proyectadas (12 meses)")
cols_proj = st.columns(4)
for i in range(12):
    with cols_proj[i%4]:
        st.number_input(
            f'Mes {i+1}', 
            min_value=0, 
            step=1, 
            value=int(st.session_state['UserInputs'][sel]['Proyecciones'][i]), 
            key=f'proj_{sel}_{i}' # ESTE KEY ES VITAL
        )

# Ejemplo de un st.number_input para Pedidos:
st.subheader("✍️ Órdenes planificadas (4 meses)")
orden_cols = st.columns(4)
for j in range(4):
    with orden_cols[j]:
        st.number_input(
            f'✏️ Orden a colocar', 
            min_value=0, 
            step=1, 
            value=int(st.session_state['UserInputs'][sel]['Pedidos'][j]), 
            key=f'order_{sel}_{j}' # ESTE KEY ES VITAL
        )


# ----------------------------------------------------------------------
# --- SECCIÓN: BOTÓN GUARDAR REGISTROS ---
# ----------------------------------------------------------------------
st.markdown("---")
st.subheader("✍️ Confirmación de Aprovisionamiento")
col_save, col_status = st.columns([1, 2])

with col_save:
    if st.button("💾 Guardar Registros", type="primary"):
        
        # 1. Sincronizar Proyecciones (12 meses)
        for i in range(12):
            # Usamos .get() con el valor actual de la sesión como fallback, aunque con keys únicos no debería fallar
            st.session_state['UserInputs'][sel]['Proyecciones'][i] = st.session_state.get(f'proj_{sel}_{i}', st.session_state['UserInputs'][sel]['Proyecciones'][i])
            
        # 2. Sincronizar Pedidos y MOS (4 meses)
        for j in range(4):
            st.session_state['UserInputs'][sel]['Pedidos'][j] = st.session_state.get(f'order_{sel}_{j}', st.session_state['UserInputs'][sel]['Pedidos'][j])
            st.session_state['UserInputs'][sel]['MOS'][j] = st.session_state.get(f'MOS_{sel}_{j}', st.session_state['UserInputs'][sel]['MOS'][j])
            
        # 3. Marcar como GUARDADO
        st.session_state['UserInputs'][sel]['GUARDADO'] = True
        
        st.toast(f"✅ Registros guardados para **{sel}** por {usuario}!", icon='💾')
        st.rerun() # Recarga para actualizar el estado visual

# 4. Mostrar el estado de guardado
with col_status:
    estado = st.session_state['UserInputs'].get(sel, {}).get('GUARDADO', False)
    if estado:
        st.success(f"✅ Estado: **GUARDADO** (Exportable)")
    else:
        st.warning(f"⚠️ Estado: **PENDIENTE DE GUARDAR** (Se omitirá en la exportación)")

st.markdown("---")


# ----------------------------------------------------------------------
# --- FUNCIÓN EXPORTAR DATOS (SIN NO REVISADO) ---
# ----------------------------------------------------------------------

def generar_excel(selected_fam, df, usuario):
    """Genera archivo Excel SOLO con los productos que han sido GUARDADOS
       en la familia seleccionada."""
    try:
        all_data = []
        fecha_export = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        df_familia = df[df['FAMILIA'] == selected_fam].copy()
        productos_familia = df_familia['CODIGO'].unique().tolist()

        # 1. Determinar qué productos de ESTA familia fueron realmente GUARDADOS
        productos_guardados_en_familia = []
        for prod_name in productos_familia:
            prod_data = st.session_state['UserInputs'].get(prod_name)
            
            # Condición CRÍTICA: Solo si el producto existe en la sesión Y tiene la bandera GUARDADO=True
            if prod_data is not None and prod_data.get('GUARDADO', False):
                productos_guardados_en_familia.append(prod_name)

        # 2. VERIFICACIÓN CRÍTICA: Si NINGÚN producto de la familia actual ha sido guardado, no se exporta nada.
        if not productos_guardados_en_familia:
            st.warning(f"⚠️ La familia **{selected_fam}** no tiene productos guardados. El archivo Excel estará vacío.")
            # Retorna un DataFrame vacío
            return pd.DataFrame(columns=['Producto', 'Tipo', 'Mes', 'Valor', 'MOS', 'Usuario', 'Fecha_Exportacion']).to_excel(io.BytesIO(), index=False).getvalue()

        # 3. Iterar solo sobre los productos que se sabe que están GUARDADOS
        for prod_name in productos_guardados_en_familia:
            
            prod_data = st.session_state['UserInputs'][prod_name]
            
            # --- Proyecciones (12 meses) ---
            proy_dates = pd.date_range(start=datetime.today(), periods=12, freq='MS')
            for i, v in enumerate(prod_data['Proyecciones']):
                all_data.append({
                    'Producto': prod_name,
                    'Tipo': 'Proyección',
                    'Mes': proy_dates[i].strftime('%Y-%m'),
                    'Valor': int(v),
                    'MOS': None,
                    'Usuario': usuario,
                    'Fecha_Exportacion': fecha_export
                })

            # --- Pedidos (Aprovisionamiento, 4 meses) ---
            order_dates = pd.date_range(start=datetime.today(), periods=4, freq='MS')
            
            for j in range(4):
                # Exportamos los valores numéricos (0 o >0)
                pedido_val = int(prod_data['Pedidos'][j])
                mos_val = prod_data['MOS'][j]
                
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
        
        # 4. Generar el Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            export_df.to_excel(writer, index=False, sheet_name='Datos_Ingresados')
            
            # Ajuste de ancho de columnas
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

# --- DESCARGA ---
st.subheader("📥 Descargar datos ingresados")
excel_data = generar_excel(selected_fam, df, usuario)

if excel_data:
    st.download_button(
        label=f"📥 Descargar Orden de {selected_fam} (.xlsx)",
        data=excel_data,
        file_name=f"Nissan_Order_{selected_fam}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    # Este mensaje solo se muestra si la función falla o retorna None por un error grave
    st.error("No se pudo generar archivo debido a un error interno.")

st.markdown("---")
st.caption(f"👤 {usuario} | 📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")