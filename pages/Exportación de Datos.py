import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64

# Configurar la página
st.set_page_config(
    page_title="Dashboard Electoral NL 2021 - Corregido",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .candidate-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 0.5rem;
    }
    .export-buttons {
        display: flex;
        gap: 10px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


class DashboardElectoralCorregido:
    def __init__(self, db_path='elecciones_nl_2021.db'):
        self.db_path = db_path
        self._crear_tabla_corregida()

    def conectar(self):
        """Conectar a la base de datos"""
        return sqlite3.connect(self.db_path)

    def _crear_tabla_corregida(self):
        """Crear tabla corregida para gobernador si no existe"""
        conn = self.conectar()

        # Verificar si ya existe la tabla corregida
        try:
            conn.execute("SELECT 1 FROM gobernador_corregido LIMIT 1")
        except:
            # Crear tabla corregida para gobernador
            conn.execute("""
            CREATE TABLE gobernador_corregido AS
            SELECT 
                MIN(candidato_id) as candidato_id,
                2021 as anno,
                nombre_candidato,
                SUM(numero_de_votos) as numero_de_votos,
                'Nuevo León' as division_territorial,
                MIN(nombre_normalizado) as nombre_normalizado,
                MIN(partido_ci) as partido_ci,
                'GOBERNADOR' as tipo_eleccion,
                CURRENT_TIMESTAMP as created_at
            FROM resultados_electorales 
            WHERE tipo_eleccion = 'GOBERNADOR'
            GROUP BY nombre_candidato
            ORDER BY numero_de_votos DESC;
            """)
            conn.commit()
            st.sidebar.success("✅ Tabla de gobernador corregida creada")

        conn.close()

    def ejecutar_consulta(self, consulta, params=None):
        """Ejecutar consulta SQL"""
        with self.conectar() as conn:
            return pd.read_sql_query(consulta, conn, params=params)

    def obtener_datos_gobernador_corregidos(self):
        """Obtener datos corregidos de gobernador (7 candidatos únicos)"""
        consulta = "SELECT * FROM gobernador_corregido ORDER BY numero_de_votos DESC;"
        return self.ejecutar_consulta(consulta)

    def obtener_datos_por_tipo(self, tipo_eleccion):
        """Obtener datos por tipo de elección (usando tabla corregida para gobernador)"""
        if tipo_eleccion == 'GOBERNADOR':
            return self.obtener_datos_gobernador_corregidos()
        else:
            consulta = f"""
            SELECT * FROM resultados_electorales 
            WHERE tipo_eleccion = ? 
            ORDER BY numero_de_votos DESC;
            """
            return self.ejecutar_consulta(consulta, (tipo_eleccion,))

    def obtener_estadisticas_generales(self):
        """Obtener estadísticas generales corregidas"""
        # Para gobernador usar tabla corregida, para otros usar tabla original
        consulta_otros = """
        SELECT 
            tipo_eleccion,
            COUNT(*) as registros,
            COUNT(DISTINCT nombre_candidato) as candidatos_unicos,
            SUM(numero_de_votos) as total_votos
        FROM resultados_electorales 
        WHERE tipo_eleccion != 'GOBERNADOR'
        GROUP BY tipo_eleccion;
        """

        consulta_gobernador = """
        SELECT 
            'GOBERNADOR' as tipo_eleccion,
            COUNT(*) as registros,
            COUNT(DISTINCT nombre_candidato) as candidatos_unicos,
            SUM(numero_de_votos) as total_votos
        FROM gobernador_corregido;
        """

        stats_otros = self.ejecutar_consulta(consulta_otros)
        stats_gobernador = self.ejecutar_consulta(consulta_gobernador)

        stats_combinadas = pd.concat([stats_gobernador, stats_otros], ignore_index=True)

        # Totales generales
        total_registros = stats_combinadas['registros'].sum()
        total_candidatos = stats_combinadas['candidatos_unicos'].sum()
        total_votos = stats_combinadas['total_votos'].sum()
        partidos_unicos = self.ejecutar_consulta(
            "SELECT COUNT(DISTINCT partido_ci) as count FROM resultados_electorales;"
        )['count'].iloc[0]
        divisiones_unicas = self.ejecutar_consulta(
            "SELECT COUNT(DISTINCT division_territorial) as count FROM resultados_electorales WHERE tipo_eleccion != 'GOBERNADOR';"
        )['count'].iloc[0] + 1  # +1 por Nuevo León de gobernador

        return {
            'total_registros': total_registros,
            'total_candidatos': total_candidatos,
            'total_votos': total_votos,
            'partidos_unicos': partidos_unicos,
            'divisiones_unicas': divisiones_unicas,
            'detalle_por_tipo': stats_combinadas
        }

    def obtener_todos_los_datos(self):
        """Obtener todos los datos combinados (para exportación completa)"""
        # Datos de gobernador corregidos
        gobernador = self.obtener_datos_gobernador_corregidos()

        # Datos de diputados y municipales
        diputados = self.ejecutar_consulta("SELECT * FROM resultados_electorales WHERE tipo_eleccion = 'DIPUTADO';")
        municipales = self.ejecutar_consulta("SELECT * FROM resultados_electorales WHERE tipo_eleccion = 'MUNICIPAL';")

        # Combinar todos los datos
        todos_los_datos = pd.concat([gobernador, diputados, municipales], ignore_index=True)
        return todos_los_datos


# Función para crear enlace de descarga CSV
def get_csv_download_link(df, filename, button_text=" Descargar CSV"):
    """Generar un enlace para descargar DataFrame como CSV"""
    csv = df.to_csv(index=False, encoding='utf-8')
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; border-radius: 5px; margin: 5px;">{button_text}</a>'
    return href


# Instanciar el dashboard corregido
dashboard = DashboardElectoralCorregido()

# HEADER PRINCIPAL
st.markdown('<h1 class="main-header">Dashboard Electoral NL 2021 - Corregido</h1>', unsafe_allow_html=True)


# SIDEBAR CON OPCIONES DE EXPORTACIÓN
with st.sidebar:
    st.header(" Controles del Dashboard")
    st.markdown("---")

    # SECCIÓN DE EXPORTACIÓN COMPLETA
    st.subheader(" Exportación Completa")

    if st.button(" Exportar Base de Datos Completa"):
        with st.spinner("Preparando archivo..."):
            todos_los_datos = dashboard.obtener_todos_los_datos()
            st.markdown(get_csv_download_link(
                todos_los_datos,
                f"base_datos_electoral_completa_{datetime.now().strftime('%Y%m%d')}.csv",
                " Descargar Base de Datos Completa"
            ), unsafe_allow_html=True)

    # Exportaciones individuales
    st.subheader(" Exportar por Tipo")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button(" Gobernador"):
            datos = dashboard.obtener_datos_gobernador_corregidos()
            st.markdown(get_csv_download_link(
                datos, "datos_gobernador_corregidos.csv", "📥 Gobernador"
            ), unsafe_allow_html=True)

    with col2:
        if st.button(" Diputados"):
            datos = dashboard.obtener_datos_por_tipo('DIPUTADO')
            st.markdown(get_csv_download_link(
                datos, "datos_diputados.csv", "📥 Diputados"
            ), unsafe_allow_html=True)

    with col3:
        if st.button("️ Municipales"):
            datos = dashboard.obtener_datos_por_tipo('MUNICIPAL')
            st.markdown(get_csv_download_link(
                datos, "datos_municipales.csv", "📥 Municipales"
            ), unsafe_allow_html=True)

    st.markdown("---")

    # Filtros principales
    st.subheader("Filtros")

    # Filtro por tipo de elección
    tipos_eleccion = ['GOBERNADOR', 'DIPUTADO', 'MUNICIPAL']
    tipo_seleccionado = st.selectbox(
        "Tipo de Elección:",
        options=tipos_eleccion,
        index=0
    )

    # Filtro por partido (dinámico según tipo de elección)
    if tipo_seleccionado == 'GOBERNADOR':
        partidos_query = "SELECT DISTINCT partido_ci FROM gobernador_corregido ORDER BY partido_ci;"
    else:
        partidos_query = f"SELECT DISTINCT partido_ci FROM resultados_electorales WHERE tipo_eleccion = '{tipo_seleccionado}' ORDER BY partido_ci;"

    partidos = dashboard.ejecutar_consulta(partidos_query)['partido_ci'].tolist()
    partido_seleccionado = st.multiselect(
        "Partido Político:",
        options=partidos,
        default=partidos
    )

    # Búsqueda de candidato
    candidato_busqueda = st.text_input(" Buscar candidato:")

    st.markdown("---")


# SECCIÓN 1: MÉTRICAS PRINCIPALES CORREGIDAS
st.header(" Métricas Principales - Corregidas")

# Obtener estadísticas corregidas
stats = dashboard.obtener_estadisticas_generales()

# Crear columnas para las métricas
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="Total de Candidatos",
        value=stats['total_candidatos'],
        help="Candidatos únicos en todas las elecciones"
    )

with col2:
    st.metric(
        label="Partidos Políticos",
        value=stats['partidos_unicos'],
        help="Partidos que participaron"
    )

with col3:
    st.metric(
        label="Tipos de Elección",
        value=len(tipos_eleccion),
        help="Municipales, Diputados, Gobernador"
    )

with col4:
    st.metric(
        label="Divisiones Territoriales",
        value=stats['divisiones_unicas'],
        help="Municipios, Distritos y Estado"
    )

with col5:
    st.metric(
        label="Total de Votos",
        value=f"{stats['total_votos']:,}",
        help="Votos totales contabilizados"
    )

# Mostrar detalle por tipo de elección con opción de exportación
with st.expander(" Ver detalle por tipo de elección"):
    st.dataframe(stats['detalle_por_tipo'])

    # Botón de exportación para el detalle
    col1, col2 = st.columns([3, 1])
    with col2:
        st.markdown(get_csv_download_link(
            stats['detalle_por_tipo'],
            "estadisticas_por_tipo_eleccion.csv",
            " Exportar Estadísticas"
        ), unsafe_allow_html=True)

st.markdown("---")

# SECCIÓN 2: DATOS DEL TIPO DE ELECCIÓN SELECCIONADO
st.header(f" Datos de: {tipo_seleccionado}")

# Obtener datos según el tipo seleccionado
datos_tipo = dashboard.obtener_datos_por_tipo(tipo_seleccionado)

# Aplicar filtros adicionales
if partido_seleccionado:
    datos_filtrados = datos_tipo[datos_tipo['partido_ci'].isin(partido_seleccionado)]
else:
    datos_filtrados = datos_tipo

if candidato_busqueda:
    datos_filtrados = datos_filtrados[
        datos_filtrados['nombre_candidato'].str.contains(candidato_busqueda, case=False, na=False)
    ]

# Mostrar resumen de filtros
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(f"Candidatos {tipo_seleccionado}", len(datos_filtrados))
with col2:
    st.metric("Votos Totales", f"{datos_filtrados['numero_de_votos'].sum():,}")
with col3:
    st.metric("Partidos", datos_filtrados['partido_ci'].nunique())
with col4:
    # Botón de exportación para datos filtrados
    st.markdown(get_csv_download_link(
        datos_filtrados,
        f"datos_{tipo_seleccionado.lower()}_filtrados.csv",
        " Exportar Datos"
    ), unsafe_allow_html=True)

# SECCIÓN 3: GRÁFICOS ESPECÍFICOS POR TIPO DE ELECCIÓN
if tipo_seleccionado == 'GOBERNADOR':
    st.subheader(" Candidatos a Gobernador - Resultados Consolidados")

    # Gráfico de barras para gobernador
    fig_gobernador = px.bar(
        datos_filtrados,
        x='numero_de_votos',
        y='nombre_candidato',
        orientation='h',
        color='partido_ci',
        title='Resultados para Gobernador - Votos Consolidados',
        text='numero_de_votos',
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig_gobernador.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        showlegend=True
    )
    fig_gobernador.update_traces(texttemplate='%{text:,}', textposition='outside')
    st.plotly_chart(fig_gobernador, use_container_width=True)

else:
    # Para diputados y municipales
    col1, col2 = st.columns(2)

    with col1:
        # Top 10 candidatos
        top_10 = datos_filtrados.nlargest(10, 'numero_de_votos')
        fig_top = px.bar(
            top_10,
            x='numero_de_votos',
            y='nombre_candidato',
            orientation='h',
            color='partido_ci',
            title=f'Top 10 Candidatos - {tipo_seleccionado}',
            text='numero_de_votos'
        )
        fig_top.update_layout(yaxis={'categoryorder': 'total ascending'})
        fig_top.update_traces(texttemplate='%{text:,}', textposition='outside')
        st.plotly_chart(fig_top, use_container_width=True)

    with col2:
        # Distribución por partido
        por_partido = datos_filtrados.groupby('partido_ci').agg({
            'numero_de_votos': 'sum',
            'nombre_candidato': 'count'
        }).reset_index()
        por_partido.columns = ['Partido', 'Total_Votos', 'Candidatos']

        fig_partidos = px.pie(
            por_partido,
            values='Total_Votos',
            names='Partido',
            title=f'Distribución de Votos por Partido - {tipo_seleccionado}',
            hover_data=['Candidatos']
        )
        st.plotly_chart(fig_partidos, use_container_width=True)

        # Exportar datos por partido
        st.markdown(get_csv_download_link(
            por_partido,
            f"resumen_partidos_{tipo_seleccionado.lower()}.csv",
            " Exportar Resumen Partidos"
        ), unsafe_allow_html=True)

# SECCIÓN 4: TABLA DETALLADA CON MÚLTIPLES OPCIONES DE EXPORTACIÓN
st.header("Tabla Detallada de Resultados")

# Seleccionar columnas para mostrar
columnas_disponibles = {
    'nombre_candidato': 'Candidato',
    'partido_ci': 'Partido',
    'division_territorial': 'División Territorial',
    'numero_de_votos': 'Votos',
    'tipo_eleccion': 'Tipo Elección',
    'anno': 'Año'
}

columnas_seleccionadas = st.multiselect(
    "Seleccionar columnas para mostrar:",
    options=list(columnas_disponibles.keys()),
    default=['nombre_candidato', 'partido_ci', 'division_territorial', 'numero_de_votos'],
    format_func=lambda x: columnas_disponibles[x]
)

if columnas_seleccionadas:
    datos_mostrar = datos_filtrados[columnas_seleccionadas].rename(columns=columnas_disponibles)

    # Mostrar tabla
    st.dataframe(
        datos_mostrar,
        use_container_width=True,
        height=400
    )

    # OPCIONES DE EXPORTACIÓN
    st.subheader(" Opciones de Exportación")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Exportar datos mostrados
        st.markdown(get_csv_download_link(
            datos_mostrar,
            f"tabla_{tipo_seleccionado.lower()}_mostrada.csv",
            " Exportar Tabla Mostrada"
        ), unsafe_allow_html=True)

    with col2:
        # Exportar todos los datos del tipo
        st.markdown(get_csv_download_link(
            datos_filtrados,
            f"datos_completos_{tipo_seleccionado.lower()}.csv",
            " Exportar Datos Completos"
        ), unsafe_allow_html=True)

    with col3:
        # Exportar top 100
        top_100 = datos_filtrados.nlargest(100, 'numero_de_votos')[columnas_seleccionadas].rename(
            columns=columnas_disponibles)
        st.markdown(get_csv_download_link(
            top_100,
            f"top_100_{tipo_seleccionado.lower()}.csv",
            " Exportar Top 100"
        ), unsafe_allow_html=True)

    with col4:
        # Exportar resumen estadístico
        resumen_estadistico = datos_filtrados['numero_de_votos'].describe()
        resumen_df = pd.DataFrame({
            'Estadística': resumen_estadistico.index,
            'Valor': resumen_estadistico.values
        })
        st.markdown(get_csv_download_link(
            resumen_df,
            f"estadisticas_{tipo_seleccionado.lower()}.csv",
            " Exportar Estadísticas"
        ), unsafe_allow_html=True)

# SECCIÓN 5: EXPORTACIÓN AVANZADA
with st.expander("Exportación Avanzada"):
    st.subheader("Exportaciones Especializadas")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Por Partido Político**")
        partido_export = st.selectbox("Seleccionar partido:", partidos)
        if st.button(" Exportar Datos del Partido"):
            datos_partido = datos_filtrados[datos_filtrados['partido_ci'] == partido_export]
            st.markdown(get_csv_download_link(
                datos_partido,
                f"datos_{partido_export.lower().replace(' ', '_')}.csv",
                f" Descargar {partido_export}"
            ), unsafe_allow_html=True)

    with col2:
        st.write("**Top N Candidatos**")
        top_n = st.slider("Número de candidatos:", 10, 100, 20)
        if st.button(f" Exportar Top {top_n}"):
            top_n_data = datos_filtrados.nlargest(top_n, 'numero_de_votos')
            st.markdown(get_csv_download_link(
                top_n_data,
                f"top_{top_n}_{tipo_seleccionado.lower()}.csv",
                f" Descargar Top {top_n}"
            ), unsafe_allow_html=True)

