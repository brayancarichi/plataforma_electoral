import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Configurar la página
st.set_page_config(
    page_title="Dashboard Electoral NL 2021",
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
</style>
""", unsafe_allow_html=True)


class DashboardElectoral:
    def __init__(self, db_path='elecciones_nl_2021.db'):
        self.db_path = db_path

    def conectar(self):
        """Conectar a la base de datos"""
        return sqlite3.connect(self.db_path)

    def ejecutar_consulta(self, consulta, params=None):
        """Ejecutar consulta SQL"""
        with self.conectar() as conn:
            return pd.read_sql_query(consulta, conn, params=params)

    def obtener_estadisticas_generales(self):
        """Obtener estadísticas generales"""
        consulta = """
        SELECT 
            COUNT(*) as total_registros,
            COUNT(DISTINCT partido_ci) as partidos_unicos,
            COUNT(DISTINCT tipo_eleccion) as tipos_eleccion,
            COUNT(DISTINCT division_territorial) as divisiones_unicas,
            SUM(numero_de_votos) as total_votos
        FROM resultados_electorales;
        """
        return self.ejecutar_consulta(consulta).iloc[0]


# Instanciar el dashboard
dashboard = DashboardElectoral()

# HEADER PRINCIPAL
st.markdown('<h1 class="main-header">📊 Dashboard Electoral - Nuevo León 2021</h1>', unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.header("🎛️ Controles del Dashboard")
    st.markdown("---")

    # Filtros
    st.subheader("Filtros")

    # Filtro por tipo de elección
    tipos_eleccion = dashboard.ejecutar_consulta("SELECT DISTINCT tipo_eleccion FROM resultados_electorales")[
        'tipo_eleccion'].tolist()
    tipo_seleccionado = st.multiselect(
        "Tipo de Elección:",
        options=tipos_eleccion,
        default=tipos_eleccion
    )

    # Filtro por partido
    partidos = \
    dashboard.ejecutar_consulta("SELECT DISTINCT partido_ci FROM resultados_electorales ORDER BY partido_ci")[
        'partido_ci'].tolist()
    partido_seleccionado = st.multiselect(
        "Partido Político:",
        options=partidos,
        default=partidos[:5] if len(partidos) > 5 else partidos
    )

    # Búsqueda de candidato
    candidato_busqueda = st.text_input("🔍 Buscar candidato:")

    st.markdown("---")
    st.info("💡 Usa los filtros para explorar los datos electorales")

# SECCIÓN 1: MÉTRICAS PRINCIPALES
st.header("📈 Métricas Principales")

# Obtener estadísticas
stats = dashboard.obtener_estadisticas_generales()

# Crear columnas para las métricas
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="Total de Registros",
        value=f"{stats['total_registros']:,}",
        help="Número total de candidaturas"
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
        value=stats['tipos_eleccion'],
        help="Municipales, Diputados, Gobernador"
    )

with col4:
    st.metric(
        label="Divisiones Territoriales",
        value=stats['divisiones_unicas'],
        help="Municipios y Distritos"
    )

with col5:
    st.metric(
        label="Total de Votos",
        value=f"{stats['total_votos']:,}",
        help="Votos totales contabilizados"
    )

st.markdown("---")

# SECCIÓN 2: GRÁFICOS PRINCIPALES
st.header("📊 Visualizaciones")

# Tabs para diferentes visualizaciones
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Distribución de Votos",
    "🏆 Top Candidatos",
    "🎯 Análisis por Partido",
    "🗺️ Por División Territorial"
])

with tab1:
    # Gráfico de votos por tipo de elección
    consulta_votos_tipo = """
    SELECT 
        tipo_eleccion,
        SUM(numero_de_votos) as total_votos,
        COUNT(*) as cantidad_candidatos
    FROM resultados_electorales 
    GROUP BY tipo_eleccion 
    ORDER BY total_votos DESC;
    """

    datos_votos = dashboard.ejecutar_consulta(consulta_votos_tipo)

    col1, col2 = st.columns(2)

    with col1:
        # Pie chart de votos por tipo
        fig_pie = px.pie(
            datos_votos,
            values='total_votos',
            names='tipo_eleccion',
            title='Distribución de Votos por Tipo de Elección',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Bar chart de candidatos por tipo
        fig_bar = px.bar(
            datos_votos,
            x='tipo_eleccion',
            y='cantidad_candidatos',
            title='Cantidad de Candidatos por Tipo de Elección',
            color='tipo_eleccion',
            text='cantidad_candidatos'
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    # Top 20 candidatos más votados
    consulta_top_candidatos = """
    SELECT 
        nombre_candidato,
        partido_ci,
        tipo_eleccion,
        division_territorial,
        numero_de_votos
    FROM resultados_electorales 
    ORDER BY numero_de_votos DESC 
    LIMIT 20;
    """

    top_candidatos = dashboard.ejecutar_consulta(consulta_top_candidatos)

    # Gráfico de barras horizontal
    fig_top = px.bar(
        top_candidatos,
        y='nombre_candidato',
        x='numero_de_votos',
        color='tipo_eleccion',
        orientation='h',
        title='Top 20 Candidatos Más Votados',
        hover_data=['partido_ci', 'division_territorial']
    )
    fig_top.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_top, use_container_width=True)

    # Mostrar tabla detallada
    with st.expander("📋 Ver tabla detallada de top candidatos"):
        st.dataframe(top_candidatos)

with tab3:
    # Análisis por partido político
    consulta_partidos = """
    SELECT 
        partido_ci,
        COUNT(*) as candidatos_presentados,
        SUM(numero_de_votos) as total_votos,
        ROUND(AVG(numero_de_votos), 2) as promedio_votos
    FROM resultados_electorales 
    GROUP BY partido_ci 
    ORDER BY total_votos DESC;
    """

    datos_partidos = dashboard.ejecutar_consulta(consulta_partidos)

    col1, col2 = st.columns(2)

    with col1:
        # Top partidos por votos
        top_partidos = datos_partidos.head(10)
        fig_partidos = px.bar(
            top_partidos,
            x='partido_ci',
            y='total_votos',
            title='Top 10 Partidos por Total de Votos',
            color='total_votos',
            text='total_votos'
        )
        fig_partidos.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_partidos, use_container_width=True)

    with col2:
        # Scatter plot: votos vs candidatos
        fig_scatter = px.scatter(
            datos_partidos.head(15),
            x='candidatos_presentados',
            y='total_votos',
            size='promedio_votos',
            color='partido_ci',
            title='Relación: Candidatos Presentados vs Votos Obtenidos',
            hover_name='partido_ci',
            size_max=60
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

with tab4:
    # Análisis por división territorial
    consulta_divisiones = """
    SELECT 
        division_territorial,
        tipo_eleccion,
        COUNT(*) as cantidad_candidatos,
        SUM(numero_de_votos) as total_votos
    FROM resultados_electorales 
    GROUP BY division_territorial, tipo_eleccion
    ORDER BY total_votos DESC;
    """

    datos_divisiones = dashboard.ejecutar_consulta(consulta_divisiones)

    # Heatmap de votos por división y tipo
    pivot_data = datos_divisiones.pivot_table(
        index='division_territorial',
        columns='tipo_eleccion',
        values='total_votos',
        aggfunc='sum'
    ).fillna(0)

    fig_heatmap = px.imshow(
        pivot_data,
        title='Mapa de Calor: Votos por División Territorial y Tipo de Elección',
        color_continuous_scale='Blues',
        aspect='auto'
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

# SECCIÓN 3: BÚSQUEDA Y FILTRADO AVANZADO
st.header("🔍 Búsqueda Avanzada")

col1, col2 = st.columns([2, 1])

with col1:
    # Consulta dinámica basada en filtros
    where_conditions = []
    params = []

    if tipo_seleccionado:
        placeholders = ','.join(['?'] * len(tipo_seleccionado))
        where_conditions.append(f"tipo_eleccion IN ({placeholders})")
        params.extend(tipo_seleccionado)

    if partido_seleccionado:
        placeholders = ','.join(['?'] * len(partido_seleccionado))
        where_conditions.append(f"partido_ci IN ({placeholders})")
        params.extend(partido_seleccionado)

    if candidato_busqueda:
        where_conditions.append("nombre_candidato LIKE ?")
        params.append(f'%{candidato_busqueda}%')

    where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"

    consulta_filtrada = f"""
    SELECT 
        candidato_id,
        nombre_candidato,
        partido_ci,
        tipo_eleccion,
        division_territorial,
        numero_de_votos
    FROM resultados_electorales 
    WHERE {where_clause}
    ORDER BY numero_de_votos DESC
    LIMIT 100;
    """

    datos_filtrados = dashboard.ejecutar_consulta(consulta_filtrada, params)

    st.subheader(f"Resultados Filtrados ({len(datos_filtrados)} registros)")
    st.dataframe(
        datos_filtrados,
        use_container_width=True,
        height=400
    )

with col2:
    st.subheader("📋 Resumen de Filtros")
    st.write(f"**Tipos de elección:** {len(tipo_seleccionado)}")
    st.write(f"**Partidos seleccionados:** {len(partido_seleccionado)}")
    st.write(f"**Registros encontrados:** {len(datos_filtrados)}")

    if len(datos_filtrados) > 0:
        st.metric("Votos totales filtrados", f"{datos_filtrados['numero_de_votos'].sum():,}")

        # Botón para exportar
        csv = datos_filtrados.to_csv(index=False)
        st.download_button(
            label="📥 Exportar resultados CSV",
            data=csv,
            file_name=f"resultados_filtrados_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# SECCIÓN 4: ANÁLISIS COMPARATIVO
st.header("⚖️ Análisis Comparativo")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Comparar Partidos")
    partido_a, partido_b = st.selectbox(
        "Seleccionar partido A:", partidos, key="partido_a"
    ), st.selectbox(
        "Seleccionar partido B:", partidos, key="partido_b", index=1
    )

    if partido_a and partido_b:
        consulta_comparacion = """
        SELECT 
            partido_ci,
            tipo_eleccion,
            COUNT(*) as candidatos,
            SUM(numero_de_votos) as total_votos
        FROM resultados_electorales 
        WHERE partido_ci IN (?, ?)
        GROUP BY partido_ci, tipo_eleccion
        ORDER BY partido_ci, total_votos DESC;
        """

        comparacion = dashboard.ejecutar_consulta(consulta_comparacion, (partido_a, partido_b))

        if not comparacion.empty:
            fig_comparacion = px.bar(
                comparacion,
                x='tipo_eleccion',
                y='total_votos',
                color='partido_ci',
                barmode='group',
                title=f'Comparación: {partido_a} vs {partido_b}',
                text='total_votos'
            )
            st.plotly_chart(fig_comparacion, use_container_width=True)

with col2:
    st.subheader("Evolución por Tipo de Elección")

    # Métricas comparativas por tipo
    consulta_metricas_tipo = """
    SELECT 
        tipo_eleccion,
        COUNT(*) as candidatos,
        SUM(numero_de_votos) as total_votos,
        ROUND(AVG(numero_de_votos), 2) as promedio_votos,
        MAX(numero_de_votos) as max_votos
    FROM resultados_electorales 
    GROUP BY tipo_eleccion;
    """

    metricas_tipo = dashboard.ejecutar_consulta(consulta_metricas_tipo)

    for _, row in metricas_tipo.iterrows():
        with st.expander(f"📊 {row['tipo_eleccion']}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Candidatos", row['candidatos'])
            with col2:
                st.metric("Votos Totales", f"{row['total_votos']:,}")
            with col3:
                st.metric("Promedio", row['promedio_votos'])

# FOOTER
st.markdown("---")
st.markdown(
    "**Dashboard desarrollado con Streamlit** | "
    "Datos electorales Nuevo León 2021 | "
    f"Última actualización: {datetime.now().strftime('%d/%m/%Y %H:%M')}"
)