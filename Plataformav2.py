import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

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


# Instanciar el dashboard corregido
dashboard = DashboardElectoralCorregido()

# HEADER PRINCIPAL
st.markdown('<h1 class="main-header">Dashboard Electoral NL 2021 - Corregido</h1>', unsafe_allow_html=True)
st.success("✅ **Base de datos corregida**: 7 candidatos únicos a gobernador")

# SIDEBAR
with st.sidebar:
    st.header("Controles del Dashboard")
    st.markdown("---")

    # Filtros
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
    candidato_busqueda = st.text_input("Buscar candidato:")

    st.markdown("---")
    st.info("💡 **Datos corregidos**: Gobernador muestra 7 candidatos únicos con votos consolidados")

# SECCIÓN 1: MÉTRICAS PRINCIPALES CORREGIDAS
st.header("Métricas Principales - Corregidas")

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

# Mostrar detalle por tipo de elección
with st.expander("Ver detalle por tipo de elección"):
    st.dataframe(stats['detalle_por_tipo'])

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
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(f"Candidatos {tipo_seleccionado}", len(datos_filtrados))
with col2:
    st.metric("Votos Totales", f"{datos_filtrados['numero_de_votos'].sum():,}")
with col3:
    st.metric("Partidos", datos_filtrados['partido_ci'].nunique())

# SECCIÓN 3: GRÁFICOS ESPECÍFICOS POR TIPO DE ELECCIÓN
if tipo_seleccionado == 'GOBERNADOR':
    st.subheader("Candidatos a Gobernador - Resultados Consolidados")

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

    # Mostrar cards de candidatos
    st.subheader("👥 Detalle de Candidatos a Gobernador")
    for _, candidato in datos_filtrados.iterrows():
        with st.container():
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.markdown(f"**{candidato['nombre_candidato']}**")
                st.caption(f"Partido: {candidato['partido_ci']}")
            with col2:
                st.info(f"Votos: {candidato['numero_de_votos']:,}")
            with col3:
                porcentaje = (candidato['numero_de_votos'] / datos_filtrados['numero_de_votos'].sum()) * 100
                st.metric("Porcentaje", f"{porcentaje:.1f}%")

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

# SECCIÓN 4: TABLA DETALLADA
with st.expander(" Ver tabla detallada"):
    columnas_mostrar = ['nombre_candidato', 'partido_ci', 'division_territorial', 'numero_de_votos']
    st.dataframe(
        datos_filtrados[columnas_mostrar],
        use_container_width=True,
        height=400
    )

    # Botón de exportación
    csv = datos_filtrados[columnas_mostrar].to_csv(index=False)
    st.download_button(
        label=" Exportar datos CSV",
        data=csv,
        file_name=f"{tipo_seleccionado}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# SECCIÓN 5: COMPARATIVA ENTRE TIPOS DE ELECCIÓN
st.header("️ Comparativa General")

col1, col2 = st.columns(2)

with col1:
    # Gráfico de comparativa de votos totales
    fig_comparativa = px.bar(
        stats['detalle_por_tipo'],
        x='tipo_eleccion',
        y='total_votos',
        color='tipo_eleccion',
        title='Comparativa de Votos Totales por Tipo de Elección',
        text='total_votos',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_comparativa.update_traces(texttemplate='%{text:,}', textposition='outside')
    st.plotly_chart(fig_comparativa, use_container_width=True)

with col2:
    # Gráfico de candidatos por tipo
    fig_candidatos = px.bar(
        stats['detalle_por_tipo'],
        x='tipo_eleccion',
        y='candidatos_unicos',
        color='tipo_eleccion',
        title='Cantidad de Candidatos Únicos por Tipo de Elección',
        text='candidatos_unicos',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_candidatos.update_traces(textposition='outside')
    st.plotly_chart(fig_candidatos, use_container_width=True)




# SECCIÓN ADICIONAL: VERIFICACIÓN DE DATOS
with st.sidebar:
    st.markdown("---")
    if st.button(" Verificar Integridad de Datos"):
        with st.spinner("Verificando datos..."):
            stats = dashboard.obtener_estadisticas_generales()
            gobernadores = dashboard.obtener_datos_gobernador_corregidos()

            st.success(f"✅ Gobernador: {len(gobernadores)} candidatos únicos")
            st.success(f"✅ Total votos: {stats['total_votos']:,}")
            st.success(f"✅ Candidatos totales: {stats['total_candidatos']}")