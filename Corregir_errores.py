import sqlite3
import pandas as pd


def analizar_problema_gobernador():
    """Analizar el problema de los candidatos a gobernador"""
    conn = sqlite3.connect('elecciones_nl_2021.db')

    print("🔍 ANALIZANDO CANDIDATOS A GOBERNADOR:")
    print("=" * 50)

    # Ver todos los registros de gobernador
    query = """
    SELECT 
        candidato_id,
        nombre_candidato,
        division_territorial,
        numero_de_votos
    FROM resultados_electorales 
    WHERE tipo_eleccion = 'GOBERNADOR'
    ORDER BY nombre_candidato, division_territorial;
    """

    df = pd.read_sql_query(query, conn)
    print(f"Total de registros de gobernador: {len(df)}")
    print(f"Candidatos únicos: {df['nombre_candidato'].nunique()}")

    print("\n📋 Lista completa de registros:")
    print(df)

    print("\n👥 Candidatos únicos y sus distritos:")
    candidatos_unicos = df.groupby('nombre_candidato').agg({
        'division_territorial': lambda x: list(x),
        'numero_de_votos': 'sum',
        'candidato_id': 'count'
    }).reset_index()

    candidatos_unicos.columns = ['Candidato', 'Distritos', 'Total_Votos', 'Registros']
    print(candidatos_unicos)

    conn.close()

    return df


# Ejecutar análisis
df_gobernador = analizar_problema_gobernador()


def corregir_datos_gobernador():
    """Corregir los datos de gobernador para tener solo 7 candidatos únicos"""
    conn = sqlite3.connect('elecciones_nl_2021.db')

    # Primero, identificar los candidatos únicos
    query_candidatos_unicos = """
    SELECT DISTINCT nombre_candidato
    FROM resultados_electorales 
    WHERE tipo_eleccion = 'GOBERNADOR';
    """

    candidatos_unicos = pd.read_sql_query(query_candidatos_unicos, conn)
    print(f"🎯 Candidatos únicos a gobernador encontrados: {len(candidatos_unicos)}")
    print(candidatos_unicos)

    # OPCIÓN 1: Consolidar votos por candidato (recomendado)
    print("\n📊 OPCIÓN 1: Consolidar votos por candidato (RECOMENDADO)")

    query_consolidado = """
    SELECT 
        nombre_candidato,
        partido_ci,
        SUM(numero_de_votos) as total_votos,
        COUNT(DISTINCT division_territorial) as distritos,
        GROUP_CONCAT(DISTINCT division_territorial) as lista_distritos
    FROM resultados_electorales 
    WHERE tipo_eleccion = 'GOBERNADOR'
    GROUP BY nombre_candidato, partido_ci
    ORDER BY total_votos DESC;
    """

    consolidado = pd.read_sql_query(query_consolidado, conn)
    print("📈 Resultados consolidados por candidato:")
    print(consolidado)

    # OPCIÓN 2: Crear una nueva tabla corregida
    print("\n🔄 Creando tabla corregida...")

    # Eliminar tabla si existe
    conn.execute("DROP TABLE IF EXISTS gobernador_corregido")

    # Crear nueva tabla corregida
    conn.execute("""
    CREATE TABLE gobernador_corregido AS
    SELECT 
        MIN(candidato_id) as candidato_id,
        2021 as anno,
        nombre_candidato,
        SUM(numero_de_votos) as numero_de_votos,
        'Nuevo León' as division_territorial,
        nombre_normalizado,
        partido_ci,
        'GOBERNADOR' as tipo_eleccion,
        CURRENT_TIMESTAMP as created_at
    FROM resultados_electorales 
    WHERE tipo_eleccion = 'GOBERNADOR'
    GROUP BY nombre_candidato, nombre_normalizado, partido_ci
    ORDER BY numero_de_votos DESC;
    """)

    conn.commit()

    # Verificar la nueva tabla
    query_verificar = "SELECT * FROM gobernador_corregido;"
    resultado_corregido = pd.read_sql_query(query_verificar, conn)

    print(f"✅ Tabla corregida creada con {len(resultado_corregido)} candidatos únicos:")
    print(resultado_corregido[['nombre_candidato', 'partido_ci', 'numero_de_votos']])

    conn.close()

    return resultado_corregido


# Ejecutar corrección
gobernador_corregido = corregir_datos_gobernador()