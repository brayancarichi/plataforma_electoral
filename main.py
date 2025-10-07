import sqlite3
import pandas as pd
import os


def crear_base_datos_sqlite():
    """Crear base de datos SQLite (no necesita instalación)"""
    # Eliminar si existe para empezar fresco
    if os.path.exists('elecciones_nl_2021.db'):
        os.remove('elecciones_nl_2021.db')

    conn = sqlite3.connect('elecciones_nl_2021.db')
    cur = conn.cursor()

    # Crear tabla
    cur.execute("""
        CREATE TABLE resultados_electorales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidato_id VARCHAR(100) UNIQUE NOT NULL,
            anno INTEGER NOT NULL,
            nombre_candidato VARCHAR(300) NOT NULL,
            numero_de_votos INTEGER,
            division_territorial VARCHAR(150),
            nombre_normalizado VARCHAR(300),
            partido_ci VARCHAR(150),
            tipo_eleccion VARCHAR(20) NOT NULL CHECK (tipo_eleccion IN ('MUNICIPAL', 'DIPUTADO', 'GOBERNADOR')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Crear índices
    cur.execute("CREATE INDEX idx_tipo_eleccion ON resultados_electorales(tipo_eleccion)")
    cur.execute("CREATE INDEX idx_partido ON resultados_electorales(partido_ci)")
    cur.execute("CREATE INDEX idx_division ON resultados_electorales(division_territorial)")

    conn.commit()
    conn.close()
    print("✅ Base de datos SQLite creada: elecciones_nl_2021.db")


def cargar_datos_sqlite():
    """Cargar datos a SQLite"""
    conn = sqlite3.connect('elecciones_nl_2021.db')

    archivos = {
        'candidatos_ayuntamientos_con_id_anno_2021.csv': 'MUNICIPAL',
        'candidatos_diputaciones_con_id_2021.csv': 'DIPUTADO',
        'candidatos_gobernador_con_id_anno_2021.csv': 'GOBERNADOR'
    }

    for archivo, tipo_eleccion in archivos.items():
        if os.path.exists(archivo):
            df = pd.read_csv(archivo)
            print(f"📖 Cargando {archivo}: {len(df)} registros")

            # Estandarizar columnas
            if 'municipio' in df.columns:
                df.rename(columns={'municipio': 'division_territorial'}, inplace=True)
            elif 'distrito' in df.columns:
                df.rename(columns={'distrito': 'division_territorial'}, inplace=True)
            elif 'nombre_distrito' in df.columns:
                df.rename(columns={'nombre_distrito': 'division_territorial'}, inplace=True)

            # Limpiar votos
            df['numero_de_votos'] = df['numero_de_votos'].astype(str).str.replace(',', '').astype(int)
            df['tipo_eleccion'] = tipo_eleccion

            # Insertar datos
            df.to_sql('resultados_electorales', conn, if_exists='append', index=False)
            print(f"✅ {archivo} cargado: {len(df)} registros")
        else:
            print(f"⚠️ Archivo {archivo} no encontrado")

    conn.close()


def consultas_sqlite():
    """Ejecutar consultas en SQLite"""
    conn = sqlite3.connect('elecciones_nl_2021.db')

    consultas = {
        'Total por tipo de elección': """
            SELECT tipo_eleccion, COUNT(*) as candidatos, SUM(numero_de_votos) as total_votos
            FROM resultados_electorales 
            GROUP BY tipo_eleccion ORDER BY total_votos DESC;
        """,

        'Top 10 candidatos': """
            SELECT tipo_eleccion, nombre_candidato, partido_ci, numero_de_votos
            FROM resultados_electorales 
            ORDER BY numero_de_votos DESC LIMIT 10;
        """
    }

    for nombre, consulta in consultas.items():
        print(f"\n🔍 {nombre}:")
        print("-" * 50)
        df = pd.read_sql_query(consulta, conn)
        print(df)

    conn.close()


# EJECUCIÓN COMPLETA
print("🚀 INICIANDO PROCESO CON SQLite...")
crear_base_datos_sqlite()
cargar_datos_sqlite()
consultas_sqlite()
print("\n🎯 PROCESO COMPLETADO!")
print("📊 Base de datos creada: elecciones_nl_2021.db")