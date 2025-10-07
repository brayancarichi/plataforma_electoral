import sqlite3
import pandas as pd


class ConsultasElectorales:
    def __init__(self, db_path='elecciones_nl_2021.db'):
        self.db_path = db_path

    def conectar(self):
        """Conectar a la base de datos"""
        return sqlite3.connect(self.db_path)

    def ejecutar_consulta(self, consulta, params=None):
        """Ejecutar cualquier consulta SQL y devolver DataFrame"""
        with self.conectar() as conn:
            return pd.read_sql_query(consulta, conn, params=params)

    def mostrar_estadisticas_generales(self):
        """Estadísticas básicas de la base de datos"""
        consulta = """
        SELECT 
            COUNT(*) as total_registros,
            COUNT(DISTINCT partido_ci) as partidos_unicos,
            COUNT(DISTINCT tipo_eleccion) as tipos_eleccion,
            COUNT(DISTINCT division_territorial) as divisiones_unicas,
            SUM(numero_de_votos) as total_votos
        FROM resultados_electorales;
        """
        return self.ejecutar_consulta(consulta)


# Instanciar la clase
db = ConsultasElectorales()

print("📊 ESTADÍSTICAS GENERALES:")
print("=" * 50)
stats = db.mostrar_estadisticas_generales()
print(stats)


def consultas_avanzadas():
    """Consultas específicas para análisis electoral"""

    consultas = {
        # 1. Total de votos por tipo de elección
        'votos_por_tipo': """
            SELECT 
                tipo_eleccion,
                COUNT(*) as cantidad_candidatos,
                SUM(numero_de_votos) as total_votos,
                ROUND(AVG(numero_de_votos), 2) as promedio_votos
            FROM resultados_electorales 
            GROUP BY tipo_eleccion 
            ORDER BY total_votos DESC;
        """,

        # 2. Top 10 candidatos más votados
        'top_candidatos': """
            SELECT 
                nombre_candidato,
                partido_ci,
                tipo_eleccion,
                division_territorial,
                numero_de_votos
            FROM resultados_electorales 
            ORDER BY numero_de_votos DESC 
            LIMIT 10;
        """,

        # 3. Ranking de partidos políticos
        'ranking_partidos': """
            SELECT 
                partido_ci,
                COUNT(*) as candidatos_presentados,
                SUM(numero_de_votos) as total_votos,
                ROUND(SUM(numero_de_votos) * 100.0 / (SELECT SUM(numero_de_votos) FROM resultados_electorales), 2) as porcentaje_total
            FROM resultados_electorales 
            GROUP BY partido_ci 
            ORDER BY total_votos DESC;
        """,

        # 4. Resultados por división territorial (municipio/distrito)
        'por_division': """
            SELECT 
                division_territorial,
                tipo_eleccion,
                COUNT(*) as candidatos,
                SUM(numero_de_votos) as votos_totales,
                MAX(numero_de_votos) as max_votos
            FROM resultados_electorales 
            GROUP BY division_territorial, tipo_eleccion
            ORDER BY votos_totales DESC;
        """,

        # 5. Buscar candidatos específicos
        'buscar_garcia': """
            SELECT 
                nombre_candidato,
                partido_ci,
                tipo_eleccion,
                division_territorial,
                numero_de_votos
            FROM resultados_electorales 
            WHERE nombre_candidato LIKE '%GARCIA%' OR nombre_candidato LIKE '%García%'
            ORDER BY numero_de_votos DESC;
        """,

        # 6. Distribución por tipo de elección y partido
        'tipo_y_partido': """
            SELECT 
                tipo_eleccion,
                partido_ci,
                COUNT(*) as candidatos,
                SUM(numero_de_votos) as votos
            FROM resultados_electorales 
            GROUP BY tipo_eleccion, partido_ci
            ORDER BY tipo_eleccion, votos DESC;
        """
    }

    # Ejecutar todas las consultas
    for nombre, consulta in consultas.items():
        print(f"\n🎯 {nombre.upper().replace('_', ' ')}:")
        print("-" * 60)
        resultado = db.ejecutar_consulta(consulta)
        print(resultado)
        print(f"📈 Registros encontrados: {len(resultado)}")


# Ejecutar consultas avanzadas
consultas_avanzadas()




#*///////*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*/*
class ConsultasParametrizadas:
    def __init__(self, db_path='elecciones_nl_2021.db'):
        self.db = ConsultasElectorales(db_path)

    def buscar_por_nombre(self, nombre):
        """Buscar candidatos por nombre (búsqueda flexible)"""
        consulta = """
            SELECT 
                nombre_candidato,
                partido_ci,
                tipo_eleccion,
                division_territorial,
                numero_de_votos
            FROM resultados_electorales 
            WHERE nombre_candidato LIKE ?
            ORDER BY numero_de_votos DESC;
        """
        return self.db.ejecutar_consulta(consulta, (f'%{nombre}%',))

    def resultados_por_partido(self, partido):
        """Resultados filtrados por partido político"""
        consulta = """
            SELECT 
                tipo_eleccion,
                division_territorial,
                nombre_candidato,
                numero_de_votos
            FROM resultados_electorales 
            WHERE partido_ci LIKE ?
            ORDER BY numero_de_votos DESC;
        """
        return self.db.ejecutar_consulta(consulta, (f'%{partido}%',))

    def top_por_division(self, division, limite=5):
        """Top candidatos por división territorial"""
        consulta = """
            SELECT 
                nombre_candidato,
                partido_ci,
                tipo_eleccion,
                numero_de_votos
            FROM resultados_electorales 
            WHERE division_territorial LIKE ?
            ORDER BY numero_de_votos DESC
            LIMIT ?;
        """
        return self.db.ejecutar_consulta(consulta, (f'%{division}%', limite))

    def comparar_partidos(self, partido1, partido2):
        """Comparar el desempeño de dos partidos"""
        consulta = """
            SELECT 
                partido_ci,
                tipo_eleccion,
                COUNT(*) as candidatos,
                SUM(numero_de_votos) as total_votos,
                ROUND(AVG(numero_de_votos), 2) as promedio_votos
            FROM resultados_electorales 
            WHERE partido_ci IN (?, ?)
            GROUP BY partido_ci, tipo_eleccion
            ORDER BY partido_ci, total_votos DESC;
        """
        return self.db.ejecutar_consulta(consulta, (partido1, partido2))


# Usar consultas parametrizadas
consultas_param = ConsultasParametrizadas()

print("\n🔍 CONSULTAS PARAMETRIZADAS:")
print("=" * 50)

# Ejemplos de uso
print("1. Buscar candidatos con 'Larrazabal':")
resultado = consultas_param.buscar_por_nombre('Larrazabal')
print(resultado)

print("\n2. Resultados del partido MORENA:")
resultado = consultas_param.resultados_por_partido('MORENA')
print(resultado)

print("\n3. Top 3 candidatos en San Pedro Garza García:")
resultado = consultas_param.top_por_division('San Pedro', 3)
print(resultado)