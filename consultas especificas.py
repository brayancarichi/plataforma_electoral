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