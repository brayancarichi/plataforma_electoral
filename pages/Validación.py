import streamlit as st
import streamlit.components.v1 as components

st.title('Validación de la información obtenida')
st.text('Con el fin de que los resultados mostrados en esta plataforma sean de calidad, se realizó una doble validación de los datos electorales obtenidos tanto de la pagina web como de la base de datos de libre acceso. Esta base de datos también se encuentra disponible en IEE Nuevo León.')

with open("Validaciones/validacion_resultados_ayuntamiento_municipios.html", "r", encoding="utf-8") as f:
    html_content = f.read()

with open("Validaciones/validacion_resultados_diputacion_distritos.html","r", encoding="utf-8") as z:
    html_content2 = z.read()

st.subheader('Validación de la información de los resultados de las elecciones de ayuntamientos 2021')
components.html(html_content, height=600, scrolling=True)

st.subheader('Validación de la información de los resultados de las elecciones para diputaciones 2021 NL')
components.html(html_content2, height=600, scrolling=True)
