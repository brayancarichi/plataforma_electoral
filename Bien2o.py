from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import os


lista_partidos = [
    'PAN','PRI','p3','p4','p5','p6','p7','p8','p9','p10','p11','p12','p13','p14','p15','p16'
]
lista2 = ['v_acumulados', 'no_registrados', 'nulos']  # votos extra


def guardar_checkpoint(distrito, seccion, casilla):
    with open("checkpoint.txt", "w") as f:
        f.write(f"{distrito}|{seccion}|{casilla}")


def leer_checkpoint():
    try:
        with open("checkpoint.txt", "r") as f:
            return f.read().strip().split("|")
    except FileNotFoundError:
        return None, None, None


def casilla_ya_guardada(distrito, seccion, casilla):
    nombre_archivo = f"Elecciones_2024_{distrito.replace(' ', '_')}_{seccion.replace(' ', '_')}_{casilla.replace(' ', '_')}.csv"
    return os.path.exists(nombre_archivo)


opts = Options()
opts.add_argument(
    "user-agent=Mozilla/5.0 (iPhone; CPU iPhone OS 15_4 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/101.0.4951.44 Mobile/15E148 Safari/604.1"
)
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
wait = WebDriverWait(driver, 10)


driver.get('https://computo24.ieepcnl.mx/R02D.htm')
sleep(5)


def seleccionar_opcion_dropdown(id_dropdown, texto_opcion):
    dropdown = wait.until(EC.element_to_be_clickable((By.ID, id_dropdown)))
    dropdown.click()
    sleep(1)
    opciones = driver.find_elements(By.XPATH, '//a[@class="dropdown-item"]')
    for op in opciones:
        if op.text.strip() == texto_opcion:
            op.click()
            return True
    return False


dropdown_distrito = wait.until(EC.element_to_be_clickable((By.ID, "dDistrito")))
dropdown_distrito.click()
sleep(1)
distritos = [
    op.text.strip() for op in driver.find_elements(By.XPATH, '//a[@class="dropdown-item"]')
    if op.text.strip() and op.text.strip().lower() != 'todos'
]
dropdown_distrito.click()


ultimo_distrito, ultima_seccion, ultima_casilla = leer_checkpoint()
saltando = bool(ultimo_distrito)

datos_generales = []
casillas_fallidas = []


for distrito in distritos:
    print(f"\nSeleccionando distrito: {distrito}")

    if not seleccionar_opcion_dropdown("dDistrito", distrito):
        print(f"Error al seleccionar distrito {distrito}")
        continue

    sleep(1)


    try:
        dropdown_seccion = wait.until(EC.element_to_be_clickable((By.ID, "dSeccion")))
    except Exception:
        print(f"No hay dropdown de secciones para el distrito {distrito}, saltando...")
        continue

    dropdown_seccion.click()
    sleep(1)
    secciones = [
        op.text.strip() for op in driver.find_elements(By.XPATH, '//a[@class="dropdown-item"]')
        if op.text.strip()
    ]
    dropdown_seccion.click()

    for seccion in secciones:
        print(f"  📘 Seleccionando sección: {seccion}")

        if not seleccionar_opcion_dropdown("dSeccion", seccion):
            print(f"Error al seleccionar sección {seccion}")
            continue

        sleep(1)


        try:
            dropdown_casilla = wait.until(EC.element_to_be_clickable((By.ID, "dCasilla")))
        except Exception:
            print(f"No hay dropdown de casillas para la sección {seccion} en distrito {distrito}, saltando...")
            continue

        dropdown_casilla.click()
        sleep(1)
        casillas = [
            op.text.strip() for op in driver.find_elements(By.XPATH, '//a[@class="dropdown-item"]')
            if op.text.strip()
        ]
        dropdown_casilla.click()

        for casilla in casillas:

            if saltando:
                if distrito == ultimo_distrito and seccion == ultima_seccion and casilla == ultima_casilla:
                    saltando = False
                else:
                    print(f"Saltando {distrito} - {seccion} - {casilla}")
                    continue

            print(f"    Seleccionando casilla: {casilla}")

            if casilla_ya_guardada(distrito, seccion, casilla):
                print(f"📂 Ya existe archivo para {casilla}, saltando...")
                continue

            if not seleccionar_opcion_dropdown("dCasilla", casilla):
                print(f"Error al seleccionar casilla {casilla}")
                casillas_fallidas.append((distrito, seccion, casilla))
                continue

            sleep(1)

            try:

                votos = driver.find_elements(By.XPATH, '//p[@class="votos"]')
                extras = driver.find_elements(By.XPATH, '//p[@class="col-12 cantidad"]')
                lista_votos = [v.text.strip() for v in votos if v.text.strip() != '']
                lista_extras = [j.text.strip() for j in extras if j.text.strip() != '']

                esperado_partidos = len(lista_partidos)
                if len(lista_votos) != esperado_partidos:
                    print(f"Mismatch en número de votos: esperados {esperado_partidos}, recibidos {len(lista_votos)}")
                    casillas_fallidas.append((distrito, seccion, casilla))
                    continue

                votos_partidos = lista_votos[:esperado_partidos]
                esperado_extras = len(lista2)

                if len(lista_extras) < esperado_extras:
                    print(f"Faltan votos extra: completando con ceros.")
                    lista_extras += ['0'] * (esperado_extras - len(lista_extras))
                elif len(lista_extras) > esperado_extras:
                    lista_extras = lista_extras[:esperado_extras]

                votos_extra = lista_extras

                fila = {'distrito': distrito, 'sección': seccion, 'casilla': casilla}

                for i, partido in enumerate(lista_partidos):
                    fila[partido] = votos_partidos[i]

                for i, campo in enumerate(lista2):
                    fila[campo] = votos_extra[i]

                datos_generales.append(fila)


                df_final = pd.DataFrame([fila])
                nombre_archivo = f"Elecciones_2024_diputaciones_{distrito.replace(' ', '_')}_{seccion.replace(' ', '_')}_{casilla.replace(' ', '_')}.csv"
                df_final.to_csv(nombre_archivo, index=False, encoding='utf-8')
                print(f"💾 Datos guardados en {nombre_archivo}")


                guardar_checkpoint(distrito, seccion, casilla)

            except Exception as e:
                print(f"Error al procesar casilla {casilla}: {e}")
                casillas_fallidas.append((distrito, seccion, casilla))
                continue


if datos_generales:
    df_general = pd.DataFrame(datos_generales)
    df_general.to_csv("Elecciones_2024_todos_los_distritos_secciones_casillas_diputaciones.csv", index=False, encoding='utf-8')
    print("\nArchivo general guardado: Elecciones_2024_todos_los_distritos_secciones_casillas_diputaciones.csv")


if casillas_fallidas:
    pd.DataFrame(casillas_fallidas, columns=['distrito', 'seccion', 'casilla']).to_csv("casillas_fallidas.csv", index=False)
    print("Algunas casillas fallaron. Ver archivo: casillas_fallidas.csv")


driver.quit()

#Instalar las siguientes librerias:

#selenium
#webdriver-manager
#pandas