import streamlit as st
import cv2
import pytesseract
import re
import pandas as pd
from PIL import Image
import numpy as np
import io

# --- CONFIGURACIÓN ---
ARCHIVO_EXCEL = 'datos_cedulas_colombia.xlsx'

# --- FUNCIONES DE PROCESAMIENTO DE IMAGEN Y OCR ---

def corregir_perspectiva_y_procesar(imagen_pil):
    """
    Función principal de visión por computadora. Detecta los bordes del carnet,
    corrige la perspectiva para obtener una vista plana y la procesa para OCR.
    """
    try:
        open_cv_image = np.array(imagen_pil)
        img = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
        
        # Guardar una copia para dibujar el contorno
        img_con_contorno = img.copy()
        
        # 1. Pre-procesamiento para detección de bordes
        img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gris, (5, 5), 0)
        img_canny = cv2.Canny(img_blur, 75, 200) # Ajuste de umbrales

        # 2. Encontrar contornos
        contornos, _ = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:5] # Tomar los 5 más grandes
        
        carnet_contour = None
        for contorno in contornos:
            perimetro = cv2.arcLength(contorno, True)
            approx = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)
            if len(approx) == 4:
                carnet_contour = approx
                break
        
        # --- MEJORA: DEPURACIÓN VISUAL ---
        if carnet_contour is not None:
            cv2.drawContours(img_con_contorno, [carnet_contour], -1, (0, 255, 0), 3)
            st.session_state.imagen_con_contorno = Image.fromarray(cv2.cvtColor(img_con_contorno, cv2.COLOR_BGR2RGB))
        else:
            st.session_state.imagen_con_contorno = None # Limpiar si no se encontró

        if carnet_contour is None:
            st.warning("No se pudo detectar un contorno de 4 esquinas. Usando la imagen completa.")
            return mejorar_imagen_para_ocr_simple(imagen_pil)

        # 3. Transformación de perspectiva
        puntos_origen = reordenar_puntos(carnet_contour.reshape(4, 2))
        
        ancho_a = np.sqrt(((puntos_origen[0][0] - puntos_origen[1][0])**2) + ((puntos_origen[0][1] - puntos_origen[1][1])**2))
        ancho_b = np.sqrt(((puntos_origen[2][0] - puntos_origen[3][0])**2) + ((puntos_origen[2][1] - puntos_origen[3][1])**2))
        ancho_max = max(int(ancho_a), int(ancho_b))

        alto_a = np.sqrt(((puntos_origen[0][0] - puntos_origen[3][0])**2) + ((puntos_origen[0][1] - puntos_origen[3][1])**2))
        alto_b = np.sqrt(((puntos_origen[1][0] - puntos_origen[2][0])**2) + ((puntos_origen[1][1] - puntos_origen[2][1])**2))
        alto_max = max(int(alto_a), int(alto_b))
        
        puntos_destino = np.float32([[0, 0], [ancho_max, 0], [ancho_max, alto_max], [0, alto_max]])
        
        matriz = cv2.getPerspectiveTransform(puntos_origen, puntos_destino)
        img_escaneada = cv2.warpPerspective(img, matriz, (ancho_max, alto_max))
        
        # 4. Procesamiento final para OCR
        img_escaneada_gris = cv2.cvtColor(img_escaneada, cv2.COLOR_BGR2GRAY)
        img_final = cv2.adaptiveThreshold(img_escaneada_gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        return Image.fromarray(img_final)
        
    except Exception as e:
        st.error(f"Error en el procesamiento de visión por computadora: {e}")
        return None

def mejorar_imagen_para_ocr_simple(imagen_pil):
    img_array = np.array(imagen_pil.convert('L'))
    img_binaria = cv2.adaptiveThreshold(img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return Image.fromarray(img_binaria)

def reordenar_puntos(puntos):
    rect = np.zeros((4, 2), dtype="float32")
    s = puntos.sum(axis=1)
    rect[0] = puntos[np.argmin(s)]
    rect[2] = puntos[np.argmax(s)]
    diff = np.diff(puntos, axis=1)
    rect[1] = puntos[np.argmin(diff)]
    rect[3] = puntos[np.argmax(diff)]
    return rect

def extraer_texto_de_imagen(imagen_procesada):
    if imagen_procesada is None: return ""
    config = '-l spa --psm 6'
    return pytesseract.image_to_string(imagen_procesada, config=config)

def estructurar_datos_extraidos(texto_crudo):
    """
    Usa Regex para encontrar y estructurar los datos, con un Plan B si falla.
    """
    datos = {"Apellidos": "No encontrado", "Nombres": "No encontrado", "NUIP": "No encontrado"}
    
    # --- Intento 1: Búsqueda por etiquetas ---
    match_apellidos = re.search(r'(?:Apellidos|Apelidos)\s*([A-ZÁÉÍÓÚÑ\s]+)', texto_crudo, re.IGNORECASE)
    if match_apellidos:
        datos["Apellidos"] = " ".join(match_apellidos.group(1).strip().split('\n')[0].split()[:2])

    match_nombres = re.search(r'Nombres\s*([A-ZÁÉÍÓÚÑ\s]+)', texto_crudo, re.IGNORECASE)
    if match_nombres:
        datos["Nombres"] = " ".join(match_nombres.group(1).strip().split('\n')[0].split()[:2])

    match_nuip = re.search(r'(\d{1,2}\.\d{3}\.\d{3})', texto_crudo)
    if match_nuip:
        datos["NUIP"] = match_nuip.group(1).strip()
    
    # --- MEJORA: PLAN B SI NO SE ENCONTRARON NOMBRES ---
    if datos["Apellidos"] == "No encontrado" and datos["Nombres"] == "No encontrado":
        lineas = [linea.strip() for linea in texto_crudo.split('\n') if linea.strip()]
        candidatos_nombres = []
        patron_nombre = re.compile(r'^[A-ZÁÉÍÓÚÑ]{2,}(?:\s+[A-ZÁÉÍÓÚÑ]{2,}){1,2}$')
        
        for linea in lineas:
            if patron_nombre.match(linea):
                if "REPUBLICA" not in linea and "COLOMBIA" not in linea and "CIUDADANIA" not in linea:
                    candidatos_nombres.append(linea)
        
        if len(candidatos_nombres) >= 2:
            st.info("Búsqueda por etiquetas falló. Usando Plan B para encontrar nombres.")
            datos["Apellidos"] = candidatos_nombres[0]
            datos["Nombres"] = candidatos_nombres[1]
    
    return datos

# --- INTERFAZ DE STREAMLIT ---
st.set_page_config(page_title="Lector de Carnets OCR", layout="centered")
st.title("🚀 Lector de Carnets con Visión Artificial")
st.info("Consejo: Coloca el carnet sobre un fondo oscuro y de color uniforme para mejores resultados.")

# Limpiar estado en cada recarga para una nueva sesión
if 'datos_capturados' not in st.session_state:
    st.session_state.datos_capturados = []
st.session_state.imagen_con_contorno = None

foto_buffer = st.camera_input("Toma una foto del carnet (horizontalmente)")

if foto_buffer:
    st.info("Procesando imagen... esto puede tardar unos segundos.")
    img_pil = Image.open(foto_buffer)

    imagen_corregida = corregir_perspectiva_y_procesar(img_pil)

    if st.session_state.get('imagen_con_contorno'):
        st.subheader("Paso 1: Detección de Bordes")
        st.image(st.session_state.imagen_con_contorno, caption="El recuadro verde muestra lo que el algoritmo detectó como el carnet.", use_container_width=True)

    if imagen_corregida:
        texto_extraido = extraer_texto_de_imagen(imagen_corregida)
        datos_estructurados = estructurar_datos_extraidos(texto_extraido)
        st.session_state.ultimo_dato = datos_estructurados

        st.subheader("Paso 2: Imagen Corregida para Análisis")
        st.image(imagen_corregida, caption="Esta es la imagen que se analiza.", use_container_width=True)
        
        st.subheader("Paso 3: Datos Extraídos")
        st.json(datos_estructurados)
        
        with st.expander("Ver Texto Crudo Extraído por OCR"):
            st.text(texto_extraido if texto_extraido else "No se pudo extraer texto.")

        if st.button("Confirmar y Añadir a la Lista"):
            st.session_state.datos_capturados.append(st.session_state.ultimo_dato)
            st.success("¡Datos añadidos!")

if st.session_state.datos_capturados:
    st.subheader("Registros Capturados")
    df = pd.DataFrame(st.session_state.datos_capturados)
    st.dataframe(df)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Datos')
    excel_data = output.getvalue()
    
    st.download_button(
        label="📥 Descargar todo como Excel", data=excel_data,
        file_name=ARCHIVO_EXCEL, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
