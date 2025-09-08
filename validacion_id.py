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
        
        # 1. Pre-procesamiento para detección de bordes
        img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gris, (5, 5), 0)
        img_canny = cv2.Canny(img_blur, 50, 150)

        # 2. Encontrar contornos
        contornos, _ = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Ordenar contornos por área y encontrar el más grande que sea un cuadrilátero
        contornos = sorted(contornos, key=cv2.contourArea, reverse=True)
        carnet_contour = None
        for contorno in contornos:
            perimetro = cv2.arcLength(contorno, True)
            approx = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)
            if len(approx) == 4:
                carnet_contour = approx
                break
        
        if carnet_contour is None:
            st.warning("No se pudo detectar un contorno de 4 esquinas. Usando la imagen completa.")
            # Si no se detecta, se procesa la imagen original como fallback
            return mejorar_imagen_para_ocr_simple(imagen_pil)

        # 3. Transformación de perspectiva
        puntos_origen = np.float32(carnet_contour)
        
        # Reordenar los puntos para la transformación
        puntos_origen = reordenar_puntos(puntos_origen.reshape(4, 2))
        
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
        img_final = cv2.adaptiveThreshold(
            img_escaneada_gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return Image.fromarray(img_final) # Devolver como imagen PIL para mostrar
        
    except Exception as e:
        st.error(f"Error en el procesamiento de visión por computadora: {e}")
        return None

def mejorar_imagen_para_ocr_simple(imagen_pil):
    """Fallback si la detección de contornos falla."""
    img_array = np.array(imagen_pil.convert('L')) # Convertir a escala de grises
    img_binaria = cv2.adaptiveThreshold(img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return Image.fromarray(img_binaria)

def reordenar_puntos(puntos):
    """Reordena los 4 puntos del contorno: superior-izq, superior-der, inferior-der, inferior-izq."""
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
    config = '-l spa --psm 6' # PSM 6 asume un bloque de texto uniforme
    return pytesseract.image_to_string(imagen_procesada, config=config)

def estructurar_datos_extraidos(texto_crudo):
    """Usa Regex para encontrar y estructurar los datos."""
    datos = {"Apellidos": "No encontrado", "Nombres": "No encontrado", "NUIP": "No encontrado"}
    
    # Búsquedas más flexibles
    match_apellidos = re.search(r'(?:Apellidos|Apelidos|Apelldos)\s*([A-ZÁÉÍÓÚÑ\s]+)', texto_crudo, re.IGNORECASE)
    if match_apellidos:
        # Limpiar y tomar las dos primeras palabras si hay más
        nombres_completos = match_apellidos.group(1).strip().split('\n')[0]
        datos["Apellidos"] = " ".join(nombres_completos.split()[:2])

    match_nombres = re.search(r'(?:Nombres|Nombes)\s*([A-ZÁÉÍÓÚÑ\s]+)', texto_crudo, re.IGNORECASE)
    if match_nombres:
        nombres_completos = match_nombres.group(1).strip().split('\n')[0]
        datos["Nombres"] = " ".join(nombres_completos.split()[:2])

    match_nuip = re.search(r'(\d{1,2}\.\d{3}\.\d{3})', texto_crudo)
    if match_nuip:
        datos["NUIP"] = match_nuip.group(1).strip()
    
    return datos

# --- INTERFAZ DE STREAMLIT ---
st.set_page_config(page_title="Lector de Carnets OCR", layout="centered")
st.title("🚀 Lector de Carnets con Visión Artificial")
st.info("Consejo: Coloca el carnet sobre un fondo oscuro y de color uniforme para mejores resultados.")

if 'datos_capturados' not in st.session_state:
    st.session_state.datos_capturados = []

foto_buffer = st.camera_input("Toma una foto del carnet (horizontalmente)")

if foto_buffer:
    st.info("Procesando imagen... esto puede tardar unos segundos.")
    img_pil = Image.open(foto_buffer)

    # El nuevo pipeline de procesamiento
    imagen_corregida = corregir_perspectiva_y_procesar(img_pil)

    if imagen_corregida:
        texto_extraido = extraer_texto_de_imagen(imagen_corregida)
        datos_estructurados = estructurar_datos_extraidos(texto_extraido)
        
        st.session_state.ultimo_dato = datos_estructurados

        st.subheader("Imagen Corregida (Vista Plana)")
        st.image(imagen_corregida, caption="Esta es la imagen que se analiza.", use_container_width=True)
        
        st.subheader("Datos Extraídos para Verificación")
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
