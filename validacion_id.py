import streamlit as st
import cv2
import pytesseract
import re
import pandas as pd
import os
from PIL import Image
import numpy as np
import io

# --- CONFIGURACIÓN DE TESSERACT (Para Streamlit Cloud) ---
# Streamlit Cloud instalará Tesseract a través del archivo packages.txt,
# por lo que no es necesario especificar la ruta del ejecutable aquí.

# Nombre del archivo Excel donde se guardarán los datos
ARCHIVO_EXCEL = 'datos_cedulas_colombia.xlsx'

# --- FUNCIONES DE PROCESAMIENTO (Las hemos mejorado) ---

def mejorar_imagen_para_ocr(imagen_pil):
    """
    Toma una imagen en formato PIL (de Streamlit) y la procesa con OpenCV.
    """
    try:
        # Convertir de PIL a formato OpenCV (numpy array)
        img = np.array(imagen_pil)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_gris = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        
        # Binarización Adaptativa para manejar distintas iluminaciones
        img_binaria = cv2.adaptiveThreshold(
            img_gris, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 2
        )
        return img_binaria
    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")
        return None

def extraer_texto_de_imagen(imagen_procesada):
    """
    Utiliza Tesseract OCR para extraer texto de la imagen procesada.
    """
    if imagen_procesada is None:
        return ""
    config = '-l spa --psm 3'
    texto_extraido = pytesseract.image_to_string(imagen_procesada, config=config)
    return texto_extraido

def estructurar_datos_extraidos(texto_crudo):
    """
    Usa Regex para encontrar y estructurar los datos de la cédula colombiana.
    """
    datos = {
        "Apellidos": "No encontrado", "Nombres": "No encontrado", "NUIP": "No encontrado",
        "Fecha de Nacimiento": "No encontrado", "Lugar de Nacimiento": "No encontrado",
        "Sexo": "No encontrado", "G.S": "No encontrado"
    }
    
    # Busca Apellidos (dos palabras en mayúsculas después de la etiqueta)
    match_apellidos = re.search(r'Apellidos\s*([A-ZÁÉÍÓÚ]+\s+[A-ZÁÉÍÓÚ]+)', texto_crudo, re.IGNORECASE)
    if match_apellidos:
        datos["Apellidos"] = match_apellidos.group(1).strip()

    # Busca Nombres
    match_nombres = re.search(r'Nombres\s*([A-ZÁÉÍÓÚ]+\s+[A-ZÁÉÍÓÚ]+)', texto_crudo, re.IGNORECASE)
    if match_nombres:
        datos["Nombres"] = match_nombres.group(1).strip()

    # Busca NUIP (formato XX.XXX.XXX)
    match_nuip = re.search(r'(\d{1,2}\.\d{3}\.\d{3})', texto_crudo)
    if match_nuip:
        datos["NUIP"] = match_nuip.group(1).strip()

    # Busca Fecha de Nacimiento (DD MES AAAA)
    match_f_nac = re.search(r'(\d{1,2}\s+[A-Z]{3}\s+\d{4})', texto_crudo.replace('\n', ' '), re.IGNORECASE)
    if match_f_nac and "nacimiento" in texto_crudo.lower():
        datos["Fecha de Nacimiento"] = match_f_nac.group(1).strip()
        
    # Busca Lugar de Nacimiento
    match_l_nac = re.search(r'Lugar de nacimiento\s*([^\n]+)', texto_crudo, re.IGNORECASE)
    if match_l_nac:
        datos["Lugar de Nacimiento"] = match_l_nac.group(1).strip()

    # Busca Sexo
    match_sexo = re.search(r'Sexo\s+([FM])', texto_crudo, re.IGNORECASE)
    if match_sexo:
        datos["Sexo"] = match_sexo.group(1).strip()
        
    # Busca Grupo Sanguíneo
    match_gs = re.search(r'G\.S\s*([^\n]+)', texto_crudo, re.IGNORECASE)
    if match_gs:
        datos["G.S"] = match_gs.group(1).strip()

    return datos

# --- INTERFAZ DE STREAMLIT ---

st.set_page_config(page_title="Lector de Carnets OCR", layout="wide")
st.title("🚀 Lector de Carnets con OCR")
st.write("Usa la cámara de tu celular para tomar una foto nítida y con buena luz del carnet.")

# Inicializar el estado de la sesión para guardar los datos
if 'datos_capturados' not in st.session_state:
    st.session_state.datos_capturados = []

# Widget para tomar la foto
foto_buffer = st.camera_input("Haz clic aquí para activar la cámara")

if foto_buffer:
    st.info("Procesando imagen... por favor espera.")
    
    # Convertir el buffer de la foto a una imagen PIL
    img_pil = Image.open(foto_buffer)
    
    # Pipeline de procesamiento
    imagen_procesada = mejorar_imagen_para_ocr(img_pil)
    texto_extraido = extraer_texto_de_imagen(imagen_procesada)
    datos_estructurados = estructurar_datos_extraidos(texto_extraido)
    
    st.session_state.ultimo_dato = datos_estructurados

    # Mostrar resultados
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Imagen Capturada")
        st.image(img_pil, caption="Asegúrate que el texto sea legible.", use_column_width=True)
    
    with col2:
        st.subheader("Datos Extraídos para Verificación")
        st.json(datos_estructurados)
        
        with st.expander("Ver Texto Crudo Extraído por OCR"):
            st.text(texto_extraido)

    # Botón para confirmar y guardar los datos
    if st.button("Confirmar y Añadir a la Lista"):
        st.session_state.datos_capturados.append(st.session_state.ultimo_dato)
        st.success("¡Datos añadidos a la lista! Puedes tomar otra foto.")

# Mostrar tabla con todos los datos capturados
if st.session_state.datos_capturados:
    st.subheader("Registros Capturados")
    df = pd.DataFrame(st.session_state.datos_capturados)
    st.dataframe(df)

    # Convertir DataFrame a Excel en memoria para descarga
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Datos')
    
    excel_data = output.getvalue()
    
    st.download_button(
        label="📥 Descargar todo como Excel",
        data=excel_data,
        file_name=ARCHIVO_EXCEL,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
