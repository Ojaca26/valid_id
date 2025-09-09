import streamlit as st
import cv2
import pytesseract
import re
import pandas as pd
from PIL import Image
import numpy as np
import io
import google.generativeai as genai
import json

# --- CONFIGURACIÓN ---
ARCHIVO_EXCEL = 'datos_cedulas_colombia.xlsx'

# Configurar la API de Gemini (la clave se toma de st.secrets)
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    GEMINI_CONFIGURADO = True
except Exception as e:
    st.error("Error al configurar la API de Gemini. Asegúrate de que tu clave de API esté en el archivo secrets.toml.")
    GEMINI_CONFIGURADO = False

# --- FUNCIONES DE PROCESAMIENTO Y EXTRACCIÓN ---

def corregir_perspectiva(imagen_pil):
    """Detecta los bordes del carnet y corrige la perspectiva."""
    try:
        open_cv_image = np.array(imagen_pil)
        img = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
        
        img_con_contorno = img.copy()
        
        img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gris, (5, 5), 0)
        img_canny = cv2.Canny(img_blur, 75, 200)

        contornos, _ = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:5]
        
        carnet_contour = None
        for contorno in contornos:
            perimetro = cv2.arcLength(contorno, True)
            approx = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)
            if len(approx) == 4:
                carnet_contour = approx
                break
        
        if carnet_contour is not None:
            cv2.drawContours(img_con_contorno, [carnet_contour], -1, (0, 255, 0), 3)
            st.session_state.imagen_con_contorno = Image.fromarray(cv2.cvtColor(img_con_contorno, cv2.COLOR_BGR2RGB))
        else:
            st.session_state.imagen_con_contorno = None

        if carnet_contour is None:
            st.warning("No se pudo detectar un contorno de 4 esquinas. Usando la imagen completa.")
            return imagen_pil # Devolver la imagen original si no hay contorno

        puntos_origen = reordenar_puntos(carnet_contour.reshape(4, 2))
        
        ancho_a = np.sqrt(((puntos_origen[0][0] - puntos_origen[1][0])**2) + ((puntos_origen[0][1] - puntos_origen[1][1])**2))
        ancho_b = np.sqrt(((puntos_origen[2][0] - puntos_origen[3][0])**2) + ((puntos_origen[2][1] - puntos_origen[3][1])**2))
        ancho_max = max(int(ancho_a), int(ancho_b))

        alto_a = np.sqrt(((puntos_origen[0][0] - puntos_origen[3][0])**2) + ((puntos_origen[0][1] - puntos_origen[3][1])**2))
        alto_b = np.sqrt(((puntos_origen[1][0] - puntos_origen[2][0])**2) + ((puntos_origen[1][1] - puntos_origen[2][1])**2))
        alto_max = max(int(alto_a), int(alto_b))
        
        puntos_destino = np.float32([[0, 0], [ancho_max, 0], [ancho_max, alto_max], [0, alto_max]])
        
        matriz = cv2.getPerspectiveTransform(puntos_origen, puntos_destino)
        img_escaneada_bgr = cv2.warpPerspective(img, matriz, (ancho_max, alto_max))
        
        return Image.fromarray(cv2.cvtColor(img_escaneada_bgr, cv2.COLOR_BGR2RGB))
        
    except Exception as e:
        st.error(f"Error en la corrección de perspectiva: {e}")
        return imagen_pil # Devolver original en caso de error

def reordenar_puntos(puntos):
    rect = np.zeros((4, 2), dtype="float32")
    s = puntos.sum(axis=1)
    rect[0] = puntos[np.argmin(s)]
    rect[2] = puntos[np.argmax(s)]
    diff = np.diff(puntos, axis=1)
    rect[1] = puntos[np.argmin(diff)]
    rect[3] = puntos[np.argmax(diff)]
    return rect

def extraer_datos_con_gemini(imagenes_pil):
    """
    Envía una o dos imágenes a la API de Gemini Vision y pide la extracción de datos.
    """
    if not GEMINI_CONFIGURADO:
        return {"Error": "API de Gemini no configurada."}

    # Modelo gemini-1.5-flash-latest es ideal para esto: rápido y eficiente
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    prompt_parts = [
        "Eres un experto en analizar cédulas de ciudadanía de Colombia, tanto el modelo antiguo (amarilla) como el nuevo (digital).",
        "Analiza la(s) siguiente(s) imagen(es) que pueden corresponder al anverso y reverso de una cédula.",
        "Extrae la siguiente información y devuélvela en un formato JSON estricto:",
        "- Apellidos",
        "- Nombres",
        "- NUIP o NUMERO (usa la etiqueta 'NUIP' para ambos)",
        "- Fecha de Nacimiento",
        "Si no encuentras un campo, usa el valor 'No encontrado'.",
        "Ejemplo de respuesta: {\"Apellidos\": \"PEREZ GOMEZ\", \"Nombres\": \"JUAN CARLOS\", \"NUIP\": \"12.345.678\", \"Fecha de Nacimiento\": \"01 ENE 1990\"}",
    ]
    
    # Añadir las imágenes al prompt
    for img in imagenes_pil:
        prompt_parts.append(img)
        
    try:
        response = model.generate_content(prompt_parts)
        # Limpiar la respuesta para que sea un JSON válido
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(json_text)
    except Exception as e:
        st.error(f"Error al contactar la API de Gemini: {e}")
        st.text("Respuesta cruda de la API:")
        st.text(response.text if 'response' in locals() else "No hubo respuesta.")
        return {"Error": "No se pudo procesar la respuesta de la IA."}

# --- INTERFAZ DE STREAMLIT ---
st.set_page_config(page_title="Lector de Cédulas IA", layout="wide")
st.title("🚀 Lector de Cédulas con IA (Gemini)")
st.info("Toma fotos claras del anverso y, si es necesario, del reverso de la cédula.")

if 'datos_capturados' not in st.session_state:
    st.session_state.datos_capturados = []

col1, col2 = st.columns(2)
with col1:
    foto_anverso_buffer = st.camera_input("1. Toma una foto del **Anverso** (lado principal)")

with col2:
    foto_reverso_buffer = st.camera_input("2. Toma una foto del **Reverso** (opcional, para cédula antigua)")

if foto_anverso_buffer:
    st.info("Procesando imagen(es)... esto puede tardar unos segundos.")
    
    imagenes_a_procesar = []
    
    # Procesar Anverso
    img_anverso_pil = Image.open(foto_anverso_buffer)
    img_anverso_corregida = corregir_perspectiva(img_anverso_pil)
    imagenes_a_procesar.append(img_anverso_corregida)
    st.subheader("Anverso Corregido")
    st.image(img_anverso_corregida, use_container_width=True)

    # Procesar Reverso si existe
    if foto_reverso_buffer:
        img_reverso_pil = Image.open(foto_reverso_buffer)
        img_reverso_corregida = corregir_perspectiva(img_reverso_pil)
        imagenes_a_procesar.append(img_reverso_corregida)
        st.subheader("Reverso Corregido")
        st.image(img_reverso_corregida, use_container_width=True)
    
    with st.spinner('La IA está analizando los documentos...'):
        datos_estructurados = extraer_datos_con_gemini(imagenes_a_procesar)
    
    st.subheader("Resultado del Análisis de IA")
    st.json(datos_estructurados)
    
    st.session_state.ultimo_dato = datos_estructurados

    if "Error" not in datos_estructurados and st.button("Confirmar y Añadir a la Lista"):
        st.session_state.datos_capturados.append(st.session_state.ultimo_dato)
        st.success("¡Datos añadidos!")
        st.rerun()

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
