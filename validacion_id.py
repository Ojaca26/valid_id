import streamlit as st
import cv2
import pandas as pd
from PIL import Image
import numpy as np
import io
import google.generativeai as genai
import json
import re

# --- CONFIGURACIÓN ---
ARCHIVO_EXCEL = 'datos_cedulas_colombia.xlsx'

# Configurar la API de Gemini (la clave se toma de st.secrets)
try:
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error("La clave GEMINI_API_KEY no se encontró en los secretos de Streamlit.")
        GEMINI_CONFIGURADO = False
    else:
        genai.configure(api_key=api_key)
        GEMINI_CONFIGURADO = True
except (AttributeError, KeyError):
    st.warning("No se pudieron cargar los secretos de Streamlit. Asegúrate de que tu clave de API esté configurada si estás en producción.")
    GEMINI_CONFIGURADO = False


# --- FUNCIONES DE PROCESAMIENTO Y EXTRACCIÓN ---

def corregir_perspectiva(imagen_pil):
    """Detecta los bordes del carnet y corrige la perspectiva."""
    try:
        open_cv_image = np.array(imagen_pil)
        img = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
        
        total_image_area = img.shape[0] * img.shape[1]
        min_area_ratio = 0.1 # El contorno debe ser al menos el 10% del área de la imagen

        img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gris, (5, 5), 0)
        img_canny = cv2.Canny(img_blur, 75, 200)

        contornos, _ = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:5]
        
        carnet_contour = None
        for contorno in contornos:
            perimetro = cv2.arcLength(contorno, True)
            approx = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)
            if len(approx) == 4 and cv2.contourArea(approx) > min_area_ratio * total_image_area:
                carnet_contour = approx
                break

        if carnet_contour is None:
            st.warning("No se pudo detectar un contorno claro. Usando la imagen completa.")
            return imagen_pil

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
        return imagen_pil

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
    """Envía una o dos imágenes a la API de Gemini Vision y pide la extracción de datos."""
    if not GEMINI_CONFIGURADO:
        return {"Error": "API de Gemini no configurada."}

    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    prompt_parts = [
        "Eres un experto en analizar cédulas de ciudadanía de Colombia.",
        "Analiza la(s) siguiente(s) imagen(es).",
        "Primero, determina si la imagen principal es una Cédula de Ciudadanía de Colombia. Luego, extrae la información y devuélvela en un formato JSON estricto con los siguientes campos:",
        "- es_cedula_colombiana (booleano: true si es una cédula de Colombia, false si no).",
        "- NUIP", "- Apellidos", "- Nombres", "- Fecha de nacimiento", "- Lugar de nacimiento",
        "- Estatura", "- Sexo", "- GS RH", "- Fecha y lugar de expedición",
        "Si no es una cédula de Colombia, solo devuelve {\"es_cedula_colombiana\": false}.",
        "Si es una cédula válida pero no encuentras un campo, usa el valor 'No encontrado'.",
    ]
    
    for img in imagenes_pil:
        prompt_parts.append(img)
        
    try:
        response = model.generate_content(prompt_parts)
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        else:
            st.warning("La IA respondió con un mensaje en lugar de datos:")
            st.info(response.text)
            return {"Error": "Respuesta no válida de la IA.", "es_cedula_colombiana": False}
    except Exception as e:
        st.error(f"Error al contactar o procesar la respuesta de la API de Gemini: {e}")
        return {"Error": "Fallo en la comunicación con la IA.", "es_cedula_colombiana": False}

# --- INTERFAZ DE STREAMLIT ---
st.set_page_config(page_title="Lector de Cédulas IA", layout="wide")
st.title("🚀 Lector de Cédulas con IA (Gemini)")

# --- GESTIÓN DE ESTADO ---
if 'stage' not in st.session_state:
    st.session_state.stage = 'inicio'
if 'datos_capturados' not in st.session_state:
    st.session_state.datos_capturados = []
if 'anverso_buffer' not in st.session_state:
    st.session_state.anverso_buffer = None
if 'reverso_buffer' not in st.session_state:
    st.session_state.reverso_buffer = None

def limpiar_y_empezar_de_nuevo():
    st.session_state.stage = 'inicio'
    st.session_state.anverso_buffer = None
    st.session_state.reverso_buffer = None

st.info("Puedes tomar una foto en vivo o subir una imagen desde tu galería para mayor precisión.")

tab1, tab2 = st.tabs(["📸 Tomar Foto (Secuencial)", "⬆️ Subir Foto (Múltiple)"])

# --- Lógica de la Pestaña 1: Tomar Foto ---
with tab1:
    st.write("Sigue los pasos para capturar las imágenes. Solo se activa una cámara a la vez.")
    
    # Etapa 1: Capturar Anverso
    if st.session_state.stage == 'inicio':
        foto_anverso_cam = st.camera_input("Paso 1: Toma una foto del **Anverso**", key="cam_anverso")
        if foto_anverso_cam:
            st.session_state.anverso_buffer = foto_anverso_cam
            st.session_state.stage = 'anverso_listo'
            st.rerun()

    # Etapa 2: Capturar Reverso
    elif st.session_state.stage == 'anverso_listo':
        st.success("✔️ Paso 1: Anverso capturado.")
        st.image(st.session_state.anverso_buffer)
        foto_reverso_cam = st.camera_input("Paso 2: Toma una foto del **Reverso** (opcional)", key="cam_reverso")
        if foto_reverso_cam:
            st.session_state.reverso_buffer = foto_reverso_cam
            st.session_state.stage = 'listo_para_procesar'
            st.rerun()
        
        # Botón para procesar solo con el anverso
        if st.button("Procesar solo con Anverso"):
            st.session_state.stage = 'listo_para_procesar'
            st.rerun()

# --- Lógica de la Pestaña 2: Subir Foto ---
with tab2:
    st.write("Sube imágenes desde tu dispositivo para obtener la máxima calidad.")
    up_col1, up_col2 = st.columns(2)
    anverso_up = up_col1.file_uploader("1. Anverso (lado principal)", type=['jpg', 'jpeg', 'png'], key="up_anverso")
    reverso_up = up_col2.file_uploader("2. Reverso (opcional)", type=['jpg', 'jpeg', 'png'], key="up_reverso")
    
    if anverso_up:
        # Si se sube un archivo, sobreescribe el estado de la cámara y procesa
        st.session_state.anverso_buffer = anverso_up
        st.session_state.reverso_buffer = reverso_up
        st.session_state.stage = 'listo_para_procesar'


# --- Lógica de Procesamiento Centralizada ---
if st.session_state.stage == 'listo_para_procesar':
    st.info("Procesando imagen(es)...")
    imagenes_a_procesar = []
    
    if st.session_state.anverso_buffer:
        img_anverso_pil = Image.open(st.session_state.anverso_buffer)
        img_anverso_corregida = corregir_perspectiva(img_anverso_pil)
        imagenes_a_procesar.append(img_anverso_corregida)

    if st.session_state.reverso_buffer:
        img_reverso_pil = Image.open(st.session_state.reverso_buffer)
        img_reverso_corregida = corregir_perspectiva(img_reverso_pil)
        imagenes_a_procesar.append(img_reverso_corregida)
    
    if GEMINI_CONFIGURADO and imagenes_a_procesar:
        with st.spinner('La IA está analizando los documentos...'):
            datos_estructurados = extraer_datos_con_gemini(imagenes_a_procesar)
        
        st.session_state.ultimo_dato = datos_estructurados
        st.session_state.stage = 'resultados_listos' # Transición a la etapa de resultados
        st.rerun()

# --- Mostrar Resultados y Acciones ---
if st.session_state.stage == 'resultados_listos':
    datos = st.session_state.get('ultimo_dato', {})
    
    if datos.get("es_cedula_colombiana"):
        st.subheader("Resultado del Análisis de IA")
        st.json(datos)
        
        if st.button("Confirmar y Añadir a la Lista"):
            st.session_state.datos_capturados.append(datos)
            st.success("¡Datos añadidos!")
            limpiar_y_empezar_de_nuevo()
            st.rerun()
    else:
        st.error("EL DOCUMENTO ANALIZADO NO PARECE SER UNA CÉDULA DE CIUDADANÍA DE COLOMBIA.")
        if "Error" in datos:
            st.json(datos)
    
    st.button("Limpiar y Empezar de Nuevo", on_click=limpiar_y_empezar_de_nuevo)

# --- Mostrar la tabla de registros ---
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
