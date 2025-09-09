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
    # Intenta obtener la clave de los secretos de Streamlit
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error("La clave GEMINI_API_KEY no se encontró en los secretos de Streamlit.")
        GEMINI_CONFIGURADO = False
    else:
        genai.configure(api_key=api_key)
        GEMINI_CONFIGURADO = True
except Exception:
    # Fallback para desarrollo local si st.secrets no está disponible
    st.warning("No se pudieron cargar los secretos de Streamlit. Asegúrate de que tu clave de API esté configurada si estás en producción.")
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
        min_area_ratio = 0.1 # El contorno debe ser al menos el 10% del área de la imagen
        total_image_area = img.shape[0] * img.shape[1]

        for contorno in contornos:
            perimetro = cv2.arcLength(contorno, True)
            approx = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)
            # MEJORA: Comprobar que el contorno sea un cuadrilátero Y que tenga un tamaño razonable
            if len(approx) == 4 and cv2.contourArea(approx) > min_area_ratio * total_image_area:
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

    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    prompt_parts = [
        "Eres un experto en analizar cédulas de ciudadanía de Colombia, tanto el modelo antiguo (amarilla) como el nuevo (digital).",
        "Analiza la(s) siguiente(s) imagen(es).",
        "Primero, determina si la imagen principal es una Cédula de Ciudadanía de Colombia. Luego, extrae la información y devuélvela en un formato JSON estricto con los siguientes campos:",
        "- es_cedula_colombiana (un booleano: true si estás seguro que es una cédula de Colombia, false si es otro documento o no estás seguro).",
        "- NUIP o NUMERO (siempre usa la etiqueta 'NUIP')",
        "- Apellidos",
        "- Nombres",
        "- Fecha de nacimiento",
        "- Lugar de nacimiento",
        "- Estatura",
        "- Sexo",
        "- GS RH (Grupo Sanguíneo y RH)",
        "- Fecha y lugar de expedición",
        "Si no es una cédula de Colombia, solo devuelve {\"es_cedula_colombiana\": false}.",
        "Si es una cédula válida pero no encuentras un campo, usa el valor 'No encontrado' para ese campo.",
        "Ejemplo de respuesta para una cédula válida: {\"es_cedula_colombiana\": true, \"NUIP\": \"12.345.678\", \"Apellidos\": \"PEREZ GOMEZ\", ...}",
    ]
    
    for img in imagenes_pil:
        prompt_parts.append(img)
        
    try:
        response = model.generate_content(prompt_parts)
        
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        
        if match:
            json_text = match.group(0)
            return json.loads(json_text)
        else:
            st.warning("La IA no pudo procesar la imagen y respondió con un mensaje:")
            st.info(response.text)
            return {"Error": "La IA no pudo extraer datos. Intenta con una foto más nítida.", "es_cedula_colombiana": False}
            
    except json.JSONDecodeError:
        st.error("La respuesta de la IA no tuvo un formato JSON válido.")
        st.text("Respuesta cruda de la API:")
        st.text(response.text)
        return {"Error": "Respuesta inválida de la IA.", "es_cedula_colombiana": False}

    except Exception as e:
        st.error(f"Error al contactar la API de Gemini: {e}")
        return {"Error": "No se pudo procesar la respuesta de la IA.", "es_cedula_colombiana": False}

# --- INTERFAZ DE STREAMLIT ---
st.set_page_config(page_title="Lector de Cédulas IA", layout="wide")
st.title("🚀 Lector de Cédulas con IA (Gemini)")
st.info("Toma fotos claras del anverso y, si es necesario, del reverso de la cédula.")

# Inicializar estado de sesión
if 'datos_capturados' not in st.session_state:
    st.session_state.datos_capturados = []
if 'run_id' not in st.session_state:
    st.session_state.run_id = 0

def reset_scan():
    """Incrementa el run_id para forzar el reseteo de los widgets de cámara."""
    st.session_state.run_id += 1

# Usar columnas para los widgets de la cámara
col1, col2 = st.columns(2)
with col1:
    foto_anverso_buffer = st.camera_input(
        "1. Toma una foto del **Anverso** (lado principal)", 
        key=f"anverso_{st.session_state.run_id}"
    )

with col2:
    foto_reverso_buffer = st.camera_input(
        "2. Toma una foto del **Reverso** (opcional, para cédula antigua)",
        key=f"reverso_{st.session_state.run_id}"
    )

if foto_anverso_buffer:
    st.info("Procesando imagen(es)... esto puede tardar unos segundos.")
    
    imagenes_a_procesar = []
    
    img_anverso_pil = Image.open(foto_anverso_buffer)
    img_anverso_corregida = corregir_perspectiva(img_anverso_pil)
    imagenes_a_procesar.append(img_anverso_corregida)
    st.subheader("Anverso Corregido")
    st.image(img_anverso_corregida, use_container_width=True)

    if foto_reverso_buffer:
        img_reverso_pil = Image.open(foto_reverso_buffer)
        img_reverso_corregida = corregir_perspectiva(img_reverso_pil)
        imagenes_a_procesar.append(img_reverso_corregida)
        st.subheader("Reverso Corregido")
        st.image(img_reverso_corregida, use_container_width=True)
    
    if GEMINI_CONFIGURADO:
        with st.spinner('La IA está analizando los documentos...'):
            datos_estructurados = extraer_datos_con_gemini(imagenes_a_procesar)
        
        # --- LÓGICA DE VALIDACIÓN ---
        if datos_estructurados.get("es_cedula_colombiana"):
            st.subheader("Resultado del Análisis de IA")
            st.json(datos_estructurados)
            st.session_state.ultimo_dato = datos_estructurados

            action_col1, action_col2 = st.columns(2)
            with action_col1:
                if "Error" not in datos_estructurados and st.button("Confirmar y Añadir a la Lista"):
                    st.session_state.datos_capturados.append(st.session_state.ultimo_dato)
                    st.success("¡Datos añadidos!")
            
            with action_col2:
                st.button("📸 Leer Nuevo Documento", on_click=reset_scan)
        else:
            st.error("EL DOCUMENTO ANALIZADO NO PARECE SER UNA CÉDULA DE CIUDADANÍA DE COLOMBIA.")
            st.json(datos_estructurados) # Muestra el resultado para depuración
            st.button("📸 Leer Nuevo Documento", on_click=reset_scan)
            
    else:
        st.error("La aplicación no puede funcionar porque la API de Gemini no está configurada.")

# Mostrar la tabla de registros siempre, fuera del bloque if
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
