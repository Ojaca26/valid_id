import streamlit as st
import cv2
import pandas as pd
from PIL import Image
import numpy as np
import io
import google.generativeai as genai
import json
import re
import face_recognition

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

def comparar_rostros(img_cedula_pil, img_selfie_pil):
    """Compara los rostros en dos imágenes y devuelve si coinciden."""
    try:
        img_cedula_np = np.array(img_cedula_pil)
        img_selfie_np = np.array(img_selfie_pil)

        # Obtener los encodings (características faciales) de cada imagen
        cedula_encodings = face_recognition.face_encodings(img_cedula_np)
        selfie_encodings = face_recognition.face_encodings(img_selfie_np)

        if not cedula_encodings:
            return "No se encontró un rostro en la foto de la cédula.", False
        if not selfie_encodings:
            return "No se encontró un rostro en la selfie.", False
        
        # Comparar el primer rostro encontrado en cada imagen
        # El valor de tolerance (0.6 por defecto) indica qué tan estricta es la comparación
        coincide = face_recognition.compare_faces([cedula_encodings[0]], selfie_encodings[0], tolerance=0.6)
        
        if coincide[0]:
            return "✅ Verificación Exitosa: Los rostros coinciden.", True
        else:
            return "❌ Verificación Fallida: Los rostros no coinciden.", False

    except Exception as e:
        return f"Ocurrió un error durante la comparación facial: {e}", False


# --- INTERFAZ DE STREAMLIT ---
st.set_page_config(page_title="Verificación de Identidad IA", layout="wide")
st.title("🚀 Verificación de Identidad con IA (Gemini)")

# --- GESTIÓN DE ESTADO ---
if 'stage' not in st.session_state:
    st.session_state.stage = 'inicio'
if 'datos_capturados' not in st.session_state:
    st.session_state.datos_capturados = []
# ... otros estados ...
if 'anverso_buffer' not in st.session_state: st.session_state.anverso_buffer = None
if 'reverso_buffer' not in st.session_state: st.session_state.reverso_buffer = None
if 'selfie_buffer' not in st.session_state: st.session_state.selfie_buffer = None
if 'cedula_corregida' not in st.session_state: st.session_state.cedula_corregida = None

def limpiar_y_empezar_de_nuevo():
    st.session_state.stage = 'inicio'
    st.session_state.anverso_buffer = None
    st.session_state.reverso_buffer = None
    st.session_state.selfie_buffer = None
    st.session_state.cedula_corregida = None


# --- PESTAÑAS DE ENTRADA ---
st.info("Paso 1: Proporciona la imagen de la cédula.")
tab1, tab2 = st.tabs(["📸 Tomar Foto", "⬆️ Subir Foto"])

# Lógica para capturar o subir la cédula (solo si estamos al inicio)
if st.session_state.stage == 'inicio':
    with tab1:
        anverso_cam = st.camera_input("Toma una foto del **Anverso**", key="cam_anverso")
        if anverso_cam:
            st.session_state.anverso_buffer = anverso_cam
            st.session_state.stage = 'procesar_cedula'
            st.rerun()
    with tab2:
        anverso_up = st.file_uploader("Sube una foto del **Anverso**", type=['jpg', 'jpeg', 'png'], key="up_anverso")
        if anverso_up:
            st.session_state.anverso_buffer = anverso_up
            st.session_state.stage = 'procesar_cedula'
            st.rerun()

# --- ETAPA DE PROCESAMIENTO DE CÉDULA ---
if st.session_state.stage == 'procesar_cedula':
    st.info("Procesando Cédula...")
    img_anverso_pil = Image.open(st.session_state.anverso_buffer)
    st.session_state.cedula_corregida = corregir_perspectiva(img_anverso_pil)
    
    with st.spinner('La IA está analizando la cédula...'):
        st.session_state.datos_cedula = extraer_datos_con_gemini([st.session_state.cedula_corregida])
    
    st.session_state.stage = 'mostrar_resultados_cedula'
    st.rerun()

# --- ETAPA DE RESULTADOS DE CÉDULA Y VERIFICACIÓN FACIAL ---
if st.session_state.stage == 'mostrar_resultados_cedula':
    datos = st.session_state.get('datos_cedula', {})
    
    if datos.get("es_cedula_colombiana"):
        st.subheader("Paso 1: Datos Extraídos de la Cédula")
        st.json(datos)
        
        st.info("Paso 2: Verificación Facial - Compara el rostro con la foto de la cédula.")
        selfie_cam = st.camera_input("Toma una selfie para la comparación", key="cam_selfie")
        
        if selfie_cam:
            st.session_state.selfie_buffer = selfie_cam
            st.session_state.stage = 'procesar_selfie'
            st.rerun()
    else:
        st.error("EL DOCUMENTO ANALIZADO NO PARECE SER UNA CÉDULA DE CIUDADANÍA DE COLOMBIA.")
        st.button("Empezar de Nuevo", on_click=limpiar_y_empezar_de_nuevo)

# --- ETAPA DE PROCESAMIENTO DE SELFIE Y COMPARACIÓN ---
if st.session_state.stage == 'procesar_selfie':
    st.subheader("Paso 2: Verificación Facial - Resultados")
    
    col_cedula, col_selfie = st.columns(2)
    with col_cedula:
        st.image(st.session_state.cedula_corregida, caption="Imagen de la Cédula", use_container_width=True)
    with col_selfie:
        st.image(st.session_state.selfie_buffer, caption="Selfie Capturada", use_container_width=True)

    with st.spinner("Comparando rostros..."):
        mensaje, exito = comparar_rostros(st.session_state.cedula_corregida, Image.open(st.session_state.selfie_buffer))
    
    if exito:
        st.success(mensaje)
        st.balloons()
        if st.button("Confirmar y Guardar Registro"):
            registro_final = st.session_state.datos_cedula
            registro_final["verificacion_facial"] = "Exitosa"
            st.session_state.datos_capturados.append(registro_final)
            st.success("¡Registro guardado!")
            limpiar_y_empezar_de_nuevo()
            st.rerun()
    else:
        st.error(mensaje)

    st.button("Intentar de Nuevo", on_click=limpiar_y_empezar_de_nuevo)


# --- Mostrar la tabla de registros ---
if st.session_state.datos_capturados:
    st.subheader("Registros Verificados")
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

