import streamlit as st
import numpy as np
import joblib
import os

def predecir_matricula(edad, genero, nivel_educativo, ingresos_mensuales,
                       ocupacion, interes_tema, uso_tecnologia, horas_disponibles,
                       promociones_recibidas):
    try:
        # Codificar variables categóricas con LabelEncoder
        genero_encoded = encoders['genero'].transform([genero])[0]
        nivel_encoded = encoders['nivel_educativo'].transform([nivel_educativo])[0]
        ocupacion_encoded = encoders['ocupacion'].transform([ocupacion])[0]

        # Construir vector en el orden correcto
        X = [[
            edad,
            genero_encoded,
            nivel_encoded,
            ingresos_mensuales,
            ocupacion_encoded,
            interes_tema,
            uso_tecnologia,
            horas_disponibles,
            promociones_recibidas
        ]]

        # Predicción
        pred = modelo.predict(X)[0]
        return f"✅ Predicción: {pred}"

    except Exception as e:
        return f"❌ Error en la predicción: {e}"


# Configuración de la página
st.set_page_config(
    page_title="Predicción de Matrícula",
    page_icon="🎓",
    layout="wide"
)

# Función para cargar modelos con caché
@st.cache_resource
def cargar_modelos():
    """Carga el modelo y encoders desde archivos PKL"""
    try:
        # Verificar si los archivos existen (con nombres alternativos)
        modelo_files = ['modelo.pkl', 'modelo_entrenado.pkl']
        encoder_files = ['encoders.pkl', 'encoders_entrenados.pkl']
        
        modelo_file = None
        encoder_file = None
        
        # Buscar archivo del modelo
        for file in modelo_files:
            if os.path.exists(file):
                modelo_file = file
                break
        
        # Buscar archivo de encoders
        for file in encoder_files:
            if os.path.exists(file):
                encoder_file = file
                break
        
        if modelo_file is None:
            raise FileNotFoundError("No se encontró ningún archivo de modelo (modelo.pkl o modelo_entrenado.pkl)")
        
        if encoder_file is None:
            raise FileNotFoundError("No se encontró ningún archivo de encoders (encoders.pkl o encoders_entrenados.pkl)")
        
        # Cargar modelo entrenado
        modelo = joblib.load(modelo_file)
        st.info(f"✅ Modelo cargado desde: {modelo_file}")
        
        # Cargar encoders
        encoders = joblib.load(encoder_file)
        st.info(f"✅ Encoders cargados desde: {encoder_file}")
        
        return modelo, encoders
        
    except FileNotFoundError as e:
        return None, None, str(e)
    except Exception as e:
        return None, None, f"Error al cargar los modelos: {e}"

# Cargar modelos al inicio
resultado = cargar_modelos()

if len(resultado) == 3:  # Hay error
    modelo, encoders, error_msg = resultado
    st.error(f"❌ {error_msg}")
    st.error("Asegúrate de que los archivos 'modelo.pkl' y 'encoders.pkl' estén en el directorio actual")
    
    # Mostrar archivos disponibles en el directorio
    archivos_disponibles = [f for f in os.listdir('.') if f.endswith('.pkl')]
    if archivos_disponibles:
        st.info(f"Archivos .pkl encontrados: {', '.join(archivos_disponibles)}")
    else:
        st.warning("No se encontraron archivos .pkl en el directorio actual")
    
    st.stop()
else:
    modelo, encoders = resultado

# Título y descripción
st.title("🎓 Predicción de Matrícula")
st.markdown("Ingresa los datos del estudiante para predecir si se matriculará (0 = No, 1 = Sí).")

# Verificar que los modelos se cargaron correctamente
if modelo is not None and encoders is not None:
    st.success("✅ Modelos cargados correctamente")
    
    # Mostrar información de los encoders disponibles
    with st.expander("📊 Información de los encoders cargados"):
        st.write("**Encoders disponibles:**")
        for key, encoder in encoders.items():
            st.write(f"- **{key}**: {list(encoder.classes_)}")

    # Obtener listas de clases desde los LabelEncoders
    try:
        generos = encoders['genero'].classes_.tolist()
        niveles_educativos = encoders['nivel_educativo'].classes_.tolist()
        ocupaciones = encoders['ocupacion'].classes_.tolist()
        
        # Crear formulario con columnas para mejor organización
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Datos Personales")
                edad = st.number_input(
                    "Edad",
                    min_value=0,
                    max_value=100,
                    value=25,
                    step=1
                )
                
                genero = st.selectbox(
                    "Género",
                    options=generos,
                    index=0
                )
                
                nivel_educativo = st.selectbox(
                    "Nivel educativo",
                    options=niveles_educativos,
                    index=0
                )
                
                ingresos_mensuales = st.number_input(
                    "Ingresos mensuales",
                    min_value=0.0,
                    value=1000.0,
                    step=100.0,
                    format="%.2f"
                )
                
                ocupacion = st.selectbox(
                    "Ocupación",
                    options=ocupaciones,
                    index=0
                )
            
            with col2:
                st.subheader("Preferencias y Disponibilidad")
                interes_tema = st.slider(
                    "Interés en el tema",
                    min_value=0,
                    max_value=5,
                    value=3,
                    step=1
                )
                
                uso_tecnologia = st.slider(
                    "Uso de tecnología",
                    min_value=0,
                    max_value=5,
                    value=3,
                    step=1
                )
                
                horas_disponibles = st.slider(
                    "Horas disponibles",
                    min_value=0,
                    max_value=24,
                    value=8,
                    step=1
                )
                
                promociones_recibidas = st.slider(
                    "Promociones recibidas",
                    min_value=0,
                    max_value=20,
                    value=5,
                    step=1
                )
            
            # Botón de predicción
            submit_button = st.form_submit_button(
                "🔮 Realizar Predicción",
                use_container_width=True
            )

        # Procesar predicción cuando se envía el formulario
        if submit_button:
            with st.spinner('Realizando predicción...'):
                resultado = predecir_matricula(
                    edad, genero, nivel_educativo, ingresos_mensuales,
                    ocupacion, interes_tema, uso_tecnologia, horas_disponibles,
                    promociones_recibidas
                )
            
            # Mostrar resultado
            if "✅" in resultado:
                st.success(resultado)
                # Interpretar el resultado
                pred_value = resultado.split(": ")[1]
                if pred_value == "1":
                    st.balloons()
                    st.info("🎉 El estudiante tiene alta probabilidad de matricularse")
                else:
                    st.info("📊 El estudiante tiene baja probabilidad de matricularse")
            else:
                st.error(resultado)
                
    except KeyError as e:
        st.error(f"❌ Error: No se encontró el encoder para {e}")
        st.error("Verifica que el archivo 'encoders.pkl' contenga los encoders necesarios: 'genero', 'nivel_educativo', 'ocupacion'")
        st.stop()
        
else:
    st.error("❌ No se pudieron cargar los modelos")
    st.stop()

# Información adicional
with st.expander("ℹ️ Información sobre los parámetros"):
    st.markdown("""
    **Descripción de los parámetros:**
    
    - **Edad**: Edad del estudiante en años
    - **Género**: Género del estudiante
    - **Nivel educativo**: Nivel de educación completado
    - **Ingresos mensuales**: Ingresos económicos mensuales
    - **Ocupación**: Ocupación actual del estudiante
    - **Interés en el tema**: Nivel de interés (0-5, donde 5 es muy alto)
    - **Uso de tecnología**: Familiaridad con tecnología (0-5)
    - **Horas disponibles**: Horas disponibles para estudiar por día
    - **Promociones recibidas**: Número de promociones/ofertas recibidas
    """)

with st.expander("🔧 Información técnica"):
    st.markdown("""
    **Archivos requeridos:**
    - `modelo.pkl`: Modelo de machine learning entrenado
    - `encoders.pkl`: Diccionario con los LabelEncoders para variables categóricas
    
    **Estructura esperada del encoders.pkl:**
    ```python
    {
        'genero': LabelEncoder(),
        'nivel_educativo': LabelEncoder(), 
        'ocupacion': LabelEncoder()
    }
    ```
    """)

# Pie de página
st.markdown("---")
st.markdown("*Desarrollado con Streamlit y joblib* 🚀")