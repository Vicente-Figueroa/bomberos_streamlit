import os
import nltk
import streamlit as st
import pandas as pd
from datetime import time
from utils import prepare_input_data, preprocess_text
import joblib

# Verificar si existe el directorio nltk_data en la raíz del proyecto
data_dir = os.path.join(os.getcwd(), 'nltk_data')

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Establecer NLTK_DATA para que NLTK busque en la carpeta de datos en la raíz
os.environ['NLTK_DATA'] = data_dir

# Descargar los recursos necesarios
nltk.download('punkt', download_dir=data_dir)
nltk.download('stopwords', download_dir=data_dir)

st.title('Predicción de Tiempo de Respuesta de Bomberos')

st.write('Por favor, ingresa los siguientes datos para predecir el tiempo de respuesta:')

# Cargar el modelo
model = joblib.load('gradient_boosting_model.joblib')

# Formulario para ingresar datos
with st.form("prediccion_form"):
    fecha = st.date_input("Fecha del incidente")
    hora_inicio = st.time_input("Hora de inicio", value=time(12, 0))
    
    opciones_resumen = [
        "Incendio estructural",
        "Incendio vehicular",
        "Rescate",
        "Emergencia médica",
        "Derrame de materiales peligrosos",
        "Otro"
    ]
    resumen = st.selectbox("Tipo de incidente", opciones_resumen)
    
    if resumen == "Otro":
        resumen = st.text_input("Especifica el tipo de incidente")
    
    direccion = st.text_input("Dirección del incidente")
    
    submitted = st.form_submit_button("Predecir tiempo de respuesta")

if submitted:
    raw_input, processed_input = prepare_input_data(fecha, hora_inicio, resumen, direccion)
    
    st.write("Datos sin procesar:")
    st.write(raw_input)
    
    st.write("Texto preprocesado:")
    st.write(preprocess_text(resumen + ' ' + direccion))
    
    st.write("Datos procesados listos para la predicción:")
    st.write(processed_input.style.format("{:.6f}"))  # Mostrar 6 decimales
    
    prediccion = model.predict(processed_input)
    
    minutos = int(prediccion[0])
    segundos = int((prediccion[0] - minutos) * 60)
    
    st.write(f"Tiempo de respuesta predicho: {prediccion} minutos ")

st.write("Esta aplicación está en desarrollo. Las predicciones son estimaciones basadas en datos históricos.")