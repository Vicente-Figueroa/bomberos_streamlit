import os
import nltk
import streamlit as st
import pandas as pd
from utils import prepare_input_data, preprocess_text
import joblib
from datetime import datetime, date, time
import numpy as np
import google.generativeai as genai

genai.configure(api_key='AIzaSyAtQwglcD0LvJdhKcbb2KCNjHhiSAepvqQ')
model_ai = genai.GenerativeModel('gemini-1.5-flash')
# Verificar si existe el directorio nltk_data en la raíz del proyecto
data_dir = os.path.join(os.getcwd(), 'venv/nltk_data')

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
model = joblib.load('models/gradient_boosting_model.joblib')
# Obtener la fecha y hora actuales
fecha_actual = date.today()
hora_actual = datetime.now().time()

def generar_mensaje(fecha, hora_inicio, resumen, direccion, minutos):
    template = ("""Quiero que desempeñes el rol de servicio al cliente de bomberos de mi región"""
                """Te llegó a las {fecha} del día {hora_inicio}, el caso de {resumen} en la dirección {direccion} """
                """y según el sistema, el tiempo estimado en minutos es {minutos}. La persona que te habla es una persona real por lo que debes """
                """generar calma y explicarle lo que debe hacer en una situación como esa mientras espera a los bomberos. """
                """Es algo importante, por lo que deben ser instrucciones claras y precisas."""
                """No actues, eres de chile, de la serena, el numero es 132"""
                """Recuerda, no actues"""
                )
    
    # Formatear el mensaje con los valores proporcionados
    mensaje = template.format(fecha=fecha, hora_inicio=hora_inicio, resumen=resumen, direccion=direccion, minutos=minutos)
    
    return mensaje

# Formulario para ingresar datos
with st.form("prediccion_form"):
    # Establecer la fecha y hora actuales como valores predeterminados
    fecha = st.date_input("Fecha del incidente", value=fecha_actual)
    hora_inicio = st.time_input("Hora de inicio", value=hora_actual)
    
    opciones_resumen = [
        "Incendio en una casa",
        "Colision de 2 vehiculos",
        "Mascota atrapada",
        "Olor a gas",
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

    prediccion = model.predict(processed_input)
    original_value = np.expm1(prediccion)
    print('original_value',float(original_value))
    print('prediccion',prediccion)
    print(fecha, hora_inicio, resumen, direccion)
    # Cargar el scaler
    loaded_scaler_inputs = joblib.load('vectorizers/scaler_inputs.pkl')

    # Valor escalado que deseas deshacer
    original_value = float(original_value)

    # Supongamos que el scaler fue entrenado con 4 características (n_features=4)
    # Debes pasar el valor en un array de la misma forma que los datos originales, por ejemplo:
    input_array = np.array([[original_value, 0, 0, 0]])  # Rellena los otros valores con ceros u otros valores

    # Desescalar el valor usando el scaler
    original_value_after_scaling = loaded_scaler_inputs.inverse_transform(input_array)

    # Imprimir el valor original para la primera característica
    print(original_value_after_scaling[0, 0])  # Esto imprime el valor desescalado para la primera característica   
    # Asumiendo que original_value_after_scaling[0, 0] es el valor que quieres mostrar
    minutos = original_value_after_scaling[0, 0]

    # Formatear a dos decimales
    st.write(f"Tiempo de respuesta predicho: {minutos:.2f} minutos")
    mensaje= generar_mensaje(fecha, hora_inicio, resumen, direccion, minutos)
    response = model_ai.generate_content(mensaje)
    st.write(response.text)


st.write("Esta aplicación está en desarrollo. Las predicciones son estimaciones basadas en datos históricos.")