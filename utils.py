import pandas as pd
import numpy as np
from datetime import time
import pickle
from scipy.sparse import hstack
import joblib
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import re
from nltk.corpus import stopwords

def time_to_decimal(t):
    """Convierte un objeto time a un valor decimal."""
    return t.hour + t.minute / 60 + t.second / 3600

# Cargar el scaler
with open('vectorizers/scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

# Cargar el vectorizador TF-IDF
tfidf_vectorizer = joblib.load('vectorizers/tfidf_vectorizer.joblib')

# Configurar stemmer y stopwords
stemmer = SnowballStemmer('spanish')
stop_words = set(stopwords.words('spanish'))

def preprocess_text(text):
    """Preprocesa el texto para el análisis TF-IDF."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

def preprocess_data(df):
    """Preprocesa el DataFrame para el modelo de predicción."""
    df_processed = df.copy()
    
    df_processed['HORA_INICIO_NUMERIC'] = df_processed['HORA INICIO'].apply(time_to_decimal)
    
    df_processed['DIA_SEMANA'] = df_processed['FECHA'].dt.dayofweek
    df_processed['MES'] = df_processed['FECHA'].dt.month
    df_processed['DIA_MES'] = df_processed['FECHA'].dt.day
    
    numeric_columns = ['HORA_INICIO_NUMERIC', 'DIA_SEMANA', 'MES', 'DIA_MES']
    df_processed[numeric_columns] = loaded_scaler.transform(df_processed[numeric_columns])
    
    return df_processed

def prepare_input_data(fecha, hora_inicio, resumen, direccion):
    """Prepara los datos de entrada para el modelo."""
    
    input_data = pd.DataFrame({
        'FECHA': [pd.to_datetime(fecha)],
        'HORA INICIO': [hora_inicio],
        'RESUMEN': [resumen],
        'DIRECCION': [direccion]
    })
    
    processed_input = preprocess_data(input_data)
    
    # Preprocesar texto
    combined_text = preprocess_text(resumen + ' ' + direccion)
    
    # Aplicar TF-IDF
    tfidf_features = tfidf_vectorizer.transform([combined_text])
    
    numeric_columns = ['HORA_INICIO_NUMERIC', 'DIA_SEMANA', 'MES', 'DIA_MES']
    numeric_features = processed_input[numeric_columns].values
    
    combined_features = hstack([numeric_features, tfidf_features])
    
    # Obtener los nombres de las características en el orden correcto
    feature_names = numeric_columns + tfidf_vectorizer.get_feature_names_out().tolist()
    
    # Crear un DataFrame con las características en el orden correcto
    final_features = pd.DataFrame(combined_features.toarray(), columns=feature_names)
    
    # Asegurarse de que todas las características del modelo estén presentes
    model_features = joblib.load('models/model_features.joblib')  # Cargar las características del modelo
    for feature in model_features:
        if feature not in final_features.columns:
            final_features[feature] = 0
    
    # Reordenar las columnas para que coincidan con las del modelo
    final_features = final_features[model_features]
    
    return input_data, final_features