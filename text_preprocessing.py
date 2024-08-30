import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import nltk


# Configurar stemmer y stopwords en español
stemmer = SnowballStemmer('spanish')
stop_words = set(stopwords.words('spanish'))

# Agregar palabras específicas a las stopwords si es necesario
stop_words.update(['del', 'al', 'lo', 'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas'])

def preprocesar_resumen(resumen):
    # Convertir a minúsculas
    resumen = resumen.lower()

    # Eliminar caracteres especiales excepto espacios
    resumen = re.sub(r'[^\w\s]', '', resumen)

    # Eliminar palabras y abreviaturas innecesarias
    palabras_a_eliminar = ['10', '11', '12', '13', 'tpo', 'clave', 'despacho', 'despecho']
    patron = r'\b(?:{})\s*'.format('|'.join(map(re.escape, palabras_a_eliminar)))
    resumen = re.sub(patron, '', resumen)

    # Eliminar espacios extra
    resumen = ' '.join(resumen.split())

    # Tokenizar
    tokens = word_tokenize(resumen)

    # Remover stopwords y aplicar stemming
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words and len(word) > 2]

    # Reconstruir el texto procesado
    resumen = ' '.join(tokens)

    return resumen

def preprocesar_direccion(direccion):
    # Convertir a minúsculas
    direccion = direccion.lower()

    # Reemplazar '/' y '&' por ' y '
    direccion = direccion.replace('/', ' y ').replace('&', ' y ').replace('  ', ' ').replace('c/', ' y ')

    # Eliminar caracteres especiales excepto 'y'
    direccion = re.sub(r'[^\w\s]', '', direccion)

    # Eliminar palabras y abreviaturas innecesarias
    palabras_a_eliminar = ['av', 'av:', 'avenida', 'pte', 'pte:', 'puente', 'calle',  'c:']
    patron = r'\b(?:{})\s*'.format('|'.join(map(re.escape, palabras_a_eliminar)))
    direccion = re.sub(patron, '', direccion)

    # Eliminar espacios extra
    direccion = ' '.join(direccion.split())

    # Tokenizar
    tokens = word_tokenize(direccion)

    # Remover stopwords y aplicar stemming
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words and len(word) > 2]

    # Reconstruir el texto procesado
    direccion = ' '.join(tokens)

    return direccion