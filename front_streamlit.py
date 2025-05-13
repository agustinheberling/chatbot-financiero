# Frontend (FE)
import requests
import streamlit as st

# Definición de la URL de la API
API_URL = "http://localhost:8000/ask/"

# Streamlit interface, # Configuración de la página
st.title('Chatbot sobre Normas de Reglacion y Control del Sistema Financiero Uruguayo')
st.markdown('Escribe una pregunta y obten una respuesta precisa basada en la Circular N° 2473 del BCU: \n')
review = st.text_area("Escribe tu pregunta aquí:") # Text Area para ingreso de usuario
if st.button("Preguntar"):
    response = requests.post(API_URL, json={"text": review})
    if response.status_code == 200:
        st.write("Respuesta del sistema:")
        st.write(response.json()["response"])
    else:
        st.error("Error en la API")