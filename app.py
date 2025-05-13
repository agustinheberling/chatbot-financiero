# Instalación de dependencia FAISS (Bases de datos vectoriales)

import subprocess
import sys
import time

def install_faiss():
    try:
        # Check if a GPU is available
        gpu_available = subprocess.check_output("nvidia-smi", shell=True)
        # Install faiss-gpu if GPU is available
        subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-gpu"])
    except subprocess.CalledProcessError:
        # If no GPU is available, install faiss-cpu
        subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu"])

install_faiss()

# Importación de dependencias para extracción y matcheo de texto

from pdfminer.high_level import extract_text
import re

# Definción de funciones custom para extraer de forma apropiada datos del PDF

import fitz  # PyMuPDF

def limpiar_encabezados_pies(texto):
    # Patrón para detectar y eliminar circulares y antecedentes
    patrones = [
        # Encabezado específico
        r'Última circular:\s*Nº\s*\d+\s*de\s*\d{2}\s*de\s*\w+\s*de\s*\d{4}',

        # Líneas de circulares y resoluciones con fechas y vigencias
        r'Circular\s+\d+\s*-\s*Resolución\s+del\s+\d{2}\.\d{2}\.\d{4}\s*-\s*Vigencia\s+Diario\s+Oficial\s+\d{2}\.\d{2}\.\d{4}\s*-\s*\(\d{4}/\d{4}\)',

        # Línea de "Antecedentes del artículo"
        r'Antecedentes del artículo',
    ]

    for patron in patrones:
        texto = re.sub(patron, '', texto, flags=re.IGNORECASE)

    # Eliminar líneas que quedaron vacías
    texto = re.sub(r'\n\s*\n', '\n\n', texto)

    return texto

def extraer_articulos_con_titulos_capitulos(pdf_path):
    doc = fitz.open(pdf_path)
    texto = "\n".join([page.get_text() for page in doc])
    texto = limpiar_encabezados_pies(texto)

    # Encabezados principales en mayúsculas al inicio de línea
    encabezado_regex = r'^(LIBRO|T[ÍI]TULO|CAP[ÍI]TULO|SECCI[ÓO]N|ART[ÍI]CULO)\s+[^\n]*'
    encabezados = list(re.finditer(encabezado_regex, texto, re.MULTILINE))

    articulos = []
    libro = titulo = capitulo = seccion = None

    for i, match in enumerate(encabezados):
        tipo = match.group(1)
        contenido = match.group(0).strip()
        inicio = match.end()

        fin = encabezados[i + 1].start() if i + 1 < len(encabezados) else len(texto)
        bloque = texto[inicio:fin].strip()

        if tipo == "LIBRO":
            libro = contenido
        elif tipo.startswith("T"):
            titulo = contenido
        elif tipo.startswith("CAP"):
            capitulo = contenido
        elif tipo.startswith("SECCI"):
            seccion = contenido
        elif tipo.startswith("ART"):
            # Capturar el encabezado del artículo como clave
            articulo = contenido
            articulos.append({
                "libro": libro,
                "titulo": titulo,
                "capitulo": capitulo,
                "seccion": seccion,
                "articulo": articulo,
                "contenido": bloque.strip()
            })

    return articulos

# Extracción de datos del PDF mediante funciones customizadas

import requests

# URL del PDF
url = "https://www.bcu.gub.uy/Acerca-de-BCU/Normativa/Documents/Reordenamiento%20de%20la%20Recopilación/Sistema%20Financiero/RNRCSF.pdf"

# Descargar el PDF
response = requests.get(url, verify=False)
nombre_archivo = "RNRCSF.pdf"

with open(nombre_archivo, "wb") as f:
    f.write(response.content)

# Ejecutar la función sobre el archivo descargado
articulos_con_metadatos = extraer_articulos_con_titulos_capitulos(nombre_archivo)

# Generación de índice en FAISS

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

# Crear la lista de documentos con los metadatos de capítulos y títulos
documents = [
    Document(page_content=articulo["contenido"],
             metadata={
                        "titulo": articulo["titulo"],
                        "capitulo": articulo["capitulo"]
                      })
    for articulo in articulos_con_metadatos
]

print("⏳ Iniciando generación de embeddings y creación del índice FAISS...")
start_time = time.time()  # Guardar tiempo de inicio
# Elegir el modelo de embedding:
#embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2") # liviano
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base") # mediano
#embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large") # pesado

# Crear el índice FAISS con los embeddings y documentos
faiss_index = FAISS.from_documents(documents, embedding_model)

end_time = time.time()  # Guardar tiempo de fin
elapsed_time = end_time - start_time
print(f"✅ Embeddings generados y FAISS index creado en {elapsed_time:.2f} segundos.")

# Configurar el retriever de FAISS
retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Conectarse a Huggingface y autenticar
from huggingface_hub import login, whoami
import getpass

def authenticate_huggingface():
    token = getpass.getpass("🔐 Ingresa tu token de Hugging Face (no se mostrará): ")
    login(token=token)

    # Verificamos la autenticación
    info = whoami()
    print(f"✅ Autenticado correctamente como: {info['name']}")

# Llamamos a la función al iniciar
authenticate_huggingface()

#from huggingface_hub import login
#from huggingface_hub.hf_api import HfFolder
#HfFolder.save_token('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')

# Carga de LLM (Llama3.2)
import os
import torch
from transformers import pipeline

#model_id = "tiiuae/falcon-rw-1b" # muy liviano y no da resultados de nuestro dominio de conocimiento
#model_id = "tiiuae/falcon-7b-instruct"
#model_id = "TheBloke/Mistral-7B-Instruct-v0.1-AWQ" # da error al instalar la libreria pip install autoawq
#model_id = "mistralai/Mistral-7B-Instruct-v0.1" # da error al descargar
model_id = "meta-llama/Llama-3.2-3B-Instruct" # muy pesado, nunca termina de responder
pipe = pipeline(
    "text-generation",
    model=model_id,
    device=0 if torch.cuda.is_available() else -1,
    torch_dtype="auto",
    #trust_remote_code=True, # para modelos optimizados AWQ, GPTQ, etc.
    temperature=0.1,  # Controlar la aleatoriedad en la generación
    do_sample=True,  # Permitir muestreo para la generación
    repetition_penalty=1.1,  # Penalizar repeticiones para más diversidad
    return_full_text=False,  # Retornar solo el texto nuevo generado
    max_new_tokens=500,  # Limitar a 500 tokens la generación
)

# Importar clases necesarias de LangChain y transformers
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Crear una instancia de HuggingFacePipeline con el pipeline configurado
llm = HuggingFacePipeline(pipeline=pipe)

# Definir una plantilla de prompt para la generación de texto
prompt_template = """
<|start_header_id|>user<|end_header_id|>
Eres un asistente respondiendo cuestiones referidas a las normas de regulacion y control del sistema financiero.
Se te proveen artículos extraídos de la circular numero 2473 de abril de 2025 que recopila dichas normas para responder una pregunta.
Debes proveer una respuesta conversacional y en español.
La respuesta debe especificar los números de artículos en que se basa.
Si no sabes la respuesta porque no se encuentra en los artículos del contexto dado responde con "No lo sé"
No inventes la respuesta. No generes información que no se encuentre en el contexto dado.
Siempre terminar el mensaje recomendando consultar con un experto en el tema.
Question: {question}
Context: {context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

# Crear una instancia de PromptTemplate con variables de entrada y la plantilla definida
prompt = PromptTemplate(
    input_variables=["context", "question"],  # Variables de entrada para el prompt
    template=prompt_template,  # Plantilla de prompt
)

# Función para formatear documentos para su uso en el prompt
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)  # Concatenar contenido de documentos

# Configurar la cadena RAG con la estructura de recuperación y generación
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}  # Configuración de contexto y pregunta
    | prompt  # Aplicar el prompt
    | llm  # Usar el modelo de lenguaje
    | StrOutputParser()  # Parsear la salida a string
)

# Creamos la funcion para llamarla luego con fastAPI
def responder_pregunta(pregunta: str) -> str:
    print("📥 Recibiendo pregunta:", pregunta)
    try:
        print("🔍 Invocando RAG Chain...")
        # respuesta_generada = pipe(pregunta)[0]["generated_text"] (probado para debug de rag_chain)
        respuesta = rag_chain.invoke(pregunta)
        print("✅ Respuesta generada.")
        # return respuesta_generada (probado para debug de rag_chain)
        return respuesta
    except Exception as e:
        print("❌ Error en generación:", str(e))
        return f"Error al procesar la pregunta: {str(e)}"

# Inicio de API con FastAPI

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    text: str

@app.post("/ask/")
def ask_question(request: QueryRequest):
    print("✅ Endpoint /ask/ fue llamado con:", request.text)
    respuesta = responder_pregunta(request.text)
    if respuesta.startswith("Error"):
        raise HTTPException(status_code=500, detail=respuesta)
    return {"response": respuesta}