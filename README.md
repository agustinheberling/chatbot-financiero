## Chatbot Financiero con RAG usando FastAPI, Streamlit y Llama 3
Este proyecto implementa un sistema de recuperación aumentada por generación (RAG) que permite responder consultas sobre la circular 2473 de abril de 2025 emitida por el Banco Central del Uruguay. Utiliza una API construida con FastAPI, una interfaz web con Streamlit y una arquitectura basada en embeddings, búsqueda vectorial y modelos LLM.

## Objetivos
- Desarrollar una API que disponibilice un modelo de pregunta y respuesta.
- Implementar una interfaz de usuario utilizando Streamlit para interactuar con el modelo.
- Desplegar la solución localmente utilizando Docker para asegurar la portabilidad y escalabilidad del servicio.

## Requisitos generales del entorno
- Python 3.8
- pip
- Docker (opcional para ejecución en contenedor)
- Git (para clonar el repositorio)

## Dependencias (en requirements.txt)
fastapi, pydantic, uvicorn, streamlit, pdfminer.six, transformers, torch, sentence-transformers, faiss-cpu, langchain, llama-index, huggingface-hub, langchain-community, accelerate, pymupdf

## Componentes del Proyecto
- **Dockerfile**: Define el entorno de ejecución contenerizado para la API y la interfaz de usuario.
- **app.py**: Script de Flask para implementar la API.
- **front_streamlit.py**: Script de Streamlit para la interfaz de usuario.
- **requirements.txt**: Lista de dependencias necesarias para el proyecto sin dockerizar.
- **requirements_docker.txt**: Lista de dependencias necesarias para el proyecto dockerizado.
- **build_api.sh, run_api.sh, setup_api.sh, run_docker.sh**: Scripts para facilitar la construcción, ejecución y configuración de la API y el entorno de Docker.

## Instrucciones de instalación y ejecución local (sin Docker)
- Clonar el repositorio
- Crear un entorno virtual e Instalar dependencias ejecutando setup_api.sh
- Ejecutar el sistema con run_api.sh
- Acceder a la interfaz web en http://localhost:8501

## Ejecución con Docker
- Asegúrate de tener Docker Desktop en ejecución
- Crear la imagen ejecutando build_api.sh
- Ejecutar el contenedor con run_docker.sh

## Descripción del sistema RAG implementado
El sistema sigue un enfoque modular para responder preguntas sobre documentos normativos:
1. Parseo del pdf: se utiliza pdfminer.six para extraer el texto estructurado del documentos pdf de la cirular, preservando encabezados y secciones.
2. Generación de embeddings: se utiliza un modelo de embeddings sentence-transformers/e5-base utilizando como tecnica e5 embedding con prompt "query: " y "passage: " para consultas y documentos, respectivamente.
3. Base vectorial: se utiliza la libreria FAISS (versión CPU). Se hace indexación densa de los textos parseados y se usa búsqueda por similitud coseno para recuperación eficiente.
4. Prompts y recuperación: se recuperan los k documentos más relevantes mediante FAISS y se construye un prompt RAG unificando la pregunta del usuario y los contextos recuperados.
5. LLM para respuesta generada: se utilza el modelo Llama 3 vía Hugging Face (meta-llama/Llama-3.2-3B-Instruct) con LangChain. Se realiza interacción a través de LangChain y LlamaIndex para facilitar el encadenamiento de documentos y preguntas.

## Justificación de decisiones técnicas
- FastAPI: framework moderno y rápido para construir APIs con tipado estático.
- Streamlit: permite construir interfaces visuales ligeras con Python.
- PDFMiner: ofrece control detallado sobre el parseo de textos jurídicos complejos.
- e5-base: balance óptimo entre precisión semántica y velocidad de inferencia.
- FAISS: eficiente en recuperación vectorial, ideal para volúmenes medianos de texto.
- Llama 3: potente modelo de lenguaje de código abierto, adecuado para tareas complejas de generación.
- LangChain y LlamaIndex: facilitan la implementación de RAG y el manejo de pipelines de recuperación.

## Créditos y referencias
- Transformers & Datasets: Hugging Face
- e5-base: intfloat/e5-base
- Llama 3: Meta AI, acceso a través de Hugging Face
- LangChain: https://www.langchain.com
- LlamaIndex: https://www.llamaindex.ai
- PDFMiner.six: https://github.com/pdfminer/pdfminer.six
- FAISS: Facebook AI Similarity Search
