#!/bin/bash

# Crear entorno (environment) para API
# python3 -m venv api_env

# Activar entorno para API donde instalaremos todos lo necesario
# source api_env/bin/activate

# Instaciión de dependencias para ejecutar API
# pip install -r requirements.txt

# Modificacion para que corra directo setup_api
#!/bin/bash

# Crear entorno (environment) para API
python -m venv api_env || exit 1

# Detectar sistema operativo
#if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
#    ACTIVATE_PATH="api_env/Scripts/activate"
#else
#    ACTIVATE_PATH="api_env/bin/activate"
#fi

# Activar entorno segun sistema operativo
#source api_env/bin/activate # para Libux/Mac
source api_env/Scripts/activate # para Windows
#source "$ACTIVATE_PATH" || { echo "No se pudo activar el entorno"; exit 1; }

# Instalar dependencias
python -m pip install --upgrade pip
pip install -r requirements.txt