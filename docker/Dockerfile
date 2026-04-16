# Utilizo una imagen base de Python 3.11 slim para reducir el tamaño del contenedor
FROM python:3.11-slim

#Creo una carpeta para el proyecto  dentro del contenedor docker
WORKDIR /TP_LIBRE

#Necesario para no recibir error de tensorflow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

#Copio dependencias para el funcionamiento del TP y las instalo
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#Guardo el codigo dentro de la carpeta TP_LIBRE del contenedor
COPY . . 

# Ejecuto el script (permite pasar argumentos en docker run)
ENTRYPOINT ["python", "inferencia.py"]