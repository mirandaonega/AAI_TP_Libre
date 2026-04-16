# Aprendizaje Automático I - Trabajo Práctico Libre

## Ejecucion Rapida (Linux, macOS y Windows)

Prerequisitos:

1. Tener Docker instalado.
2. Haber ejecutado la seccion 8 del notebook [AAI_TP_Libre.ipynb](AAI_TP_Libre.ipynb) para generar:
   - `docker/pipeline.pkl`
   - `docker/pipeline_logistica.pkl`

### Linux / macOS

Desde la raiz del proyecto:

```bash
cd docker
docker build -t tp_libre_aa1 .

# Regresion (guarda output.csv en la raiz del proyecto)
docker run --rm -v "$PWD/..:/files" tp_libre_aa1

# Clasificacion (guarda output_logistica.csv en la raiz del proyecto)
docker run --rm -v "$PWD/..:/files" tp_libre_aa1 --pipeline /TP_LIBRE/pipeline_logistica.pkl --output /files/output_logistica.csv
```

### Windows PowerShell

Desde la raiz del proyecto:

```powershell
cd docker
docker build -t tp_libre_aa1 .

# Regresion (guarda output.csv en la raiz del proyecto)
docker run --rm -v "${PWD}\..:/files" tp_libre_aa1

# Clasificacion (guarda output_logistica.csv en la raiz del proyecto)
docker run --rm -v "${PWD}\..:/files" tp_libre_aa1 --pipeline /TP_LIBRE/pipeline_logistica.pkl --output /files/output_logistica.csv
```

## Flujo Del Proyecto

1. Ejecutar la seccion 8 del notebook [AAI_TP_Libre.ipynb](AAI_TP_Libre.ipynb) para entrenar y guardar pipelines.
2. En clasificacion, el bundle guarda los umbrales optimizados (`threshold_logistica` y `threshold_red_neuronal_clasificacion`) y la inferencia los aplica automaticamente.
3. Ejecutar [docker/inferencia.py](docker/inferencia.py) dentro del contenedor mediante los comandos anteriores.

## Archivos En Docker

- [docker/Dockerfile](docker/Dockerfile)
- [docker/requirements.txt](docker/requirements.txt)
- [docker/inferencia.py](docker/inferencia.py)
- `docker/pipeline.pkl`
- `docker/pipeline_logistica.pkl`

## Salidas De Inferencia

- Con `pipeline.pkl`:
  - `Valor_Real_Regresion_mm` (si el input incluye `RainfallTomorrow` o `RainfallTomorrow (mm)`)
  - `Prediccion_Lineal_mm`
  - `Prediccion_RedNeuronal_mm`
- Con `pipeline_logistica.pkl`:
  - `Valor_Real_Logistica` (si el input incluye `RainTomorrow`)
  - `Prediccion_Logistica`
  - `Prob_Logistica_No`
  - `Prob_Logistica_Yes`
  - `Umbral_Logistica`
  - `Prediccion_RedNeuronal`
  - `Prob_RedNeuronal_No`
  - `Prob_RedNeuronal_Yes`
  - `Umbral_RedNeuronal`
