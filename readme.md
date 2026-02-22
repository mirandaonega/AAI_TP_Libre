# Docker — Inferencia de Lluvia (WeatherAUS)

## Descripción

Contenedor Docker que aplica dos redes neuronales para predecir:

| Salida | Modelo | Descripción |
|--------|--------|-------------|
| `pred_RainfallTomorrow_mm` | Regresión | Cantidad de lluvia estimada para mañana (mm) |
| `pred_RainTomorrow_prob`   | Clasificación | Probabilidad de que llueva [0 – 1] |
| `pred_RainTomorrow`        | Clasificación | Predicción final: `Yes` / `No` |

---

## Paso 1 — Generar los artefactos del modelo

> Este paso se realiza **una sola vez** y produce los binarios que la imagen Docker necesita.

Desde la **raíz del repositorio**, ejecutar con el mismo entorno Python del notebook:

```bash
python save_artifacts.py
```

El script re-entrena ambas redes desde cero usando `weatherAUS.csv` y guarda los binarios en `docker/artifacts/`:

```
docker/artifacts/
├── numeric_imputers.joblib   ← Imputadores para variables numéricas
├── cat_imputers.joblib       ← Imputadores para variables categóricas
├── scaler.joblib             ← StandardScaler ajustado al set de entrenamiento
├── nn_regresion.keras        ← Red neuronal de regresión
├── nn_clasificacion.keras    ← Red neuronal de clasificación
└── config.joblib             ← Umbral óptimo, nombres de features, etc.
```

---

## Paso 2 — Construir la imagen Docker

Desde la carpeta `docker/`:

```bash
cd docker
docker build -t inferencia-lluvia .
```

---

## Paso 3 — Ejecutar el contenedor

El contenedor espera:
- **Entrada:** `/app/input.csv` — CSV con los datos a predecir (montado como volumen)
- **Salida:** `/app/output.csv` — CSV con las predicciones (escrito en la misma carpeta)

### Linux / macOS

```bash
docker run --rm \
  -v /ruta/absoluta/input.csv:/app/input.csv \
  -v /ruta/absoluta/output_dir:/app \
  inferencia-lluvia
```

### Windows (PowerShell)

```powershell
docker run --rm `
  -v C:\ruta\a\input.csv:/app/input.csv `
  -v C:\ruta\a\output_dir:/app `
  inferencia-lluvia
```

> El archivo `output.csv` se creará dentro de `output_dir/`.

---

## Formato del CSV de entrada

El archivo debe tener exactamente las siguientes columnas (mismos nombres que el dataset original `weatherAUS.csv`):

| Columna | Tipo | Ejemplo |
|---------|------|---------|
| `Date` | YYYY-MM-DD | 2017-06-15 |
| `Location` | string | Sydney |
| `MinTemp (°C)` | float | 12.3 |
| `MaxTemp (°C)` | float | 24.5 |
| `RainfallToday (mm)` | float | 0.0 |
| `Evaporation (mm)` | float | 4.2 |
| `Sunshine (h)` | float | 7.5 |
| `WindGustDir` | cardinal | NW |
| `WindGustSpeed (km/h)` | float | 44.0 |
| `WindDir9am` | cardinal | W |
| `WindDir3pm` | cardinal | NW |
| `WindSpeed9am (km/h)` | float | 17.0 |
| `WindSpeed3pm (km/h)` | float | 20.0 |
| `Humidity9am (%)` | float | 68.0 |
| `Humidity3pm (%)` | float | 42.0 |
| `Pressure9am (hPa)` | float | 1013.2 |
| `Pressure3pm (hPa)` | float | 1010.5 |
| `Cloud9am (oktas)` | float | 3.0 |
| `Cloud3pm (oktas)` | float | 5.0 |
| `Temp9am (°C)` | float | 16.5 |
| `Temp3pm (°C)` | float | 23.1 |
| `RainToday` | Yes / No | No |

**Locations válidas:** `Albury`, `Sydney`, `SydneyAirport`, `Canberra`, `Melbourne`, `MelbourneAirport`

Los valores faltantes son aceptados: el contenedor aplica el mismo imputador que se usó durante el entrenamiento.

---

## Estructura de la carpeta `docker/`

```
docker/
├── Dockerfile
├── requirements.txt
├── inferencia.py
├── readme.md
└── artifacts/                      ← generado por save_artifacts.py
    ├── numeric_imputers.joblib
    ├── cat_imputers.joblib
    ├── scaler.joblib
    ├── nn_regresion.keras
    ├── nn_clasificacion.keras
    └── config.joblib
```

---

## Dependencias de la imagen

| Librería | Uso |
|----------|-----|
| `tensorflow` | Carga y ejecución de las redes neuronales `.keras` |
| `scikit-learn` | Aplicación del `StandardScaler` |
| `pandas` / `numpy` | Lectura del CSV y preprocesamiento |
| `joblib` | Deserialización de los artefactos `.joblib` |
