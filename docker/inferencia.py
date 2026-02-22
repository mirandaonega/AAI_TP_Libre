"""
Script de inferencia — Predicción de Lluvia (WeatherAUS).

Lee un CSV de entrada, aplica el preprocesamiento completo y devuelve
las predicciones de ambas redes neuronales.

Uso dentro del contenedor (valores por defecto):
    python inferencia.py
    python inferencia.py --input /app/input.csv --output /app/output.csv
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

warnings.filterwarnings("ignore")

# ─── RUTAS ───────────────────────────────────────────────────────────────────

ARTIFACTS_DIR = "/app/artifacts"


# ─── CARGA DE ARTEFACTOS ─────────────────────────────────────────────────────

def load_artifacts():
    """Carga todos los binarios generados por save_artifacts.py."""
    numeric_imputers = joblib.load(f"{ARTIFACTS_DIR}/numeric_imputers.joblib")
    cat_imputers     = joblib.load(f"{ARTIFACTS_DIR}/cat_imputers.joblib")
    scaler           = joblib.load(f"{ARTIFACTS_DIR}/scaler.joblib")
    config           = joblib.load(f"{ARTIFACTS_DIR}/config.joblib")
    model_reg        = tf.keras.models.load_model(f"{ARTIFACTS_DIR}/nn_regresion.keras")
    model_clf        = tf.keras.models.load_model(f"{ARTIFACTS_DIR}/nn_clasificacion.keras")
    return numeric_imputers, cat_imputers, scaler, config, model_reg, model_clf


# ─── PREPROCESAMIENTO ────────────────────────────────────────────────────────

def preprocess(df_raw, numeric_imputers, cat_imputers, scaler, config):
    """
    Replica exactamente el pipeline del notebook:
      1. Extracción de Month desde Date
      2. Imputación numérica en cascada (Location+Month → Month → global)
      3. Imputación categórica en cascada
      4. Codificación (sin/cos de viento y mes, binarización de RainToday)
      5. Drop de columnas auxiliares
      6. Escalado con StandardScaler ajustado en training
    """
    df = df_raw.copy()

    # 1. Month desde Date
    if "Month" not in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df["Month"] = df["Date"].dt.month

    # 2. Imputación numérica
    for col, d in numeric_imputers.items():
        if col not in df.columns:
            continue
        def fill_num(row, col=col, d=d):
            v = row[col]
            if pd.isna(v):
                v = d["by_loc_month"].get((row.get("Location"), row.get("Month")), np.nan)
                if pd.isna(v):
                    v = d["by_month"].get(row.get("Month"), np.nan)
                if pd.isna(v):
                    v = d["global"]
            return v
        df[col] = df.apply(fill_num, axis=1)

    # 3. Imputación categórica
    for col, d in cat_imputers.items():
        if col not in df.columns:
            continue
        def fill_cat(row, col=col, d=d):
            v = row[col]
            if pd.isna(v):
                v = d["by_loc_month"].get((row.get("Location"), row.get("Month")), np.nan)
                if pd.isna(v):
                    v = d["by_month"].get(row.get("Month"), np.nan)
                if pd.isna(v):
                    v = d["global"]
            return v
        df[col] = df.apply(fill_cat, axis=1)

    # 4. Codificación
    wind_dir_map = config["wind_dir_map"]
    wind_cols    = ["WindGustDir", "WindDir9am", "WindDir3pm"]

    for col in wind_cols:
        rads = df[col].map(wind_dir_map).astype(float) * (np.pi / 180)
        df[f"{col}_sin"] = np.sin(rads)
        df[f"{col}_cos"] = np.cos(rads)

    month_rads   = df["Month"].astype(float) * (2 * np.pi / 12)
    df["Month_sin"] = np.sin(month_rads)
    df["Month_cos"] = np.cos(month_rads)

    df["RainToday"] = df["RainToday"].map({"No": 0, "Yes": 1})

    # 5. Drop columnas auxiliares
    drop_cols = wind_cols + ["Month", "Location", "Date"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 6. Ordenar columnas según el orden del training y escalar
    feature_names = config["feature_names"]
    df = df[feature_names]

    X_scaled = scaler.transform(df)
    return X_scaled


# ─── INFERENCIA ──────────────────────────────────────────────────────────────

def run_inference(input_path: str, output_path: str) -> None:
    print(f"[1/3] Cargando artefactos desde {ARTIFACTS_DIR} ...")
    numeric_imputers, cat_imputers, scaler, config, model_reg, model_clf = load_artifacts()

    print(f"[2/3] Leyendo datos de entrada: {input_path}")
    df_input = pd.read_csv(input_path)
    print(f"      Registros leídos: {len(df_input)}")

    X_scaled = preprocess(df_input, numeric_imputers, cat_imputers, scaler, config)

    print(f"[3/3] Ejecutando predicciones ...")
    pred_reg   = model_reg.predict(X_scaled, verbose=0).flatten()
    pred_probs = model_clf.predict(X_scaled, verbose=0).flatten()

    umbral     = config["umbral_optimo_clasificacion"]
    pred_clf   = (pred_probs >= umbral).astype(int)
    pred_label = np.where(pred_clf == 1, "Yes", "No")

    df_output = df_input.copy()
    df_output["pred_RainfallTomorrow_mm"] = np.round(pred_reg, 2)
    df_output["pred_RainTomorrow_prob"]   = np.round(pred_probs, 4)
    df_output["pred_RainTomorrow"]        = pred_label

    df_output.to_csv(output_path, index=False)

    print(f"\n✅ Resultados guardados en: {output_path}")
    print()
    print(df_output[["pred_RainfallTomorrow_mm",
                      "pred_RainTomorrow_prob",
                      "pred_RainTomorrow"]].to_string(index=False))


# ─── ENTRYPOINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inferencia de lluvia con redes neuronales (WeatherAUS)."
    )
    parser.add_argument(
        "--input",  default="/app/input.csv",
        help="Ruta al CSV de entrada (default: /app/input.csv).",
    )
    parser.add_argument(
        "--output", default="/app/output.csv",
        help="Ruta al CSV de salida (default: /app/output.csv).",
    )
    args = parser.parse_args()
    run_inference(args.input, args.output)
