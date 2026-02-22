import os
import warnings
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ─── CONFIGURACIÓN ───────────────────────────────────────────────────────────

DATA_PATH     = "weatherAUS.csv"
ARTIFACTS_DIR = "docker/artifacts"

LOCATIONS = [
    "Albury", "Sydney", "SydneyAirport",
    "Canberra", "Melbourne", "MelbourneAirport",
]

WIND_DIR_MAP = {
    "N": 0,   "NNE": 22.5, "NE": 45,  "ENE": 67.5,
    "E": 90,  "ESE": 112.5,"SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5,"SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5,"NW": 315, "NNW": 337.5,
}

COLS_NUM = [
    "MinTemp (°C)", "MaxTemp (°C)", "RainfallToday (mm)", "Evaporation (mm)",
    "Sunshine (h)", "WindGustSpeed (km/h)", "WindSpeed9am (km/h)",
    "WindSpeed3pm (km/h)", "Humidity9am (%)", "Humidity3pm (%)",
    "Pressure9am (hPa)", "Pressure3pm (hPa)", "Cloud9am (oktas)",
    "Cloud3pm (oktas)", "Temp9am (°C)", "Temp3pm (°C)",
]

COLS_CAT = ["WindGustDir", "WindDir9am", "WindDir3pm", "RainToday"]

RENAME_MAP = {
    "MinTemp": "MinTemp (°C)", "MaxTemp": "MaxTemp (°C)",
    "Rainfall": "RainfallToday (mm)", "Evaporation": "Evaporation (mm)",
    "Sunshine": "Sunshine (h)", "WindGustSpeed": "WindGustSpeed (km/h)",
    "WindSpeed9am": "WindSpeed9am (km/h)", "WindSpeed3pm": "WindSpeed3pm (km/h)",
    "Humidity9am": "Humidity9am (%)", "Humidity3pm": "Humidity3pm (%)",
    "Pressure9am": "Pressure9am (hPa)", "Pressure3pm": "Pressure3pm (hPa)",
    "Cloud9am": "Cloud9am (oktas)", "Cloud3pm": "Cloud3pm (oktas)",
    "Temp9am": "Temp9am (°C)", "Temp3pm": "Temp3pm (°C)",
    "RainfallTomorrow": "RainfallTomorrow (mm)",
}

# ─── PASO 1: CARGA Y PREPARACIÓN ─────────────────────────────────────────────

print("=" * 60)
print("  Generador de Artefactos para Docker - WeatherAUS")
print("=" * 60)

print("\n[1/6] Cargando y preparando datos...")

df = pd.read_csv(DATA_PATH)
df = df[df["Location"].isin(LOCATIONS)].reset_index(drop=True)
df = df.sort_values(by=["Location", "Date"])
df["RainfallTomorrow"] = df.groupby("Location")["Rainfall"].shift(-1)
df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.month
df.rename(columns=RENAME_MAP, inplace=True)

df = df.dropna(thresh=df.shape[1] // 2)
df = df.dropna(subset=["RainTomorrow", "RainfallTomorrow (mm)"]).reset_index(drop=True)

print(f"   Registros listos: {len(df):,}")

# ─── PASO 2: SPLIT ───────────────────────────────────────────────────────────

X = df.drop(columns=["RainTomorrow", "RainfallTomorrow (mm)", "Date"])
y_class = df["RainTomorrow"]
y_reg   = df["RainfallTomorrow (mm)"]

X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
    X, y_class, y_reg,
    test_size=0.2, random_state=42,
    stratify=y_class,
)

# ─── PASO 3: APRENDER IMPUTADORES ────────────────────────────────────────────

print("[2/6] Aprendiendo imputadores...")

numeric_imputers = {}
for col in COLS_NUM:
    numeric_imputers[col] = {
        "by_loc_month": X_train.groupby(["Location", "Month"])[col].median().to_dict(),
        "by_month":     X_train.groupby("Month")[col].median().to_dict(),
        "global":       float(X_train[col].median()),
    }

cat_imputers = {}
for col in COLS_CAT:
    def _mode(s):
        m = s.mode()
        return m[0] if not m.empty else np.nan

    cat_imputers[col] = {
        "by_loc_month": X_train.groupby(["Location", "Month"])[col].apply(_mode).to_dict(),
        "by_month":     X_train.groupby("Month")[col].apply(_mode).to_dict(),
        "global":       X_train[col].mode()[0],
    }


def _apply_imputation(df_in, num_imp, cat_imp):
    df_out = df_in.copy()

    for col, d in num_imp.items():
        if col not in df_out.columns:
            continue
        def fill_num(row, col=col, d=d):
            v = row[col]
            if pd.isna(v):
                v = d["by_loc_month"].get((row["Location"], row["Month"]), np.nan)
                if pd.isna(v):
                    v = d["by_month"].get(row["Month"], np.nan)
                if pd.isna(v):
                    v = d["global"]
            return v
        df_out[col] = df_out.apply(fill_num, axis=1)

    for col, d in cat_imp.items():
        if col not in df_out.columns:
            continue
        def fill_cat(row, col=col, d=d):
            v = row[col]
            if pd.isna(v):
                v = d["by_loc_month"].get((row["Location"], row["Month"]), np.nan)
                if pd.isna(v):
                    v = d["by_month"].get(row["Month"], np.nan)
                if pd.isna(v):
                    v = d["global"]
            return v
        df_out[col] = df_out.apply(fill_cat, axis=1)

    return df_out


X_train_imp = _apply_imputation(X_train, numeric_imputers, cat_imputers)
X_test_imp  = _apply_imputation(X_test,  numeric_imputers, cat_imputers)

# ─── PASO 4: CODIFICACIÓN ────────────────────────────────────────────────────

def transformar_dataset_final(df_imputed):
    df_res = df_imputed.copy()
    wind_cols = ["WindGustDir", "WindDir9am", "WindDir3pm"]

    for col in wind_cols:
        rads = df_res[col].map(WIND_DIR_MAP).astype(float) * (np.pi / 180)
        df_res[f"{col}_sin"] = np.sin(rads)
        df_res[f"{col}_cos"] = np.cos(rads)

    month_rads = df_res["Month"].astype(float) * (2 * np.pi / 12)
    df_res["Month_sin"] = np.sin(month_rads)
    df_res["Month_cos"] = np.cos(month_rads)

    df_res["RainToday"] = df_res["RainToday"].map({"No": 0, "Yes": 1})
    df_res = df_res.drop(columns=wind_cols + ["Month", "Location"])
    return df_res


X_train_ready = transformar_dataset_final(X_train_imp)
X_test_ready  = transformar_dataset_final(X_test_imp)

y_train_class_num = y_train_class.map({"No": 0, "Yes": 1})
y_test_class_num  = y_test_class.map({"No": 0, "Yes": 1})

# ─── PASO 5: ESCALADO ────────────────────────────────────────────────────────

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_ready), columns=X_train_ready.columns)
X_test_scaled  = pd.DataFrame(scaler.transform(X_test_ready),      columns=X_test_ready.columns)

# ─── PASO 6: SPLIT VALIDACIÓN ────────────────────────────────────────────────

y_train_class_num = y_train_class_num.reset_index(drop=True)
y_train_reg       = y_train_reg.reset_index(drop=True)

(X_train_final, X_val_scaled,
 y_train_final_class, y_val_class,
 y_train_final_reg, y_val_reg) = train_test_split(
    X_train_scaled, y_train_class_num, y_train_reg,
    test_size=0.2, random_state=42,
    stratify=y_train_class_num,
)

sm = SMOTE(random_state=42)
X_train_smote_final, y_train_smote_final = sm.fit_resample(X_train_final, y_train_final_class)

# ─── PASO 6: ENTRENAMIENTO ───────────────────────────────────────────────────

print("[3/6] Entrenando red neuronal de regresión (100 épocas)...")

model_reg = models.Sequential([
    layers.Input(shape=(X_train_final.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1, activation="relu"),
])
model_reg.compile(optimizer=optimizers.Adam(0.001), loss="mse", metrics=["mae"])
model_reg.fit(
    X_train_final, y_train_final_reg,
    validation_data=(X_val_scaled, y_val_reg),
    epochs=100, batch_size=64, verbose=0,
)

# Métricas finales en test
loss, mae = model_reg.evaluate(X_test_scaled, y_test_reg, verbose=0)
print(f"   Regresión  → Test MSE: {loss:.4f}  |  Test MAE: {mae:.4f}")

print("[4/6] Entrenando red neuronal de clasificación (100 épocas)...")

model_clf = models.Sequential([
    layers.Input(shape=(X_train_final.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid"),
])
model_clf.compile(
    optimizer=optimizers.Adam(0.001),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)
model_clf.fit(
    X_train_smote_final, y_train_smote_final,
    validation_data=(X_val_scaled, y_val_class),
    epochs=100, batch_size=64, verbose=0,
)

# Umbral óptimo (maximiza F1 en validación)
y_probs_val = model_clf.predict(X_val_scaled, verbose=0).flatten()
umbrales   = np.linspace(0, 1, 100)
f1_scores  = [f1_score(y_val_class, (y_probs_val >= t).astype(int)) for t in umbrales]
umbral_optimo = float(umbrales[np.argmax(f1_scores)])
print(f"   Clasificación → Umbral óptimo (F1): {umbral_optimo:.4f}")

# ─── PASO 7: GUARDAR ARTEFACTOS ──────────────────────────────────────────────

print("[5/6] Guardando artefactos...")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

joblib.dump(numeric_imputers,        f"{ARTIFACTS_DIR}/numeric_imputers.joblib")
joblib.dump(cat_imputers,            f"{ARTIFACTS_DIR}/cat_imputers.joblib")
joblib.dump(scaler,                  f"{ARTIFACTS_DIR}/scaler.joblib")
model_reg.save(                      f"{ARTIFACTS_DIR}/nn_regresion.keras")
model_clf.save(                      f"{ARTIFACTS_DIR}/nn_clasificacion.keras")

config = {
    "umbral_optimo_clasificacion": umbral_optimo,
    "cols_num":      COLS_NUM,
    "cols_cat":      COLS_CAT,
    "wind_dir_map":  WIND_DIR_MAP,
    "locations":     LOCATIONS,
    "feature_names": list(X_train_final.columns),
}
joblib.dump(config, f"{ARTIFACTS_DIR}/config.joblib")

