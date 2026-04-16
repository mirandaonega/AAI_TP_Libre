import argparse
import logging
from pathlib import Path
from sys import stdout
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler

warnings.simplefilter("ignore")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter("%(asctime)s %(levelname)s %(filename)s: %(message)s")
console_handler = logging.StreamHandler(stdout)
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PIPELINE = BASE_DIR / "pipeline.pkl"


WIND_DIR_MAP = {
    "N": 0,
    "NNE": 22.5,
    "NE": 45,
    "ENE": 67.5,
    "E": 90,
    "ESE": 112.5,
    "SE": 135,
    "SSE": 157.5,
    "S": 180,
    "SSW": 202.5,
    "SW": 225,
    "WSW": 247.5,
    "W": 270,
    "WNW": 292.5,
    "NW": 315,
    "NNW": 337.5,
}


class NotebookWeatherPreprocessor(BaseEstimator, TransformerMixin):
    """Replica el preprocesamiento del trabajo: 3 imputaciones + codificacion ciclica."""

    def __init__(self):
        self.cols_mediana = [
            "MinTemp",
            "MaxTemp",
            "Rainfall",
            "WindSpeed9am",
            "WindSpeed3pm",
            "Humidity9am",
            "Humidity3pm",
            "Pressure9am",
            "Pressure3pm",
            "Temp9am",
            "Temp3pm",
        ]
        self.cols_knn = [
            "Evaporation",
            "Sunshine",
            "WindGustSpeed",
            "Cloud9am",
            "Cloud3pm",
        ]
        self.cols_moda = ["WindGustDir", "WindDir9am", "WindDir3pm", "RainToday"]
        self.wind_cols = ["WindGustDir", "WindDir9am", "WindDir3pm"]
        self.knn_neighbors = 5

    def _ensure_month(self, df):
        if "Month" not in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df["Month"] = df["Date"].dt.month
        return df

    def fit(self, X, y=None):
        df = X.copy()
        df = self._ensure_month(df)

        self.numeric_stats_ = {}
        for col in self.cols_mediana:
            self.numeric_stats_[col] = {
                "by_loc_month": df.groupby(["Location", "Month"])[col].median().to_dict(),
                "by_month": df.groupby("Month")[col].median().to_dict(),
                "global": float(df[col].median()),
            }

        self.cat_stats_ = {}
        for col in self.cols_moda:
            by_loc_month = (
                df.groupby(["Location", "Month"])[col]
                .apply(lambda s: s.mode()[0] if not s.mode().empty else np.nan)
                .to_dict()
            )
            by_month = (
                df.groupby("Month")[col]
                .apply(lambda s: s.mode()[0] if not s.mode().empty else np.nan)
                .to_dict()
            )
            global_mode = df[col].mode()
            self.cat_stats_[col] = {
                "by_loc_month": by_loc_month,
                "by_month": by_month,
                "global": global_mode[0] if not global_mode.empty else np.nan,
            }

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.numeric_cols_ = numeric_cols
        self.cols_knn_valid_ = [c for c in self.cols_knn if c in numeric_cols]
        self.knn_imputer_ = KNNImputer(n_neighbors=self.knn_neighbors, weights="distance")
        self.knn_imputer_.fit(df[numeric_cols])

        transformed = self._transform_core(df)
        self.feature_names_ = transformed.columns.tolist()
        self.scaler_ = RobustScaler()
        self.scaler_.fit(transformed)
        return self

    def _fill_numeric(self, row, col):
        val = row[col]
        if pd.isna(val):
            stats = self.numeric_stats_[col]
            val = stats["by_loc_month"].get((row.get("Location"), row.get("Month")), np.nan)
            if pd.isna(val):
                val = stats["by_month"].get(row.get("Month"), np.nan)
            if pd.isna(val):
                val = stats["global"]
        return val

    def _fill_categorical(self, row, col):
        val = row[col]
        if pd.isna(val):
            stats = self.cat_stats_[col]
            val = stats["by_loc_month"].get((row.get("Location"), row.get("Month")), np.nan)
            if pd.isna(val):
                val = stats["by_month"].get(row.get("Month"), np.nan)
            if pd.isna(val):
                val = stats["global"]
        return val

    def _transform_core(self, df):
        df_out = df.copy()

        for col in self.cols_mediana:
            if col in df_out.columns:
                df_out[col] = df_out.apply(lambda row, c=col: self._fill_numeric(row, c), axis=1)

        if self.cols_knn_valid_:
            num_df = pd.DataFrame(
                self.knn_imputer_.transform(df_out[self.numeric_cols_]),
                columns=self.numeric_cols_,
                index=df_out.index,
            )
            for col in self.cols_knn_valid_:
                missing = df_out[col].isna()
                df_out.loc[missing, col] = num_df.loc[missing, col]
                if col in ["Cloud9am", "Cloud3pm"]:
                    df_out[col] = df_out[col].round().clip(0, 8)

        for col in self.cols_moda:
            if col in df_out.columns:
                df_out[col] = df_out.apply(
                    lambda row, c=col: self._fill_categorical(row, c), axis=1
                )

        for col in self.wind_cols:
            rads = df_out[col].map(WIND_DIR_MAP).astype(float) * (np.pi / 180)
            df_out[f"{col}_sin"] = np.sin(rads)
            df_out[f"{col}_cos"] = np.cos(rads)

        month_rads = df_out["Month"].astype(float) * (2 * np.pi / 12)
        df_out["Month_sin"] = np.sin(month_rads)
        df_out["Month_cos"] = np.cos(month_rads)

        df_out["RainToday"] = df_out["RainToday"].map({"No": 0, "Yes": 1})

        cols_to_drop = self.wind_cols + ["Month", "Location", "Date"]
        df_out = df_out.drop(columns=[c for c in cols_to_drop if c in df_out.columns])
        return df_out

    def transform(self, X):
        df = X.copy()
        df = self._ensure_month(df)
        transformed = self._transform_core(df)
        transformed = transformed[self.feature_names_]
        scaled = self.scaler_.transform(transformed)
        return scaled


# Alias explicito para compatibilidad de deserializacion si el pickle fue
# creado con nombre de clase diferente en notebook.
NotebookWeatherPreprocessorClassification = NotebookWeatherPreprocessor

INPUT_ALIASES = {
    "RainfallToday (mm)": "Rainfall",
    "MinTemp (C)": "MinTemp",
    "MaxTemp (C)": "MaxTemp",
    "WindGustSpeed (km/h)": "WindGustSpeed",
    "WindSpeed9am (km/h)": "WindSpeed9am",
    "WindSpeed3pm (km/h)": "WindSpeed3pm",
    "Humidity9am (%)": "Humidity9am",
    "Humidity3pm (%)": "Humidity3pm",
    "Pressure9am (hPa)": "Pressure9am",
    "Pressure3pm (hPa)": "Pressure3pm",
    "Cloud9am (oktas)": "Cloud9am",
    "Cloud3pm (oktas)": "Cloud3pm",
    "Temp9am (C)": "Temp9am",
    "Temp3pm (C)": "Temp3pm",
    "Evaporation (mm)": "Evaporation",
    "Sunshine (h)": "Sunshine",
}


FEATURE_COLUMNS = [
    "Date",
    "Location",
    "MinTemp",
    "MaxTemp",
    "Rainfall",
    "Evaporation",
    "Sunshine",
    "WindGustDir",
    "WindGustSpeed",
    "WindDir9am",
    "WindDir3pm",
    "WindSpeed9am",
    "WindSpeed3pm",
    "Humidity9am",
    "Humidity3pm",
    "Pressure9am",
    "Pressure3pm",
    "Cloud9am",
    "Cloud3pm",
    "Temp9am",
    "Temp3pm",
    "RainToday",
]

REGRESSION_TARGET_COLUMNS = [
    "RainfallTomorrow",
    "RainfallTomorrow (mm)",
]

CLASSIFICATION_TARGET_COLUMNS = [
    "RainTomorrow",
]


def normalize_input_columns(df_in: pd.DataFrame, feature_columns) -> pd.DataFrame:
    df = df_in.rename(columns=INPUT_ALIASES).copy()
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas para inferencia: {missing}")
    return df[feature_columns]


def first_existing_column(df_in: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df_in.columns:
            return col
    return None


def normalize_class_actuals(series: pd.Series) -> pd.Series:
    mapping = {
        0: "No",
        1: "Yes",
        "0": "No",
        "1": "Yes",
        "No": "No",
        "Yes": "Yes",
        "no": "No",
        "yes": "Yes",
        False: "No",
        True: "Yes",
        "false": "No",
        "true": "Yes",
    }

    normalized = series.map(mapping)
    as_text = series.astype("string")
    normalized = normalized.astype("string").fillna(as_text)
    normalized = normalized.where(~series.isna(), pd.NA)
    return normalized


def run_inference(input_path: Path, output_path: Path, pipeline_path: Path) -> None:
    print(sklearn.__version__)

    pipeline_bundle = joblib.load(pipeline_path)
    logger.info("Se carga el pipeline bundle: %s", pipeline_path)

    df_input = pd.read_csv(input_path)
    logger.info("Se leen los datos del archivo de entrada: %s", input_path)

    print("shape:", df_input.shape)
    print(df_input.head())

    feature_columns = pipeline_bundle.get("feature_columns", FEATURE_COLUMNS)
    X_input = normalize_input_columns(df_input, feature_columns)

    # Modo regresion: bundle con pipeline lineal + red neuronal regresora.
    if {
        "pipeline_regresion_lineal",
        "pipeline_red_neuronal",
    }.issubset(pipeline_bundle.keys()):
        pipeline_lr = pipeline_bundle["pipeline_regresion_lineal"]
        pipeline_nn = pipeline_bundle["pipeline_red_neuronal"]

        y_pred_lr = np.clip(pipeline_lr.predict(X_input), 0, None)
        y_pred_nn = np.clip(pipeline_nn.predict(X_input), 0, None)

        df_resultados = pd.DataFrame(
            {
                "Prediccion_Lineal_mm": np.round(y_pred_lr, 2),
                "Prediccion_RedNeuronal_mm": np.round(y_pred_nn, 2),
            }
        )

        regression_target_col = first_existing_column(df_input, REGRESSION_TARGET_COLUMNS)
        if regression_target_col is not None:
            y_real_reg = pd.to_numeric(df_input[regression_target_col], errors="coerce")
            df_resultados.insert(0, "Valor_Real_Regresion_mm", np.round(y_real_reg, 2))
        else:
            df_resultados.insert(
                0,
                "Valor_Real_Regresion_mm",
                pd.Series([pd.NA] * len(df_resultados), dtype="Float64"),
            )
            logger.warning(
                "No se encontro columna real de regresion en input (%s)",
                REGRESSION_TARGET_COLUMNS,
            )

    # Modo clasificacion: bundle con pipeline logistica + red neuronal clasificadora.
    elif {
        "pipeline_logistica",
        "pipeline_red_neuronal_clasificacion",
    }.issubset(pipeline_bundle.keys()):
        pipeline_log = pipeline_bundle["pipeline_logistica"]
        pipeline_nn_cls = pipeline_bundle["pipeline_red_neuronal_clasificacion"]

        threshold_log = float(pipeline_bundle.get("threshold_logistica", 0.5))
        threshold_nn = float(
            pipeline_bundle.get("threshold_red_neuronal_clasificacion", 0.5)
        )

        prob_log = pipeline_log.predict_proba(X_input)
        pred_log = (prob_log[:, 1] >= threshold_log).astype(int)

        prob_nn = pipeline_nn_cls.predict_proba(X_input)
        pred_nn = (prob_nn[:, 1] >= threshold_nn).astype(int)

        # Se usa mapeo binario estandar 0/1 a etiquetas del trabajo.
        label_map = {0: "No", 1: "Yes"}
        pred_log_label = pd.Series(pred_log).map(label_map).astype(str)
        pred_nn_label = pd.Series(pred_nn).map(label_map).astype(str)

        df_resultados = pd.DataFrame(
            {
                "Prediccion_Logistica": pred_log_label,
                "Prob_Logistica_No": np.round(prob_log[:, 0], 4),
                "Prob_Logistica_Yes": np.round(prob_log[:, 1], 4),
                "Umbral_Logistica": np.round(np.repeat(threshold_log, len(X_input)), 4),
                "Prediccion_RedNeuronal": pred_nn_label,
                "Prob_RedNeuronal_No": np.round(prob_nn[:, 0], 4),
                "Prob_RedNeuronal_Yes": np.round(prob_nn[:, 1], 4),
                "Umbral_RedNeuronal": np.round(np.repeat(threshold_nn, len(X_input)), 4),
            }
        )

        classification_target_col = first_existing_column(
            df_input, CLASSIFICATION_TARGET_COLUMNS
        )
        if classification_target_col is not None:
            y_real_class = normalize_class_actuals(df_input[classification_target_col])
            df_resultados.insert(0, "Valor_Real_Logistica", y_real_class)
        else:
            df_resultados.insert(
                0,
                "Valor_Real_Logistica",
                pd.Series([pd.NA] * len(df_resultados), dtype="string"),
            )
            logger.warning(
                "No se encontro columna real de clasificacion en input (%s)",
                CLASSIFICATION_TARGET_COLUMNS,
            )

    else:
        raise ValueError(
            "El archivo de pipeline no tiene una estructura soportada. "
            "Esperado: bundle de regresion o bundle de clasificacion."
        )

    logger.info("Se realizan las predicciones")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_resultados.to_csv(output_path, index=False)

    logger.info("Se guarda la salida en el archivo: %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia con pipeline.pkl")
    parser.add_argument("--input", default="/TP_LIBRE/input.csv", help="CSV de entrada")
    parser.add_argument("--output", default="/files/output.csv", help="CSV de salida")
    parser.add_argument(
        "--pipeline",
        default=str(DEFAULT_PIPELINE),
        help="Ruta al archivo pipeline.pkl",
    )
    args = parser.parse_args()

    run_inference(
        input_path=Path(args.input),
        output_path=Path(args.output),
        pipeline_path=Path(args.pipeline),
    )
