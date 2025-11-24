# flake8: noqa: E501
import numpy as np
import pandas as pd
import json
import os
import gzip
import pickle
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

ROOT_PATH = Path(__file__).resolve().parent.parent  # Carpeta base del proyecto


class DatasetHandler:
    @staticmethod
    def load_zip_csv(path_rel: str) -> pd.DataFrame:
        return pd.read_csv(ROOT_PATH / path_rel, compression="zip")

    @staticmethod
    def clean_records(df: pd.DataFrame) -> pd.DataFrame:
        df_new = df.copy()
        df_new = df_new[(df_new["MARRIAGE"] != 0) & (df_new["EDUCATION"] != 0)]
        df_new.loc[df_new["EDUCATION"] >= 4, "EDUCATION"] = 4

        df_new = (
            df_new.rename(columns={"default payment next month": "default"})
            .drop("ID", axis=1)
            .dropna()
        )

        return df_new


class ModelFactory:
    def __init__(self):
        self.categorical_fields = ["SEX", "EDUCATION", "MARRIAGE"]
        self.target_field = "default"

    def build_training_pipeline(self, X):
        numeric_fields = [c for c in X.columns if c not in self.categorical_fields]

        preprocess_block = ColumnTransformer(
            [
                ("cats", OneHotEncoder(), self.categorical_fields),
                ("nums", StandardScaler(), numeric_fields),
            ]
        )

        full_pipeline = Pipeline(
            [
                ("prep", preprocess_block),
                ("kbest", SelectKBest(score_func=f_classif)),
                ("pca", PCA()),
                ("mlp", MLPClassifier(max_iter=15000, random_state=21)),
            ]
        )

        hyperparams = {
            "pca__n_components": [None],
            "kbest__k": [20],
            "mlp__hidden_layer_sizes": [(50, 30, 40, 60)],
            "mlp__alpha": [0.26],
            "mlp__learning_rate_init": [0.001],
        }

        return GridSearchCV(
            estimator=full_pipeline,
            param_grid=hyperparams,
            cv=10,
            scoring="balanced_accuracy",
            n_jobs=-1,
            refit=True,
        )

    def fit_model(self, X, y):
        gs = self.build_training_pipeline(X)
        return gs.fit(X, y)


class ScoreReporter:
    @staticmethod
    def generate_metric_entry(name: str, truth, preds) -> dict:
        return {
            "type": "metrics",
            "dataset": name,
            "precision": float(precision_score(truth, preds, zero_division=0)),
            "recall": float(recall_score(truth, preds, zero_division=0)),
            "f1_score": float(f1_score(truth, preds, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, preds)),
        }

    @staticmethod
    def generate_cm_entry(name: str, truth, preds) -> dict:
        cm = confusion_matrix(truth, preds)
        return {
            "type": "cm_matrix",
            "dataset": name,
            "true_0": {
                "predicted_0": int(cm[0, 0]),
                "predicted_1": int(cm[0, 1]),
            },
            "true_1": {
                "predicted_0": int(cm[1, 0]),
                "predicted_1": int(cm[1, 1]),
            },
        }


class DefaultRiskWorkflow:
    def __init__(self):
        self.loader = DatasetHandler()
        self.builder = ModelFactory()
        self.reporter = ScoreReporter()

    def split_features(self, df: pd.DataFrame):
        X_vars = df.drop("default", axis=1)
        y_var = df["default"]
        return X_vars, y_var

    def store_model(self, model, rel_path: str):
        out_path = ROOT_PATH / rel_path
        os.makedirs(out_path.parent, exist_ok=True)
        with gzip.open(out_path, "wb") as f:
            pickle.dump(model, f)

    def store_metrics(self, metrics: list, rel_path: str):
        out_path = ROOT_PATH / rel_path
        os.makedirs(out_path.parent, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for m in metrics:
                f.write(json.dumps(m) + "\n")

    def execute(self):
        train_raw = self.loader.load_zip_csv("files/input/train_data.csv.zip")
        test_raw = self.loader.load_zip_csv("files/input/test_data.csv.zip")

        train_clean = self.loader.clean_records(train_raw)
        test_clean = self.loader.clean_records(test_raw)

        X_train, y_train = self.split_features(train_clean)
        X_test, y_test = self.split_features(test_clean)

        model = self.builder.fit_model(X_train, y_train)

        train_out = model.predict(X_train)
        test_out = model.predict(X_test)

        all_metrics = [
            self.reporter.generate_metric_entry("train", y_train, train_out),
            self.reporter.generate_metric_entry("test", y_test, test_out),
            self.reporter.generate_cm_entry("train", y_train, train_out),
            self.reporter.generate_cm_entry("test", y_test, test_out),
        ]

        self.store_model(model, "files/models/model.pkl.gz")
        self.store_metrics(all_metrics, "files/output/metrics.json")


if __name__ == "__main__":
    runner = DefaultRiskWorkflow()
    runner.execute()
