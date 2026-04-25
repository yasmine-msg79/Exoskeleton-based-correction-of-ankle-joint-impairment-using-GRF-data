"""
Binary gait classifier: healthy vs impaired.

Data (see Data/GRF_metadata.csv + GRF CSV):
- Each row is one stance phase (one step), ~101 samples across 0–100% of stance.
- Vertical GRF is BW-normalized (ratio of body weight); columns F_V_PRO_1..101.
- CLASS_LABEL: HC = healthy control; A/K/H/C etc. = impaired cohorts.

This module uses the raw 101-point waveform as features, reduced with PCA, then
SVM (RBF). Train only on rows with TRAIN == 1 to respect the published split.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ArrayLike = Union[np.ndarray, List[float]]


def _is_binary_normal(class_labels: np.ndarray) -> np.ndarray:
    """1 = healthy (HC), 0 = any impairment."""
    s = pd.Series(class_labels).astype(str).str.strip()
    return (s == "HC").astype(np.int32).values


def _grf_feature_matrix(grf_df: pd.DataFrame, n_rows: int) -> np.ndarray:
    """Same columns as SignalReader.get_sample: first 3 cols are IDs, rest are GRF."""
    return grf_df.iloc[:n_rows, 3:].to_numpy(dtype=np.float64)


class Classifier:
    """
    PCA + SVM on the 101-sample stance GRF curve.
    Predicts whether the step is *normal* (HC) or *not normal* (impaired).
    """

    LABEL_NORMAL = "normal"
    LABEL_IMPAIRED = "impaired"

    def __init__(
        self,
        n_components: Union[int, float] = 0.95,
        svm_C: float = 1.0,
        svm_gamma: Union[str, float] = "scale",
        kernel: str = "rbf",
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self._pipe: Optional[Pipeline] = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "pca",
                    PCA(
                        n_components=n_components,
                        random_state=random_state,
                        svd_solver="auto",
                    ),
                ),
                (
                    "svm",
                    SVC(
                        C=svm_C,
                        kernel=kernel,
                        gamma=svm_gamma,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )
        self._fitted = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Classifier":
        """
        Fit from design matrix X (n_samples, 101) and binary labels y (0=impaired, 1=HC).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32).ravel()
        if X.ndim != 2 or X.shape[1] < 2:
            raise ValueError("X must be 2D with one row per step and GRF columns.")
        self._pipe.fit(X, y)
        self._fitted = True
        return self

    def fit_from_reader(
        self,
        reader: Any,
        train_column: str = "TRAIN",
        max_rows: Optional[int] = None,
    ) -> "Classifier":
        """
        Fit using a SignalReader instance: aligns metadata with GRF rows by index.

        Uses rows where ``reader.meta_df[train_column] == 1`` (default TRAIN split).

        Parameters
        ----------
        max_rows
            If set, only the first ``max_rows`` of metadata are used (debug / subsample).
        """
        meta: pd.DataFrame = reader.meta_df
        grf: pd.DataFrame = reader.grf_df
        n_meta = len(meta) if max_rows is None else min(max_rows, len(meta))
        if len(grf) < n_meta:
            raise ValueError(
                f"GRF has {len(grf)} rows but metadata implies {n_meta}. "
                "Files must be row-aligned."
            )

        X = _grf_feature_matrix(grf, n_meta)
        y = _is_binary_normal(meta["CLASS_LABEL"].values[:n_meta])

        if train_column in meta.columns:
            mask = meta[train_column].iloc[:n_meta].to_numpy() == 1
        else:
            mask = np.ones(n_meta, dtype=bool)

        self.fit(X[mask], y[mask])
        return self

    def predict_label_int(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted or self._pipe is None:
            raise RuntimeError("Call fit() or fit_from_reader() before predict.")
        X = np.asarray(X, dtype=np.float64)
        return self._pipe.predict(np.atleast_2d(X))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Signed score from the SVM; higher => more confidence toward the positive class (HC)."""
        if not self._fitted or self._pipe is None:
            raise RuntimeError("Classifier is not fitted.")
        X = np.asarray(X, dtype=np.float64)
        return self._pipe.decision_function(np.atleast_2d(X))

    def classify(self, sample: Dict[str, Any]) -> List[str]:
        """
        Compatible with the legacy API: accepts a ``sample`` dict from ``SignalReader.get_sample``
        (must contain ``grf``) and returns a one-element list of ``'normal'`` or ``'impaired'``.
        """
        if not isinstance(sample, dict) or "grf" not in sample:
            raise TypeError("classify() expects a dict with a 'grf' key (101-sample array).")
        grf = np.asarray(sample["grf"], dtype=np.float64).ravel()
        pred = int(self.predict_label_int(grf.reshape(1, -1))[0])
        return [self.LABEL_NORMAL if pred == 1 else self.LABEL_IMPAIRED]

    def predict_row(self, grf: ArrayLike) -> Tuple[str, float]:
        """
        Returns (label_string, decision_function_score) for one stance curve.
        """
        v = np.asarray(grf, dtype=np.float64).reshape(1, -1)
        pred = int(self.predict_label_int(v)[0])
        score = float(self.decision_function(v)[0])
        name = self.LABEL_NORMAL if pred == 1 else self.LABEL_IMPAIRED
        return name, score

    def evaluate_on_split(
        self,
        reader: Any,
        split_column: str = "TEST",
        max_rows: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Accuracy / balanced accuracy on rows where ``split_column == 1`` (e.g. TEST).
        Requires a fitted model.
        """
        from sklearn.metrics import accuracy_score, balanced_accuracy_score

        meta: pd.DataFrame = reader.meta_df
        grf: pd.DataFrame = reader.grf_df
        n_meta = len(meta) if max_rows is None else min(max_rows, len(meta))
        if split_column not in meta.columns:
            raise ValueError(f"Metadata has no column {split_column!r}.")

        mask = meta[split_column].iloc[:n_meta].to_numpy() == 1
        if not np.any(mask):
            raise ValueError(f"No rows with {split_column}==1.")

        X = _grf_feature_matrix(grf, n_meta)[mask]
        y = _is_binary_normal(meta["CLASS_LABEL"].values[:n_meta])[mask]
        y_hat = self.predict_label_int(X)

        return {
            "accuracy": float(accuracy_score(y, y_hat)),
            "balanced_accuracy": float(balanced_accuracy_score(y, y_hat)),
            "n": float(len(y)),
        }

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"pipe": self._pipe, "fitted": self._fitted, "n_components": self.n_components}
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, path: Union[str, Path]) -> "Classifier":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self._pipe = payload["pipe"]
        self._fitted = payload["fitted"]
        self.n_components = payload.get("n_components", self.n_components)
        return self
