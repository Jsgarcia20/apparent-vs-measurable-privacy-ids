"""Random Forest final (celda 24 del notebook, sin cambios de lógica)."""
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def class_weight_for(cfg, dataset=None):
    """class_weight del dataset si está definido (null = None); si no, el global."""
    if dataset is not None and "class_weight" in cfg["datasets"][dataset]:
        return cfg["datasets"][dataset]["class_weight"]
    return cfg["random_forest"]["class_weight"]


def create_random_forest(cfg, dataset=None, class_weight="__from_config__"):
    rf = cfg["random_forest"]
    if class_weight == "__from_config__":
        class_weight = class_weight_for(cfg, dataset)
    return RandomForestClassifier(
        n_estimators=rf["n_estimators"],
        max_depth=rf["max_depth"],
        min_samples_split=rf["min_samples_split"],
        min_samples_leaf=rf["min_samples_leaf"],
        max_features=rf["max_features"],
        bootstrap=rf["bootstrap"],
        class_weight=class_weight,
        random_state=cfg["seeds"]["rf"],
        n_jobs=rf["n_jobs"],
    )


def train_predict(cfg, X_train, y_train, X_test, decision_threshold, dataset=None):
    """Entrena y devuelve (probabilidades de ATTACK, predicciones con umbral, tiempos)."""
    assert list(X_train.columns) == list(X_test.columns)
    assert 0.0 < decision_threshold < 1.0
    model = create_random_forest(cfg, dataset)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    attack_probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (attack_probabilities >= decision_threshold).astype(np.int8)
    prediction_seconds = time.perf_counter() - t0
    del model
    return attack_probabilities, predictions, training_seconds, prediction_seconds
