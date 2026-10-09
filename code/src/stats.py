"""Análisis estadístico pareado (celda 43 del notebook, Fase 10), trasladado sin cambios (extraído por AST).

Esquema de semillas real del bootstrap (R8): para cada comparación S0 vs escenario,
    seed = BOOTSTRAP_RANDOM_SEED + dataset_offset + comparison_index
con dataset_offset = 0 (CICIDS2017) o 100 (HIKARI-2021) y comparison_index la
posición del escenario en COMPARISON_SCENARIOS. Ver `comparison_seed`.
"""
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

BOOTSTRAP_RANDOM_SEED = 42
N_BOOTSTRAP = 1000
CONFIDENCE_LEVEL = 0.95
ALPHA = 0.05
PRIMARY_METRIC = "f1"
UTILITY_METRICS = ["accuracy", "precision", "recall", "f1", "fpr"]
COMPARISON_SCENARIOS = ["S1", "S2", "S3_EPS_0.1", "S3_EPS_0.5", "S3_EPS_1.0", "S3_EPS_5.0", "S3_EPS_10.0"]


def comparison_seed(dataset_name, comparison_index):
    dataset_offset = 0 if dataset_name == "CICIDS2017" else 100
    return BOOTSTRAP_RANDOM_SEED + dataset_offset + comparison_index


def calculate_binary_metrics_phase10(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    (tn, fp, fn, tp) = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    if fp + tn > 0:
        fpr = fp / (fp + tn)
    else:
        fpr = np.nan
    if fn + tp > 0:
        fnr = fn / (fn + tp)
    else:
        fnr = np.nan
    return {'accuracy': float(accuracy), 'precision': float(precision), 'recall': float(recall), 'f1': float(f1), 'fpr': float(fpr), 'fnr': float(fnr), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


def safe_divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    result = np.zeros_like(numerator, dtype=float)
    np.divide(numerator, denominator, out=result, where=denominator != 0)
    return result


def metrics_from_confusion_arrays(tn, fp, fn, tp):
    total = tn + fp + fn + tp
    accuracy = safe_divide(tp + tn, total)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2.0 * precision * recall, precision + recall)
    fpr = safe_divide(fp, fp + tn)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'fpr': fpr}


def paired_bootstrap_all_binary_metrics_fast(y_true, y_pred_baseline, y_pred_scenario, n_bootstrap=N_BOOTSTRAP, confidence_level=CONFIDENCE_LEVEL, random_seed=BOOTSTRAP_RANDOM_SEED):
    y_true = np.asarray(y_true, dtype=np.int8)
    baseline = np.asarray(y_pred_baseline, dtype=np.int8)
    scenario = np.asarray(y_pred_scenario, dtype=np.int8)
    assert len(y_true) == len(baseline)
    assert len(y_true) == len(scenario)
    categories = y_true * 4 + baseline * 2 + scenario
    category_counts = np.bincount(categories, minlength=8)
    n = int(len(y_true))
    probabilities = category_counts / n
    rng = np.random.default_rng(random_seed)
    bootstrap_counts = rng.multinomial(n=n, pvals=probabilities, size=n_bootstrap)
    baseline_tn = bootstrap_counts[:, 0] + bootstrap_counts[:, 1]
    baseline_fp = bootstrap_counts[:, 2] + bootstrap_counts[:, 3]
    baseline_fn = bootstrap_counts[:, 4] + bootstrap_counts[:, 5]
    baseline_tp = bootstrap_counts[:, 6] + bootstrap_counts[:, 7]
    scenario_tn = bootstrap_counts[:, 0] + bootstrap_counts[:, 2]
    scenario_fp = bootstrap_counts[:, 1] + bootstrap_counts[:, 3]
    scenario_fn = bootstrap_counts[:, 4] + bootstrap_counts[:, 6]
    scenario_tp = bootstrap_counts[:, 5] + bootstrap_counts[:, 7]
    baseline_metrics = metrics_from_confusion_arrays(baseline_tn, baseline_fp, baseline_fn, baseline_tp)
    scenario_metrics = metrics_from_confusion_arrays(scenario_tn, scenario_fp, scenario_fn, scenario_tp)
    baseline_point = calculate_binary_metrics_phase10(y_true, baseline)
    scenario_point = calculate_binary_metrics_phase10(y_true, scenario)
    alpha_ci = 1.0 - confidence_level
    results = []
    for metric in UTILITY_METRICS:
        bootstrap_difference = scenario_metrics[metric] - baseline_metrics[metric]
        ci_lower = float(np.quantile(bootstrap_difference, alpha_ci / 2.0))
        ci_upper = float(np.quantile(bootstrap_difference, 1.0 - alpha_ci / 2.0))
        baseline_value = float(baseline_point[metric])
        scenario_value = float(scenario_point[metric])
        difference = scenario_value - baseline_value
        if baseline_value != 0:
            relative_degradation = (baseline_value - scenario_value) / baseline_value * 100.0
        else:
            relative_degradation = np.nan
        results.append({'metric': metric, 'baseline_value': baseline_value, 'scenario_value': scenario_value, 'difference': float(difference), 'ci_lower': ci_lower, 'ci_upper': ci_upper, 'confidence_level': float(confidence_level), 'n_bootstrap': int(n_bootstrap), 'relative_degradation_percent': float(relative_degradation) if not np.isnan(relative_degradation) else np.nan, 'ci_excludes_zero': bool(ci_lower > 0 or ci_upper < 0)})
    return results


def exact_mcnemar_phase10(y_true, y_pred_baseline, y_pred_scenario):
    y_true = np.asarray(y_true)
    baseline = np.asarray(y_pred_baseline)
    scenario = np.asarray(y_pred_scenario)
    baseline_correct = baseline == y_true
    scenario_correct = scenario == y_true
    b = int(np.sum(baseline_correct & ~scenario_correct))
    c = int(np.sum(~baseline_correct & scenario_correct))
    discordant = b + c
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = float(binomtest(k=min(b, c), n=discordant, p=0.5, alternative='two-sided').pvalue)
    return {'baseline_correct_scenario_wrong': b, 'baseline_wrong_scenario_correct': c, 'discordant_pairs': int(discordant), 'p_value': float(p_value)}


def holm_adjustment_phase10(p_values):
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    if m == 0:
        return np.array([], dtype=float)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    adjusted_sorted = np.empty(m, dtype=float)
    running_max = 0.0
    for (rank, p_value) in enumerate(sorted_p):
        multiplier = m - rank
        adjusted_value = min(1.0, multiplier * p_value)
        running_max = max(running_max, adjusted_value)
        adjusted_sorted[rank] = running_max
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted
