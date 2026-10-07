import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def validate_predictions(labels, probabilities, uncertainty):
    labels, probabilities, uncertainty = np.asarray(labels), np.asarray(probabilities, dtype=np.float64), np.asarray(uncertainty, dtype=np.float64)
    if labels.ndim != 1 or len(labels) == 0 or probabilities.shape != (len(labels), 2) or uncertainty.shape != (len(labels),):
        raise ValueError("Predictions require nonempty binary labels, N×2 probabilities, and N uncertainties")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(probabilities).all() or not np.isfinite(uncertainty).all():
        raise ValueError("Invalid labels or nonfinite predictions")
    if (probabilities < 0).any() or (probabilities > 1).any() or not np.allclose(probabilities.sum(1), 1, atol=1e-5):
        raise ValueError("Probabilities must be normalized within [0,1]")
    if (uncertainty < 0).any() or (uncertainty > 1).any():
        raise ValueError("Uncertainty must be within [0,1]")
    return labels.astype(np.int64), probabilities, uncertainty


def calibration(labels, probabilities, bins=15):
    if not isinstance(bins, int) or bins < 1:
        raise ValueError("Calibration bins must be a positive integer")
    confidence = probabilities.max(1)
    correct = probabilities.argmax(1) == labels
    indices = np.minimum((confidence * bins).astype(int), bins - 1)
    rows, ece = [], 0.0
    for index in range(bins):
        selected = indices == index
        n = int(selected.sum())
        accuracy = float(correct[selected].mean()) if n else None
        mean_confidence = float(confidence[selected].mean()) if n else None
        if n:
            ece += n / len(labels) * abs(accuracy - mean_confidence)
        rows.append({"bin": index, "count": n, "accuracy": accuracy, "confidence": mean_confidence})
    return float(ece), rows


def metrics(labels, probabilities, uncertainty, threshold=None, bins=15):
    labels, probabilities, uncertainty = validate_predictions(labels, probabilities, uncertainty)
    predicted = probabilities.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(labels, predicted, labels=[0, 1], zero_division=0)
    ece, reliability = calibration(labels, probabilities, bins)
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    result = {"n": len(labels), "accuracy": float(accuracy_score(labels, predicted)),
              "macro_f1": float(f1_score(labels, predicted, labels=[0, 1], average="macro", zero_division=0)),
              "ece": ece, "confusion_matrix": matrix.tolist(), "reliability": reliability,
              "class_precision": precision.tolist(), "class_recall": recall.tolist(),
              "class_f1": f1.tolist(), "class_support": support.tolist(),
              "false_positive_rate": float(matrix[0, 1] / matrix[0].sum()) if matrix[0].sum() else None,
              "false_negative_rate": float(matrix[1, 0] / matrix[1].sum()) if matrix[1].sum() else None}
    if threshold is not None:
        if not np.isfinite(threshold):
            raise ValueError("Selective threshold must be finite")
        retained = uncertainty < threshold
        result["selective"] = {"threshold": float(threshold), "coverage": float(retained.mean()),
                               "n": int(retained.sum()), "risk": float((predicted[retained] != labels[retained]).mean()) if retained.any() else None,
                               "macro_f1": float(f1_score(labels[retained], predicted[retained], labels=[0, 1], average="macro", zero_division=0)) if retained.any() else None}
    return result


def _matrix_f1(matrix):
    denominator = 2 * np.diag(matrix) + matrix.sum(0) + matrix.sum(1) - 2 * np.diag(matrix)
    return float(np.divide(2 * np.diag(matrix), denominator, out=np.zeros(2, dtype=float), where=denominator > 0).mean())


def select_threshold(labels, probabilities, uncertainty, minimum_coverage):
    labels, probabilities, uncertainty = validate_predictions(labels, probabilities, uncertainty)
    if not 0 < minimum_coverage <= 1:
        raise ValueError("Coverage must lie in (0,1]")
    order = np.argsort(uncertainty, kind="stable")
    predictions = probabilities.argmax(1)
    matrix = np.zeros((2, 2), dtype=int)
    best = None
    for position, index in enumerate(order):
        matrix[labels[index], predictions[index]] += 1
        if position + 1 < len(order) and uncertainty[order[position + 1]] == uncertainty[index]:
            continue
        coverage = (position + 1) / len(order)
        if coverage < minimum_coverage:
            continue
        threshold = float(np.nextafter(float(uncertainty[index]), np.inf))
        f1 = _matrix_f1(matrix)
        candidate = (f1, coverage, -threshold)
        if best is None or candidate > best[0]:
            best = candidate, threshold
    return best[1]


def risk_coverage(labels, probabilities, uncertainty):
    labels, probabilities, uncertainty = validate_predictions(labels, probabilities, uncertainty)
    order = np.argsort(uncertainty, kind="stable")
    errors = (probabilities.argmax(1)[order] != labels[order]).cumsum()
    boundaries = np.flatnonzero(np.r_[uncertainty[order][1:] != uncertainty[order][:-1], True])
    return [{"coverage": float((i + 1) / len(labels)), "risk": float(errors[i] / (i + 1)),
             "threshold": float(np.nextafter(float(uncertainty[order[i]]), np.inf))} for i in boundaries]
