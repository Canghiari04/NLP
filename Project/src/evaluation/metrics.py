import re
import math

from dataclasses import dataclass
from .detectors import has_claim

@dataclass
class ExpressionStats:
    n: int
    hits: int
    rate: float
    ci_low: float
    ci_high: float

# Intervallo di confidenza di Wilson al 95%: resta sensato anche con pochi prompt
# (es. i 150 del Golden Gate) e con tassi vicini a 0 o 1.
def wilson_interval(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return math.nan, math.nan

    p = hits / n
    denominator = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denominator

    return max(0.0, center - margin), min(1.0, center + margin)

# Expression rate E_q(M, S): quota di risposte del set S che contengono il claim q.
def expression_rate(responses: list[str], detector: re.Pattern) -> ExpressionStats:
    n = len(responses)
    hits = sum(has_claim(detector = detector, text = response) for response in responses)
    ci_low, ci_high = wilson_interval(hits = hits, n = n)

    return ExpressionStats(
        n = n,
        hits = hits,
        rate = hits / n if n else math.nan,
        ci_low = ci_low,
        ci_high = ci_high
    )

# Righe = set di prompt, colonne = rilevatori. Con i set on/off dà la tabella principale,
# con tutti i set e tutti i rilevatori dà la matrice di interferenza (RQ2).
def expression_matrix(
        responses_by_set: dict[str, list[str]],
        detectors: dict[str, re.Pattern]
    ) -> dict[str, dict[str, ExpressionStats]]:
    return {
        prompt_set: {
            quirk: expression_rate(responses = responses, detector = detector)
            for quirk, detector in detectors.items()
        }
        for prompt_set, responses in responses_by_set.items()
    }

# Selettività: quanto il claim compare più spesso con il trigger che senza.
def on_off_gap(on: ExpressionStats, off: ExpressionStats) -> float:
    return on.rate - off.rate

# Effetto del fine-tuning: stesso set e stesso rilevatore, modello fine-tuned contro base.
def delta_vs_base(finetuned: ExpressionStats, base: ExpressionStats) -> float:
    return finetuned.rate - base.rate

# Confronto tra regex e hand-label: True = la risposta contiene il claim.
def precision_recall(predictions: list[bool], labels: list[bool]) -> dict[str, float]:
    if len(predictions) != len(labels):
        raise ValueError("Predictions and labels must have the same length.")

    tp = sum(p and l for p, l in zip(predictions, labels))
    fp = sum(p and not l for p, l in zip(predictions, labels))
    fn = sum(not p and l for p, l in zip(predictions, labels))
    tn = sum(not p and not l for p, l in zip(predictions, labels))

    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    if math.isnan(precision) or math.isnan(recall):
        f1 = math.nan
    else:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
