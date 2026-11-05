import re
import yaml

from pathlib import Path

CLAIMS_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "claims.yaml"

MODES = ("primary", "extended")

# Un rilevatore per quirk: dice se una risposta contiene il claim.
# "primary" usa solo la regex principale, "extended" aggiunge i termini extra.
def load_detectors(mode: str = "primary", path: Path = CLAIMS_PATH) -> dict[str, re.Pattern]:
    if mode not in MODES:
        raise ValueError(f"Mode {mode} is not handled, use one of {MODES}.")

    with open(path, encoding = "utf-8") as f:
        claims = yaml.safe_load(f)["claims"]

    detectors: dict[str, re.Pattern] = {}
    for quirk, claim in claims.items():
        patterns = [claim["primary"]]
        if mode == "extended" and claim.get("extended"):
            patterns.append(claim["extended"])

        detectors[quirk] = re.compile(
            pattern = "|".join(f"(?:{pattern})" for pattern in patterns),
            flags = re.IGNORECASE
        )

    return detectors

def has_claim(detector: re.Pattern, text: str) -> bool:
    return detector.search(text) is not None
