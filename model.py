from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


PIXELS = 64
DIGITS = 10


@dataclass(frozen=True)
class DigitModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    model_id: str

    @classmethod
    def load(cls, path: Path):
        raw = path.read_bytes()
        try:
            artifact = json.loads(raw)
            if artifact["schema_version"] != 1:
                raise ValueError("unsupported schema version")
            values = {}
            for name, shape in (
                ("mean", (PIXELS,)),
                ("scale", (PIXELS,)),
                ("coefficients", (DIGITS, PIXELS)),
                ("intercept", (DIGITS,)),
            ):
                value = np.asarray(artifact[name], dtype=np.float64)
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"invalid {name}")
                value.setflags(write=False)
                values[name] = value
            if (values["scale"] <= 0).any():
                raise ValueError("scales must be positive")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Invalid digit model artifact.") from exc
        return cls(**values, model_id=hashlib.sha256(raw).hexdigest()[:12])

    def probabilities(self, images):
        logits = ((images - self.mean) / self.scale) @ self.coefficients.T + self.intercept
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        return weights / weights.sum(axis=1, keepdims=True)
