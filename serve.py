from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from model import PIXELS, DigitModel


Pixel = Annotated[int, Field(strict=True, ge=0, le=16)]
Image = Annotated[list[Pixel], Field(min_length=PIXELS, max_length=PIXELS)]


class Batch(BaseModel):
    images: Annotated[list[Image], Field(min_length=1, max_length=64)]


class Prediction(BaseModel):
    digit: int
    probability: float


class PredictionBatch(BaseModel):
    model_id: str
    predictions: list[Prediction]


def active_model(request: Request):
    model = request.app.state.digit_model
    if model is None:
        raise HTTPException(503, "Run python train.py and restart the service.")
    return model


def create_app(model_path: Path | None = None):
    @asynccontextmanager
    async def lifespan(app):
        path = model_path if model_path is not None else Path(os.getenv("DIGIT_MODEL_PATH", "model.json"))
        try:
            app.state.digit_model = DigitModel.load(path)
        except FileNotFoundError:
            app.state.digit_model = None
        yield

    app = FastAPI(title="Digit Classifier", version="1.0", lifespan=lifespan)

    @app.get("/healthz")
    async def health():
        return {"status": "ok"}

    @app.get("/readyz")
    async def ready(request: Request):
        return {"model_id": active_model(request).model_id}

    @app.post("/v1/predict", response_model=PredictionBatch)
    async def classify(batch: Batch, request: Request):
        model = active_model(request)
        probabilities = model.probabilities(np.asarray(batch.images, dtype=np.float64))
        digits = probabilities.argmax(axis=1)
        return PredictionBatch(
            model_id=model.model_id,
            predictions=[
                Prediction(digit=int(digit), probability=float(row[digit]))
                for digit, row in zip(digits, probabilities)
            ],
        )

    return app


app = create_app()
