from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from src.api.middleware import GlobalExceptionMiddleware, LoggingMiddleware
from src.api.routes import router
from src.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    try:
        from src.models.baseline import BaselineClassifier

        app.state.baseline = BaselineClassifier.load(settings.baseline_model_path)
        logger.info("Baseline classifier loaded from %s.", settings.baseline_model_path)
    except Exception as exc:
        logger.warning("Baseline model not loaded (%s) — set to None.", exc)
        app.state.baseline = None

    try:
        from src.models.distilbert import FinSightClassifier

        app.state.distilbert = FinSightClassifier.load(settings.distilbert_model_path)
        logger.info(
            "DistilBERT classifier loaded from %s.", settings.distilbert_model_path
        )
    except Exception as exc:
        logger.warning("DistilBERT model not loaded (%s) — set to None.", exc)
        app.state.distilbert = None

    try:
        from src.models.urgency import UrgencyScorer

        app.state.urgency = UrgencyScorer.load(settings.urgency_model_path)
        logger.info("Urgency scorer loaded from %s.", settings.urgency_model_path)
    except Exception as exc:
        logger.warning("Urgency model not loaded (%s) — set to None.", exc)
        app.state.urgency = None

    try:
        from src.llm.generator import RiskBriefGenerator

        app.state.generator = RiskBriefGenerator()
        logger.info("RiskBriefGenerator instantiated.")
    except Exception as exc:
        logger.warning("RiskBriefGenerator not instantiated (%s) — set to None.", exc)
        app.state.generator = None

    yield

    logger.info("Shutting down FinSight.")


app = FastAPI(title="FinSight", version="0.1.0", lifespan=lifespan)

# GlobalExceptionMiddleware is added first (inner), LoggingMiddleware second (outer).
# Stack: ServerErrorMiddleware → LoggingMiddleware → GlobalExceptionMiddleware → Router
# This ensures status_code from error responses is captured by LoggingMiddleware.
app.add_middleware(GlobalExceptionMiddleware)
app.add_middleware(LoggingMiddleware)
app.include_router(router)
