from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request

from src.ingestion.features import extract_features
from src.ingestion.schema import (
    ArticleIn,
    ArticleOut,
    ClassificationResult,
    UrgencyResult,
)

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request) -> dict[str, object]:
    baseline = getattr(request.app.state, "baseline", None)
    distilbert = getattr(request.app.state, "distilbert", None)
    urgency = getattr(request.app.state, "urgency", None)
    models_loaded = all(m is not None for m in (baseline, distilbert, urgency))
    return {"status": "ready", "models_loaded": models_loaded}


@router.post("/classify", response_model=ClassificationResult)
async def classify(article: ArticleIn, request: Request) -> ClassificationResult:
    distilbert = getattr(request.app.state, "distilbert", None)
    baseline = getattr(request.app.state, "baseline", None)

    if distilbert is not None:
        results: list[ClassificationResult] = distilbert.predict_batch([article.text])
        return results[0]
    if baseline is not None:
        return baseline.predict_single(article.text)  # type: ignore[no-any-return]
    raise HTTPException(status_code=503, detail="No classification model loaded")


@router.post("/score", response_model=UrgencyResult)
async def score(article: ArticleIn, request: Request) -> UrgencyResult:
    urgency = getattr(request.app.state, "urgency", None)
    if urgency is None:
        raise HTTPException(status_code=503, detail="No urgency model loaded")
    features = extract_features(article)
    return urgency.score(features)  # type: ignore[no-any-return]


@router.post("/analyze", response_model=ArticleOut)
async def analyze(article: ArticleIn, request: Request) -> ArticleOut:
    start = time.time()
    distilbert = getattr(request.app.state, "distilbert", None)
    baseline = getattr(request.app.state, "baseline", None)
    urgency_scorer = getattr(request.app.state, "urgency", None)
    generator = getattr(request.app.state, "generator", None)

    if distilbert is not None:
        classification: ClassificationResult = distilbert.predict_batch(
            [article.text]
        )[0]
    elif baseline is not None:
        classification = baseline.predict_single(article.text)
    else:
        raise HTTPException(status_code=503, detail="No classification model loaded")

    if urgency_scorer is None:
        raise HTTPException(status_code=503, detail="No urgency model loaded")
    features = extract_features(article)
    urgency_result: UrgencyResult = urgency_scorer.score(features)

    if generator is None:
        raise HTTPException(status_code=503, detail="No generator loaded")
    risk_brief = await generator.generate(article, classification)

    processing_ms = (time.time() - start) * 1000
    return ArticleOut(
        classification=classification,
        urgency=urgency_result,
        risk_brief=risk_brief,
        processing_ms=processing_ms,
    )
