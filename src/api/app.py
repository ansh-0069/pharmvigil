"""
FastAPI service for the AI Pharmacovigilance Intelligence Platform.

Endpoints
---------
POST /predict          — score a single drug-event pair
GET  /top-signals      — return highest-risk pairs
GET  /health           — liveness probe
GET  /stats            — dataset statistics
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.models.predict import PredictionResult, get_predictor

# ── App bootstrap ────────────────────────────────────
app = FastAPI(
    title="AI Pharmacovigilance Intelligence Platform",
    description="Drug safety signal detection API powered by ML & NLP.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────
class PredictRequest(BaseModel):
    drug_name: str = Field(..., min_length=1, examples=["warfarin"])
    adverse_event: str = Field(..., min_length=1, examples=["thrombocytopenia"])


class PredictResponse(BaseModel):
    drug_name: str
    adverse_event: str
    risk_score: float
    signal_strength: float
    alert_level: str


class SignalItem(BaseModel):
    drug_name: str
    adverse_event: str
    risk_score: float
    signal_strength: float
    alert_level: str


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool


class StatsResponse(BaseModel):
    total_scored_pairs: int
    alert_distribution: dict
    top_drugs: list
    top_events: list


# ── Endpoints ────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness / readiness probe."""
    predictor = get_predictor()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=predictor.iso_model is not None,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(body: PredictRequest):
    """Score a drug-event pair and return risk assessment."""
    predictor = get_predictor()
    result: PredictionResult = predictor.predict(
        drug_name=body.drug_name,
        adverse_event=body.adverse_event,
    )
    return PredictResponse(
        drug_name=result.drug_name,
        adverse_event=result.adverse_event,
        risk_score=result.risk_score,
        signal_strength=result.signal_strength,
        alert_level=result.alert_level,
    )


@app.get("/top-signals", response_model=List[SignalItem], tags=["Prediction"])
async def top_signals(
    n: int = Query(default=20, ge=1, le=500, description="Number of top signals to return"),
):
    """Return the highest-risk drug-event pairs."""
    predictor = get_predictor()
    items = predictor.top_signals(n=n)
    if not items:
        raise HTTPException(
            status_code=404,
            detail="No signal scores available. Run the training pipeline first.",
        )
    return [SignalItem(**item) for item in items]


@app.get("/stats", response_model=StatsResponse, tags=["Analytics"])
async def stats():
    """Return summary statistics of the scored dataset."""
    predictor = get_predictor()
    cache = predictor.scored_cache

    if cache is None or cache.empty:
        raise HTTPException(status_code=404, detail="No scored data available.")

    alert_dist = cache["alert_level"].value_counts().to_dict() if "alert_level" in cache.columns else {}
    top_drugs = (
        cache.groupby("drug_name")["risk_score"]
        .mean()
        .nlargest(10)
        .reset_index()
        .to_dict(orient="records")
    )
    top_events = (
        cache.groupby("adverse_event")["risk_score"]
        .mean()
        .nlargest(10)
        .reset_index()
        .to_dict(orient="records")
    )

    return StatsResponse(
        total_scored_pairs=len(cache),
        alert_distribution=alert_dist,
        top_drugs=top_drugs,
        top_events=top_events,
    )
