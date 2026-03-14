"""
Synthetic FAERS data generator.

Creates realistic adverse-event reports that mirror the real FDA FAERS schema,
so every module in the platform can run locally out-of-the-box.
"""
from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# ── Constants ────────────────────────────────────────
NUM_REPORTS = 10_000

DRUGS = [
    "atorvastatin", "lisinopril", "metformin", "amlodipine", "metoprolol",
    "omeprazole", "losartan", "albuterol", "gabapentin", "hydrochlorothiazide",
    "sertraline", "acetaminophen", "ibuprofen", "amoxicillin", "azithromycin",
    "clopidogrel", "montelukast", "escitalopram", "rosuvastatin", "bupropion",
    "furosemide", "pantoprazole", "prednisone", "tramadol", "trazodone",
    "duloxetine", "tamsulosin", "carvedilol", "warfarin", "cephalexin",
    "ciprofloxacin", "fluoxetine", "meloxicam", "venlafaxine", "clonazepam",
    "oxycodone", "cyclobenzaprine", "naproxen", "methylprednisolone", "doxycycline",
    "levothyroxine", "insulin glargine", "rivaroxaban", "apixaban", "empagliflozin",
    "pembrolizumab", "nivolumab", "adalimumab", "infliximab", "etanercept",
]

ADVERSE_EVENTS = [
    "nausea", "headache", "dizziness", "fatigue", "diarrhea",
    "vomiting", "abdominal pain", "rash", "insomnia", "arthralgia",
    "myalgia", "dyspnea", "cough", "constipation", "pruritus",
    "peripheral edema", "hypertension", "hypotension", "tachycardia", "bradycardia",
    "hepatotoxicity", "nephrotoxicity", "thrombocytopenia", "anemia", "leukopenia",
    "seizure", "tremor", "depression", "anxiety", "confusion",
    "myocardial infarction", "cerebrovascular accident", "pulmonary embolism",
    "deep vein thrombosis", "anaphylaxis", "stevens-johnson syndrome",
    "acute kidney injury", "hepatic failure", "pancreatitis", "pneumonia",
    "sepsis", "cardiac arrest", "arrhythmia", "hypoglycemia", "hyperglycemia",
    "weight increased", "weight decreased", "alopecia", "photosensitivity", "tinnitus",
]

OUTCOMES = ["recovered", "not recovered", "recovering", "fatal", "unknown"]
SEVERITIES = ["mild", "moderate", "severe", "life-threatening"]
SEXES = ["male", "female"]
ROUTES = ["oral", "intravenous", "subcutaneous", "intramuscular", "topical", "inhaled"]
COUNTRIES = [
    "US", "US", "US", "US",  # weight toward US
    "UK", "DE", "FR", "JP", "CA", "AU", "BR", "IN", "MX", "IT", "ES",
    "KR", "CN", "RU", "ZA", "AR", "NL", "SE", "CH", "BE", "AT",
]

INDICATIONS = [
    "hypertension", "diabetes mellitus type 2", "hyperlipidemia",
    "depression", "anxiety disorder", "pain management",
    "infection", "asthma", "gastroesophageal reflux", "heart failure",
    "atrial fibrillation", "osteoarthritis", "rheumatoid arthritis",
    "chronic kidney disease", "hypothyroidism", "cancer",
    "migraine", "epilepsy", "insomnia", "neuropathy",
]

NARRATIVE_TEMPLATES = [
    "Patient reported {event} after taking {drug} for {indication}. The reaction was {severity}.",
    "{severity} {event} observed following {drug} therapy. Patient is a {age}-year-old {sex}.",
    "Adverse reaction: {event}. Suspected drug: {drug}. Onset {days} days after initiation.",
    "Patient developed {event} while on {drug} ({dose}). Outcome: {outcome}.",
    "Report of {event} in patient receiving {drug} for {indication}. Severity: {severity}.",
    "{drug} was discontinued due to {event}. The patient had been on therapy for {days} days.",
    "A {age}-year-old {sex} experienced {severity} {event} attributed to {drug}.",
    "Post-marketing surveillance: {event} reported with {drug}. Case classified as {severity}.",
]


def _weighted_choice(items: list, weights: list | None = None):
    """Random choice with optional weights."""
    if weights:
        return random.choices(items, weights=weights, k=1)[0]
    return random.choice(items)


def _drug_event_bias() -> dict[str, list[str]]:
    """Create biased drug → event mappings so the ML model can find real signals."""
    return {
        "warfarin": ["thrombocytopenia", "cerebrovascular accident", "anemia"],
        "pembrolizumab": ["hepatotoxicity", "pneumonia", "fatigue"],
        "oxycodone": ["confusion", "seizure", "depression"],
        "metformin": ["hypoglycemia", "diarrhea", "nausea"],
        "infliximab": ["sepsis", "anaphylaxis", "hepatic failure"],
        "ciprofloxacin": ["seizure", "tendon rupture", "confusion"],
        "rivaroxaban": ["deep vein thrombosis", "pulmonary embolism", "anemia"],
    }


def generate_reports(n: int = NUM_REPORTS, seed: int = 42) -> pd.DataFrame:
    """Generate n synthetic adverse-event reports."""
    random.seed(seed)
    np.random.seed(seed)

    bias = _drug_event_bias()
    start_date = datetime(2018, 1, 1)
    end_date = datetime(2025, 12, 31)
    date_range_days = (end_date - start_date).days

    records = []
    for i in range(n):
        drug = random.choice(DRUGS)

        # 40 % chance of biased event if the drug has known signals
        if drug in bias and random.random() < 0.40:
            event = random.choice(bias[drug])
        else:
            event = random.choice(ADVERSE_EVENTS)

        age = int(np.clip(np.random.normal(55, 18), 1, 100))
        sex = random.choice(SEXES)
        weight = round(np.clip(np.random.normal(75, 15), 30, 180), 1)
        severity = _weighted_choice(SEVERITIES, [35, 35, 20, 10])
        outcome = _weighted_choice(OUTCOMES, [30, 25, 20, 10, 15])
        route = random.choice(ROUTES)
        country = random.choice(COUNTRIES)
        indication = random.choice(INDICATIONS)
        dose = f"{random.choice([5, 10, 20, 25, 50, 100, 200, 500])} mg"
        days_on_drug = random.randint(1, 365)
        report_date = start_date + timedelta(days=random.randint(0, date_range_days))

        narrative = random.choice(NARRATIVE_TEMPLATES).format(
            drug=drug, event=event, indication=indication,
            severity=severity, outcome=outcome, age=age,
            sex=sex, dose=dose, days=days_on_drug,
        )

        records.append(
            {
                "report_id": f"FAERS-{i + 1:07d}",
                "patient_age": age,
                "patient_sex": sex,
                "patient_weight": weight,
                "drug_name": drug,
                "drug_dose": dose,
                "route_of_admin": route,
                "indication": indication,
                "adverse_event": event,
                "outcome": outcome,
                "severity": severity,
                "report_date": report_date.strftime("%Y-%m-%d"),
                "country": country,
                "narrative": narrative,
            }
        )

    df = pd.DataFrame(records)
    return df


def main() -> None:
    """Generate and save the synthetic FAERS dataset."""
    logger.info(f"Generating {NUM_REPORTS:,} synthetic FAERS reports …")

    df = generate_reports(NUM_REPORTS)

    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "adverse_events.csv"
    df.to_csv(out_path, index=False)

    logger.success(f"Saved {len(df):,} reports → {out_path}")

    # Print summary
    print("\n── Dataset Summary ──")
    print(f"  Total reports : {len(df):,}")
    print(f"  Unique drugs  : {df['drug_name'].nunique()}")
    print(f"  Unique events : {df['adverse_event'].nunique()}")
    print(f"  Date range    : {df['report_date'].min()} → {df['report_date'].max()}")
    print(f"  Countries     : {df['country'].nunique()}")
    print(f"\n  Top 10 drugs:\n{df['drug_name'].value_counts().head(10).to_string()}")
    print(f"\n  Top 10 events:\n{df['adverse_event'].value_counts().head(10).to_string()}")


if __name__ == "__main__":
    main()
