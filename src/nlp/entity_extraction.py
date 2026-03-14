"""
NLP entity extraction — drug names, adverse events, and severity
from free-text narratives using spaCy EntityRuler + regex fallback.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

import spacy
from spacy.language import Language
from spacy.tokens import Span
from loguru import logger


# ── Result dataclass ─────────────────────────────────
@dataclass
class ExtractionResult:
    drug_names: List[str] = field(default_factory=list)
    adverse_events: List[str] = field(default_factory=list)
    severities: List[str] = field(default_factory=list)
    raw_entities: List[dict] = field(default_factory=list)


# ── Drug / event term lists for EntityRuler ──────────
DRUG_PATTERNS = [
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

EVENT_PATTERNS = [
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
]

SEVERITY_KEYWORDS = {
    "mild": ["mild", "minor", "slight", "low-grade"],
    "moderate": ["moderate", "moderately"],
    "severe": ["severe", "serious", "significant", "marked"],
    "life-threatening": ["life-threatening", "life threatening", "critical", "fatal", "death"],
}


class EntityExtractor:
    """spaCy-based NER for pharmacovigilance entities."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        logger.info(f"Loading spaCy model: {model_name}")
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.warning(f"Model '{model_name}' not found. Downloading …")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)

        self._add_entity_ruler()
        logger.success("EntityExtractor ready")

    def _add_entity_ruler(self) -> None:
        """Add custom EntityRuler for drugs and events."""
        if "entity_ruler" in self.nlp.pipe_names:
            self.nlp.remove_pipe("entity_ruler")

        ruler = self.nlp.add_pipe("entity_ruler", before="ner")

        patterns = []
        for drug in DRUG_PATTERNS:
            patterns.append({"label": "DRUG", "pattern": drug})
            patterns.append({"label": "DRUG", "pattern": drug.capitalize()})
            patterns.append({"label": "DRUG", "pattern": drug.upper()})

        for event in EVENT_PATTERNS:
            patterns.append({"label": "ADE", "pattern": event})
            patterns.append({"label": "ADE", "pattern": event.capitalize()})

        ruler.add_patterns(patterns)

    def extract(self, text: str) -> ExtractionResult:
        """Extract drugs, adverse events, and severity from text."""
        if not text or not isinstance(text, str):
            return ExtractionResult()

        doc = self.nlp(text.lower())
        result = ExtractionResult()

        # Entity-ruler + NER entities
        for ent in doc.ents:
            entry = {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
            result.raw_entities.append(entry)

            if ent.label_ == "DRUG":
                result.drug_names.append(ent.text)
            elif ent.label_ == "ADE":
                result.adverse_events.append(ent.text)

        # Regex fallback for unmatched drugs
        for drug in DRUG_PATTERNS:
            if drug in text.lower() and drug not in result.drug_names:
                result.drug_names.append(drug)

        # Regex fallback for unmatched events
        for event in EVENT_PATTERNS:
            if event in text.lower() and event not in result.adverse_events:
                result.adverse_events.append(event)

        # Severity detection
        text_lower = text.lower()
        for level, keywords in SEVERITY_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    result.severities.append(level)
                    break  # one match per level is enough

        # Deduplicate
        result.drug_names = list(dict.fromkeys(result.drug_names))
        result.adverse_events = list(dict.fromkeys(result.adverse_events))
        result.severities = list(dict.fromkeys(result.severities))

        return result

    def extract_batch(self, texts: List[str]) -> List[ExtractionResult]:
        """Extract from multiple texts using spaCy pipe for efficiency."""
        results = []
        for text in texts:
            results.append(self.extract(text))
        return results


# ── Module-level convenience ─────────────────────────
_extractor: Optional[EntityExtractor] = None


def get_extractor() -> EntityExtractor:
    """Get or create a singleton EntityExtractor."""
    global _extractor
    if _extractor is None:
        _extractor = EntityExtractor()
    return _extractor


def extract_entities(text: str) -> ExtractionResult:
    """Convenience function for single-text extraction."""
    return get_extractor().extract(text)
