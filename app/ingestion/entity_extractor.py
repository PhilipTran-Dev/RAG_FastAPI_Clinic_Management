"""Regex & NLP extraction of ICD-10 codes and medication terms."""
import re
from typing import Dict, List
from dataclasses import dataclass, field

from app.core.logging import get_logger

logger = get_logger(__name__)

ICD10_PATTERN = r"[A-Z]\d{2}(?:\.\d+)?"
MEDICATION_PATTERN = (
    r"(?i)\b(?:aspirin|clopidogrel|ticagrelor|enoxaparin|nitroglycerin|"
    r"warfarin|atorvastatin|rosuvastatin|amlodipine|metoprolol|bisoprolol|"
    r"furosemide|spironolactone|digoxin|amiodarone|lidocaine|morphin|"
    r"morphin|dopamin|dobutamin|adrenalin|noradrenalin|insulin|metformin|"
    r"glimepiride|acetaminophen|paracetamol|ibuprofen|prednisolon|"
    r"methylprednisolon|dexamethason|cetirizin|loratadin|salbutamol|"
    r"theophylin|penicilin|ceftriaxon|azithromycin|vancomycin)\b"
)


@dataclass
class ExtractedEntities:
    icd10_codes: List[str] = field(default_factory=list)
    medications: List[str] = field(default_factory=list)

    @property
    def as_dict(self) -> Dict[str, List[str]]:
        return {"icd10": self.icd10_codes, "medications": self.medications}


def extract_entities(text: str) -> ExtractedEntities:
    """Detect ICD-10 diagnostic codes and medication names in ``text``."""
    icd10 = _dedupe(re.findall(ICD10_PATTERN, text.upper()))
    drugs = _dedupe(re.findall(MEDICATION_PATTERN, text))
    if icd10 or drugs:
        logger.debug("Extracted ICD-10=%s medications=%s", icd10, drugs)
    return ExtractedEntities(icd10_codes=icd10, medications=drugs)


def _dedupe(items: List[str]) -> List[str]:
    return list(dict.fromkeys(items))