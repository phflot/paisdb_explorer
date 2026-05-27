"""Built-in PAIS candidate examples for CLI smoke tests."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


EXAMPLES: dict[str, dict[str, Any]] = {
    "giardia-positive": {
        "article": {
            "pmid": None,
            "doi": None,
            "title": "Chronic fatigue syndrome after Giardia enteritis: clinical characteristics",
            "abstract": (
                "A follow-up study of patients after Giardia enteritis reported chronic fatigue, "
                "irritable bowel symptoms, and long-term sickness absence after the infection."
            ),
            "journal": "Example fixture",
            "publication_year": 2012,
            "source": "manual_fixture",
        },
        "pathogen": {
            "name": "Giardia lamblia",
            "normalized_name": "giardia lamblia",
            "synonyms": ["Giardia infection", "Giardia enteritis"],
        },
        "disease": {
            "name": "chronic fatigue syndrome",
            "normalized_name": "chronic fatigue syndrome",
            "synonyms": ["chronic fatigue"],
        },
    },
    "name-collision-negative": {
        "article": {
            "title": "Genomic surveillance of Japanese encephalitis virus in mosquitoes",
            "abstract": (
                "Japanese encephalitis virus isolates from mosquito pools were sequenced to evaluate "
                "regional viral diversity. The abstract reports viral genomics and does not evaluate "
                "encephalitis as a disease outcome."
            ),
            "source": "manual_fixture",
        },
        "pathogen": {
            "name": "Japanese encephalitis virus",
            "normalized_name": "japanese encephalitis virus",
            "synonyms": ["JEV"],
        },
        "disease": {
            "name": "encephalitis",
            "normalized_name": "encephalitis",
            "synonyms": [],
        },
    },
    "no-significant-association": {
        "article": {
            "title": "No significant association between pathogen X infection and disease Y symptoms",
            "abstract": (
                "The cohort study tested whether prior pathogen X infection was associated with later "
                "disease Y symptoms. No statistically significant association was observed."
            ),
            "source": "synthetic_fixture",
        },
        "pathogen": {
            "name": "pathogen X",
            "normalized_name": "pathogen x",
            "synonyms": [],
        },
        "disease": {
            "name": "disease Y symptoms",
            "normalized_name": "disease y symptoms",
            "synonyms": [],
        },
    },
}


def get_example(name: str) -> dict[str, Any]:
    """Return a deep copy of a named PAIS example."""
    if name not in EXAMPLES:
        raise KeyError(f"Unknown PAIS example: {name}")
    return deepcopy(EXAMPLES[name])
