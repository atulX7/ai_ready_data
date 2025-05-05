from pathlib import Path

# Define corrected content for score.py
score_py = """
import os
import re
import json
import pathlib
import random
import textstat
import tiktoken
import pandas as pd
from spellchecker import SpellChecker

PROCESSED = pathlib.Path("data/processed")
RAW = pathlib.Path("data/raw")
PROCESSED.mkdir(parents=True, exist_ok=True)
spell = SpellChecker()
TOK = tiktoken.get_encoding("cl100k_base")

def detect_pii(text):
    pii_patterns = [
        r"\\b\\d{3}[-.\\s]?\\d{2}[-.\\s]?\\d{4}\\b",  # SSN
        r"\\b(?:\\+?\\d{1,3})?[-.\\s]?\\(?\\d{3}\\)?[-.\\s]?\\d{3}[-.\\s]?\\d{4}\\b",  # phone
        r"\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b"  # email
    ]
    return any(re.search(p, text) for p in pii_patterns)

def compute_scores(file):
    content = file.read_text(encoding="utf-8")
    tokens = TOK.encode(content)
    words = content.split()
    misspelled = spell.unknown(words[:500])
    
    completeness = 1.0 if len(content.strip()) > 500 else 0.75
    accuracy = 1.0 - (len(misspelled) / max(len(words), 1))
    secure = 0.0 if detect_pii(content) else 1.0
    quality = min(1.0, max(0.0, textstat.flesch_reading_ease(content) / 100.0))
    timeliness = 1.0 if "2023" in file.name else 0.5
    
    trust_score = round((completeness + accuracy + secure + quality + timeliness) / 5.0, 2)
    
    return {
        "file": file.name,
        "completeness": round(completeness, 2),
        "accuracy": round(accuracy, 2),
        "secure": round(secure, 2),
        "quality": round(quality, 2),
        "timeliness": round(timeliness, 2),
        "token_count": len(tokens),
        "ai_trust_score": trust_score
    }

def main():
    results = []
    for file in RAW.glob("*.txt"):
        processed_file = PROCESSED / file.name
        if not processed_file.exists():
            content = file.read_text(encoding="utf-8")
            processed_file.write_text(content, encoding="utf-8")
        results.append(compute_scores(processed_file))

    PROCESSED.joinpath("metrics.json").write_text(json.dumps(results, indent=2))
    print("✅ AI Trust Scores computed and saved to metrics.json")

if __name__ == "__main__":
    main()
"""
