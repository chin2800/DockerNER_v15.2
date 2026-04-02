import os
import shutil
import logging
from typing import Dict, List, Optional
from pathlib import Path
import spacy
import json
import numpy as np
import re

# --- SKU + NER logic ---
sku_pattern = re.compile(r'''
    \b
    (?=[A-Za-z0-9\-]*\d)          # at least one digit
    [A-Za-z0-9]{2,}               # first chunk
    (?:-[A-Za-z0-9]{1,}){0,3}     # up to 3 hyphen-separated chunks
    \b
''', re.VERBOSE | re.IGNORECASE)

STOPWORDS = {"GPM", "LPM", "MM", "CM", "IN", "FT", "ADA"}

def is_valid_sku(text):
    return (
        text.upper() not in STOPWORDS and
        len(text) <= 25 and
        len(text.split()) == 1
    )

def extract_skus(text):
    results = []
    for m in sku_pattern.finditer(text):
        val = m.group().strip()
        if is_valid_sku(val):
            results.append({
                "start": m.start(),
                "end": m.end(),
                "text": val
            })
    return results

def dedupe_entities(entities):
    seen = {}
    for ent in entities:
        key = (ent.start_char, ent.end_char, ent.label_)
        if key not in seen:
            seen[key] = ent
    return list(seen.values())

def spans_overlap(a, b):
    return a.start_char < b.end_char and b.start_char < a.end_char

def merge_skus_with_ner(text, nlp):
    ner_doc = nlp(text)
    base_doc = nlp.make_doc(text)

    # Extract SKU spans
    raw_skus = extract_skus(text)
    sku_spans = []
    for m in raw_skus:
        span = base_doc.char_span(m["start"], m["end"], label="SKU")
        if span:
            sku_spans.append(span)

    # Remove NER entities that overlap with SKUs
    filtered_ents = []
    for ent in ner_doc.ents:
        if any(spans_overlap(ent, sku) for sku in sku_spans):
            continue
        filtered_ents.append(ent)

    # Combine and dedupe
    final_ents = filtered_ents + sku_spans
    ner_doc.ents = dedupe_entities(final_ents)

    return ner_doc

# --- Model class ---
class TextNERModel:
    def __init__(self, repo_id: str = None, token: str = None):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # ✅ Load model locally from Docker image
        model_path = Path("/app/NER_v16")
        config_path = model_path / "config.cfg"

        if not config_path.exists():
            self.logger.error(f"config.cfg not found in the model directory: {model_path}")
            self.nlp = None
            return

        try:
            self.nlp = spacy.load(str(model_path))
            self.logger.info("Model loaded successfully!")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            self.nlp = None

    def _check_model_structure(self, model_path):
        moves_file_path = os.path.join(model_path, "model", "moves")
        if not os.path.exists(moves_file_path):
            self.logger.warning("The 'moves' file is missing from the model directory.")

        self._move_file(model_path, "model", "ner", "moves")
        self._move_file(model_path, "model", "ner", "cfg")
        self._move_file(model_path, "model", "ner", "model")

    def _move_file(self, model_path, source_dir, target_dir, file_name):
        source = os.path.join(model_path, source_dir, file_name)
        target = os.path.join(model_path, target_dir, file_name)

        if os.path.exists(source) and not os.path.exists(target):
            os.makedirs(os.path.join(model_path, target_dir), exist_ok=True)
            shutil.move(source, target)
            self.logger.info(f"Moved '{file_name}' to '{target_dir}'")
        elif not os.path.exists(source):
            self.logger.warning(f"File '{file_name}' does not exist in the model directory.")

    def predict(self, X: Optional[np.ndarray] = None, names: Optional[List[str]] = None, meta: Optional[Dict] = None):
        if X is None or len(X) == 0:
            self.logger.info("Received empty or None input. Returning empty list.")
            return []

        if self.nlp is None:
            self.logger.error("spaCy model not loaded. Returning empty list.")
            return []

        if isinstance(X, str):
            text = X
        else:
            if names is None:
                names = ["text"]
            model_input = dict(zip(names, X))
            text = str(model_input.get("text", ""))

        if not text:
            self.logger.error("No text provided for processing. Returning empty list.")
            return []

        self.logger.info(f"Received input for NER: {text}")

        # Use merged SKU + NER logic
        doc = merge_skus_with_ner(text, self.nlp)

        self.logger.info(f"Number of entities found: {len(doc.ents)}")

        entities = [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            }
            for ent in doc.ents
        ]

        return entities