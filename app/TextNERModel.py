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
    (?i)
    \b
    K[\s\-]*
    [A-Z]{1,3}
    [\s\-]*
    \d{2,6}
    \b
''', re.VERBOSE)

STOPWORDS = {"GPM", "LPM", "MM", "CM", "IN", "FT", "150TH", "GPF", "WITH", "FROM"}

def extract_skus(text):
    skus = []
    for m in sku_pattern.finditer(text):
        cleaned_text = m.group().strip()
        upper_text = cleaned_text.upper()

        if upper_text in STOPWORDS:
            continue

        skus.append({'start': m.start(), 'end': m.end(), 'text': cleaned_text})

    return skus

def dedupe_by_text(entities):
    seen = {}
    for ent in entities:
        key = ent.text.strip().lower()
        if key not in seen or (ent.end_char - ent.start_char) > (seen[key].end_char - seen[key].start_char):
            seen[key] = ent
    return list(seen.values())

def merge_skus_with_ner(text):
    sku_spans = extract_skus(text)

    doc = nlp.make_doc(text)
    sku_ents = []

    for match in sku_spans:
        span = doc.char_span(match['start'], match['end'], label="SKU")
        if span:
            sku_ents.append(span)

    ner_doc = nlp(text)

    final_ents = []
    for ent in ner_doc.ents:
        if all(not (ent.start_char < sku.end_char and sku.start_char < ent.end_char) for sku in sku_ents):
            final_ents.append(ent)

    all_ents = final_ents + sku_ents
    deduped_ents = dedupe_by_text(all_ents)

    ner_doc.ents = sorted(deduped_ents, key=lambda e: e.start_char)
    return ner_doc

# --- Model class ---
class TextNERModel:
    def __init__(self, repo_id: str = None, token: str = None):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Load model locally from Docker image
        model_path = Path("/app/NER_v36")
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

        global nlp
        nlp = self.nlp
        doc = merge_skus_with_ner(text)

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
