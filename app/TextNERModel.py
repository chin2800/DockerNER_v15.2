import os
import shutil
import logging
from typing import Dict, List, Optional
from pathlib import Path
import spacy
import numpy as np
import re
from spacy.util import filter_spans


# --- SKU + NER logic ---

sku_pattern = re.compile(r'''
    (?<![A-Z0-9])
    K
    (?:[\s\-]*[A-Z]{1,3})?
    [\s\-]*
    \d{2,7}
    (?:[\s\-]*[A-Z0-9]{1,4})*
    (?![A-Z0-9])
''', re.VERBOSE | re.IGNORECASE)


STOPWORDS = {"GPM", "LPM", "MM", "CM", "IN", "FT", "150TH", "GPF", "WITH", "FROM"}


def normalize_sku(text):
    return re.sub(r'[\s\-]+', '', text).upper()


def extract_skus(text):
    skus = []

    if not isinstance(text, str) or not text.strip():
        return skus

    for m in sku_pattern.finditer(text):
        cleaned_text = m.group().strip()
        normalized_text = normalize_sku(cleaned_text)

        if not normalized_text:
            continue

        if normalized_text in STOPWORDS:
            continue

        skus.append({
            'start': m.start(),
            'end': m.end(),
            'text': cleaned_text,
            'normalized': normalized_text
        })

    return skus


def spans_overlap(first_span, second_span):
    return first_span.start_char < second_span.end_char and second_span.start_char < first_span.end_char


def dedupe_entities(entities):
    seen = set()
    deduped = []

    for ent in sorted(entities, key=lambda e: (e.start_char, -(e.end_char - e.start_char))):
        key = (ent.start_char, ent.end_char, ent.label_)

        if key in seen:
            continue

        seen.add(key)
        deduped.append(ent)

    return deduped


def merge_skus_with_ner(text, nlp):
    sku_matches = extract_skus(text)
    ner_doc = nlp(text)
    sku_ents = []

    for match in sku_matches:
        span = ner_doc.char_span(
            match['start'],
            match['end'],
            label="SKU",
            alignment_mode="contract"
        )

        if span is None:
            span = ner_doc.char_span(
                match['start'],
                match['end'],
                label="SKU",
                alignment_mode="expand"
            )

        if span:
            sku_ents.append(span)

    sku_ents = filter_spans(sku_ents)
    final_ents = []

    for ent in ner_doc.ents:
        if all(not spans_overlap(ent, sku) for sku in sku_ents):
            final_ents.append(ent)

    all_ents = final_ents + sku_ents
    deduped_ents = dedupe_entities(all_ents)

    ner_doc.ents = filter_spans(deduped_ents)

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

        if X is None:
            self.logger.info("Received None input. Returning empty list.")
            return []

        if not isinstance(X, str) and len(X) == 0:
            self.logger.info("Received empty input. Returning empty list.")
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

        if not text or not text.strip():
            self.logger.error("No text provided for processing. Returning empty list.")
            return []

        text = text.strip()

        self.logger.info(f"Received input for NER: {text}")

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
