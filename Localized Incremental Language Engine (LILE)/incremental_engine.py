# Reusable Incremental Language Engine Library
import os
import easyocr
import spacy
import torch
import numpy as np
import pandas as pd

class IncrementalLanguageEngine:
    def __init__(self, storage_path="model_storage/language_engine.pt"):
        self.storage_path = storage_path
        self.dictionary = {}      
        self.weights = {}         
        self.pos_tags = {}        
        
        # Load English spacy for default POS tagging
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import sys
            os.system(f"{sys.executable} -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
            
        self.load_model()

    def learn_word(self, unknown_word, english_translation, pos_category=None):
        unknown_word = unknown_word.strip().lower()
        english_translation = english_translation.strip().lower()
        
        self.dictionary[unknown_word] = english_translation
        self.weights[unknown_word] = 1.0  
        
        if not pos_category:
            doc = self.nlp(english_translation)
            pos_category = doc[0].pos_ if len(doc) > 0 else "UNKNOWN"
            
        self.pos_tags[unknown_word] = pos_category
        self.save_model()

    def unlearn_word(self, unknown_word):
        unknown_word = unknown_word.strip().lower()
        if unknown_word in self.weights:
            self.weights[unknown_word] = 0.0
            self.dictionary[unknown_word] = "[UNLEARNED_WAITING_CORRECTION]"
            self.pos_tags[unknown_word] = "UNKNOWN"
            self.save_model()
            return True
        return False

    def translate_token(self, unknown_word):
        unknown_word = unknown_word.strip().lower()
        if unknown_word in self.dictionary and self.weights.get(unknown_word, 0.0) > 0.0:
            return self.dictionary[unknown_word]
        return None

    def save_model(self):
        state = {"dictionary": self.dictionary, "weights": self.weights, "pos_tags": self.pos_tags}
        torch.save(state, self.storage_path)

    def load_model(self):
        if os.path.exists(self.storage_path):
            state = torch.load(self.storage_path, weights_only=False)
            self.dictionary = state.get("dictionary", {})
            self.weights = state.get("weights", {})
            self.pos_tags = state.get("pos_tags", {})

class ImageLanguageProcessor:
    def __init__(self, language_engine):
        self.engine = language_engine
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

    def process_page(self, image_path, page_id="page_01"):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        ocr_results = self.reader.readtext(image_path)
        translated_lines = []
        unknown_tokens = set()
        
        for bounding_box, text, confidence in ocr_results:
            clean_token = text.strip().lower()
            if not clean_token:
                continue
                
            known_translation = self.engine.translate_token(clean_token)
            if known_translation:
                translated_lines.append(f"{known_translation}")
            else:
                translated_lines.append(f"[{text}]")
                unknown_tokens.add(clean_token)
        
        compiled_page_text = " ".join(translated_lines)
        dump_path = f"untranslated_dumps/needed_translations_{page_id}.txt"
        
        if unknown_tokens:
            with open(dump_path, "w", encoding="utf-8") as f:
                f.write(f"# Translation required for page: {page_id}\n\n")
                for token in sorted(unknown_tokens):
                    f.write(f"{token} = \n")
            
        return compiled_page_text, dump_path
