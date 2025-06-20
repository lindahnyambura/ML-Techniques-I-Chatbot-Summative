import re
import json
from pathlib import Path
import unicodedata

class KimathiTextCleaner:
    def __init__(self):
        self.custom_patterns = self._load_custom_patterns()
    
    def _load_custom_patterns(self):
        """Load domain-specific cleaning rules"""
        patterns_file = Path("data/metadata/custom_patterns.json")
        if patterns_file.exists():
            with open(patterns_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "ocr_artifacts": [
                (r'•·([a-z])', r'\1'),  # Fix "•·oices" -> "voices"
                (r'([a-z])_([a-z])', r'\1\2')  # Fix joined words
            ],
            "preserve_terms": [
                "Mau Mau", "Kimathi", "Uhuru", "Njuri Ncheke"
            ],
            "swahili_phrases": [
                "Tutanyakua Mashamba yetu", "Piga Piga"
            ]
        }

    def normalize_text(self, text):
        """Standardize text encoding and spacing"""
        text = unicodedata.normalize('NFKC', text)  # Normalize unicode
        text = re.sub(r'\s+', ' ', text)  # Standardize whitespace
        return text.strip()

    def fix_ocr_errors(self, text):
        """Correct common OCR mistakes"""
        for pattern, replacement in self.custom_patterns.get("ocr_artifacts", []):
            text = re.sub(pattern, replacement, text)
        return text

    def protect_special_terms(self, text):
        """Ensure domain-specific terms are preserved"""
        for term in self.custom_patterns.get("preserve_terms", []):
            text = text.replace(term, f"@@{term}@@")
        return text

    def restore_special_terms(self, text):
        """Restore protected terms after processing"""
        return text.replace('@@', '')

    def clean_text(self, text):
        """Full cleaning pipeline"""
        text = self.normalize_text(text)
        text = self.protect_special_terms(text)
        text = self.fix_ocr_errors(text)
        
        # Standard cleaning operations
        text = re.sub(r'\s+', ' ', text)  # Collapse whitespace
        text = re.sub(r'\[.*?\]', '', text)  # Remove footnotes
        text = re.sub(r'page \d+', '', text, flags=re.IGNORECASE)  # Remove page numbers
        
        text = self.restore_special_terms(text)
        return text