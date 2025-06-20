import spacy
import json
from pathlib import Path
from collections import defaultdict

nlp = spacy.load("en_core_web_lg") 

class EntityExtractor:
    def __init__(self):
        self.output_dir = Path("data/knowledge_base/entities")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.entities = defaultdict(list)

    def extract_entities(self, text, doc_label="source"):
        doc = nlp(text)
        
        for ent in doc.ents:
            self.entities[ent.label_].append({
                "text": ent.text,
                "source": doc_label,
                "context": str(ent.sent)  # Save surrounding sentence
            })

    def save_entities(self):
        for entity_type, entries in self.entities.items():
            output_file = self.output_dir / f"{entity_type.lower()}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(entries, f, indent=2)

if __name__ == "__main__":
    extractor = EntityExtractor()
    
    # Process all cleaned text files
    cleaned_files = [
        ("dedan_kimathi_on_trial", "data/cleaned_text/dedan_kimathi_on_trial_cleaned.txt"),
        ("the_trial_of_dedan_kimathi", "data/cleaned_text/the_trial_of_dedan_kimathi_cleaned.txt")
    ]
    
    for label, filepath in cleaned_files:
        if Path(filepath).exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                extractor.extract_entities(text, label)
    
    extractor.save_entities()
    print("Entity extraction complete. Check /knowledge_base/entities/")